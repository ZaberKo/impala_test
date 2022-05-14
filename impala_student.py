import copy
import pickle
from ref.impala import *
from ray.rllib.execution.common import _get_shared_metrics
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import merge_dicts
from pathlib import Path

STUDENT_DEFAULT_CONFIG=merge_dicts(DEFAULT_CONFIG,{
    "teacher_checkpoint": "",
    "teacher_config": {
        "num_workers": 1
    },
})

class ImpalaStudentTrainer(ImpalaTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return STUDENT_DEFAULT_CONFIG
    def setup(self, config: PartialTrainerConfigDict):
        super().setup(config)

        # create teacher workers

        self.teacher_config = merge_dicts(self.config, copy.deepcopy(self.config["teacher_config"]))

        self.teacher_workers=WorkerSet(
                env_creator=self.env_creator,
                validate_env=self.validate_env,
                policy_class=self.get_default_policy_class(self.teacher_config),
                trainer_config=self.teacher_config,
                num_workers=self.teacher_config["num_workers"],
                local_worker=True,
                logdir=self.logdir,
            )

        # restore teacher workers from checkpoint
        teacher_checkpoint=Path(self.config["teacher_checkpoint"]).expanduser()
        with teacher_checkpoint.open("rb") as f:
            teacher_state=pickle.load(f)
        
        teacher_worker_state=teacher_state["worker"] # contains model weights
        self.teacher_workers.local_worker().restore(teacher_worker_state)
        teacher_worker_state_ref = ray.put(teacher_worker_state)
        for r in self.teacher_workers.remote_workers():
            r.restore.remote(teacher_worker_state_ref)

       

    def _kwargs_for_execution_plan(self):
        kwargs= super()._kwargs_for_execution_plan()
        kwargs["teacher_workers"]=self.teacher_workers
        return kwargs

    @staticmethod
    def execution_plan(workers, config, **kwargs):
        teacher_workers=kwargs["teacher_workers"]


        train_batches = gather_experiences_directly(teacher_workers, config)

        # Start the learner thread.
        learner_thread = make_learner_thread(workers.local_worker(), config)
        learner_thread.start()

        # This sub-flow sends experiences to the learner.
        enqueue_op = train_batches \
            .for_each(Enqueue(learner_thread.inqueue))
        
        # !!!No need to update teacher workers.
        # if workers.remote_workers():
        #     enqueue_op = enqueue_op.zip_with_source_actor() \
        #         .for_each(BroadcastUpdateLearnerWeights(
        #             learner_thread, workers,
        #             broadcast_interval=config["broadcast_interval"]))

        def record_steps_trained(item):
            count, fetches = item
            metrics = _get_shared_metrics()
            # Manually update the steps trained counter since the learner
            # thread is executing outside the pipeline.
            metrics.counters[STEPS_TRAINED_THIS_ITER_COUNTER] += count
            metrics.counters[STEPS_TRAINED_COUNTER] += count
            return item

        # This sub-flow updates the steps trained counter based on learner
        # output.
        dequeue_op = Dequeue(
            learner_thread.outqueue, check=learner_thread.is_alive) \
            .for_each(record_steps_trained)


        merged_op = Concurrently(
            [enqueue_op, dequeue_op], mode="async", output_indexes=[1])
        # merged_op = Concurrently(
        #         [enqueue_op, dequeue_op], mode="round_robin", output_indexes=[1],
        #         round_robin_weights=[1,"*"]
        #     )

        # Callback for APPO to use to update KL, target network periodically.
        # The input to the callback is the learner fetches dict.
        if config["after_train_step"]:
            merged_op = merged_op.for_each(lambda t: t[1]).for_each(
                config["after_train_step"](workers, config))
        
        def reset_trained_timesteps(item):
            metrics = _get_shared_metrics()
            # reset
            metrics.counters[STEPS_TRAINED_THIS_ITER_COUNTER]=0
            return item

        return StandardMetricsReporting(merged_op, workers, config) \
            .for_each(learner_thread.add_learner_metrics) \
            .for_each(reset_trained_timesteps)

    @classmethod
    def default_resource_request(cls, config):
        cf = dict(cls.get_default_config(), **config)

        eval_config = cf["evaluation_config"]

        teacher_config=cf["teacher_config"]

        # Return PlacementGroupFactory containing all needed resources
        # (already properly defined as device bundles).
        return PlacementGroupFactory(
            bundles=[{
                # Driver + Aggregation Workers:
                # Force to be on same node to maximize data bandwidth
                # between aggregation workers and the learner (driver).
                # Aggregation workers tree-aggregate experiences collected
                # from RolloutWorkers (n rollout workers map to m
                # aggregation workers, where m < n) and always use 1 CPU
                # each.
                "CPU": cf["num_cpus_for_driver"] +
                cf["num_aggregation_workers"],
                "GPU": 0 if cf["_fake_gpus"] else cf["num_gpus"],
            }] + [
                {
                    # RolloutWorkers on teacher workers
                    "CPU": teacher_config.get("num_cpus_per_worker",
                                           cf["num_cpus_per_worker"]),
                    "GPU": teacher_config.get("num_gpus_per_worker",
                                           cf["num_gpus_per_worker"]),
                } for _ in range(teacher_config["num_workers"])
            ] + ([
                {
                    # Evaluation (remote) workers.
                    # Note: The local eval worker is located on the driver
                    # CPU or not even created iff >0 eval workers.
                    "CPU": eval_config.get("num_cpus_per_worker",
                                           cf["num_cpus_per_worker"]),
                    "GPU": eval_config.get("num_gpus_per_worker",
                                           cf["num_gpus_per_worker"]),
                } for _ in range(cf["evaluation_num_workers"])
            ] if cf["evaluation_interval"] else []),
            strategy=config.get("placement_strategy", "PACK"))
