import math
import copy
import pickle
from ref.impala import *
from ray.rllib.execution.common import _get_shared_metrics,_get_global_vars
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import merge_dicts
from pathlib import Path

from ray.rllib.utils.debug import update_global_seed_if_necessary
from collections import defaultdict
from typing import DefaultDict,Set
from ray.actor import ActorHandle

STUDENT_DEFAULT_CONFIG=merge_dicts(DEFAULT_CONFIG,{
    "teacher_checkpoint": "",
    "teacher_config": {
        "num_workers": 1,
        "num_envs_per_worker": 1,
    },
})

class UpdateGlobalVars:
    def __init__(self, workers):
        self.workers = workers

    def __call__(self, item):
        # Update global vars of the local worker of student worker.
        self.workers.local_worker().set_global_vars(_get_global_vars())




class ImpalaStudentTrainer(ImpalaTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return STUDENT_DEFAULT_CONFIG

    def teacher_setup(self):
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


    def setup(self, config: PartialTrainerConfigDict):

        # Setup our config: Merge the user-supplied config (which could
        # be a partial config dict with the class' default).
        self.config = self.merge_trainer_configs(
            self.get_default_config(), config, self._allow_unknown_configs
        )
        self.config["env"] = self._env_id

        # Validate the framework settings in config.
        self.validate_framework(self.config)

        # Setup the self.env_creator callable (to be passed
        # e.g. to RolloutWorkers' c'tors).
        self.env_creator = self._get_env_creator_from_env_id(self._env_id)

        # Set Trainer's seed after we have - if necessary - enabled
        # tf eager-execution.
        update_global_seed_if_necessary(self.config["framework"], self.config["seed"])

        self.validate_config(self.config)
        self.callbacks = self.config["callbacks"]()
        log_level = self.config.get("log_level")
        if log_level in ["WARN", "ERROR"]:
            logger.info(
                "Current log_level is {}. For more information, "
                "set 'log_level': 'INFO' / 'DEBUG' or use the -v and "
                "-vv flags.".format(log_level)
            )
        if self.config.get("log_level"):
            logging.getLogger("ray.rllib").setLevel(self.config["log_level"])

        # Create local replay buffer if necessary.
        self.local_replay_buffer = self._create_local_replay_buffer_if_necessary(
            self.config
        )

        # Create a dict, mapping ActorHandles to sets of open remote
        # requests (object refs). This way, we keep track, of which actors
        # inside this Trainer (e.g. a remote RolloutWorker) have
        # already been sent how many (e.g. `sample()`) requests.
        self.remote_requests_in_flight: DefaultDict[
            ActorHandle, Set[ray.ObjectRef]
        ] = defaultdict(set)

        self.workers: Optional[WorkerSet] = None
        self.train_exec_impl = None


        # Only if user did not override `_init()`:
        # - Create rollout workers here automatically.
        # - Run the execution plan to create the local iterator to `next()`
        #   in each training iteration.
        # This matches the behavior of using `build_trainer()`, which
        # should no longer be used.
        self.workers = WorkerSet(
            env_creator=self.env_creator,
            validate_env=self.validate_env,
            policy_class=self.get_default_policy_class(self.config),
            trainer_config=self.config,
            num_workers=self.config["num_workers"],
            local_worker=True,
            logdir=self.logdir,
        )

        self.teacher_setup()

        # Function defining one single training iteration's behavior.
        if self.config["_disable_execution_plan_api"]:
            # TODO: Ensure remote workers are initially in sync with the
            # local worker.
            # self.workers.sync_weights()
            pass  # TODO: Uncommenting line above breaks tf2+eager_tracing for A3C.
        # LocalIterator-creating "execution plan".
        # Only call this once here to create `self.train_exec_impl`,
        # which is a ray.util.iter.LocalIterator that will be `next`'d
        # on each training iteration.
        else:
            self.train_exec_impl = self.execution_plan(
                self.workers, self.config, **self._kwargs_for_execution_plan()
            )

        # Now that workers have been created, update our policies
        # dict in config[multiagent] (with the correct original/
        # unpreprocessed spaces).
        self.config["multiagent"][
            "policies"
        ] = self.workers.local_worker().policy_dict

        # Evaluation WorkerSet setup.
        # User would like to setup a separate evaluation worker set.

        # Update with evaluation settings:
        user_eval_config = copy.deepcopy(self.config["evaluation_config"])

        # Assert that user has not unset "in_evaluation".
        assert (
            "in_evaluation" not in user_eval_config
            or user_eval_config["in_evaluation"] is True
        )

        # Merge user-provided eval config with the base config. This makes sure
        # the eval config is always complete, no matter whether we have eval
        # workers or perform evaluation on the (non-eval) local worker.
        eval_config = merge_dicts(self.config, user_eval_config)
        self.config["evaluation_config"] = eval_config

        if self.config.get("evaluation_num_workers", 0) > 0 or self.config.get(
            "evaluation_interval"
        ):
            logger.debug(f"Using evaluation_config: {user_eval_config}.")

            # Validate evaluation config.
            self.validate_config(eval_config)

            # Set the `in_evaluation` flag.
            eval_config["in_evaluation"] = True

            # Evaluation duration unit: episodes.
            # Switch on `complete_episode` rollouts. Also, make sure
            # rollout fragments are short so we never have more than one
            # episode in one rollout.
            if eval_config["evaluation_duration_unit"] == "episodes":
                eval_config.update(
                    {
                        "batch_mode": "complete_episodes",
                        "rollout_fragment_length": 1,
                    }
                )
            # Evaluation duration unit: timesteps.
            # - Set `batch_mode=truncate_episodes` so we don't perform rollouts
            #   strictly along episode borders.
            # Set `rollout_fragment_length` such that desired steps are divided
            # equally amongst workers or - in "auto" duration mode - set it
            # to a reasonably small number (10), such that a single `sample()`
            # call doesn't take too much time so we can stop evaluation as soon
            # as possible after the train step is completed.
            else:
                eval_config.update(
                    {
                        "batch_mode": "truncate_episodes",
                        "rollout_fragment_length": 10
                        if self.config["evaluation_duration"] == "auto"
                        else int(
                            math.ceil(
                                self.config["evaluation_duration"]
                                / (self.config["evaluation_num_workers"] or 1)
                            )
                        ),
                    }
                )

            self.config["evaluation_config"] = eval_config

            env_id = self._register_if_needed(eval_config.get("env"), eval_config)
            env_creator = self._get_env_creator_from_env_id(env_id)

            # Create a separate evaluation worker set for evaluation.
            # If evaluation_num_workers=0, use the evaluation set's local
            # worker for evaluation, otherwise, use its remote workers
            # (parallelized evaluation).
            self.evaluation_workers: WorkerSet = WorkerSet(
                env_creator=env_creator,
                validate_env=None,
                policy_class=self.get_default_policy_class(self.config),
                trainer_config=eval_config,
                num_workers=self.config["evaluation_num_workers"],
                # Don't even create a local worker if num_workers > 0.
                local_worker=False,
                logdir=self.logdir,
            )

        # Run any callbacks after trainer initialization is done.
        self.callbacks.on_trainer_init(trainer=self)

       

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

        # update timestep for schedulers
        enqueue_op=enqueue_op.for_each(UpdateGlobalVars(workers))

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

        return StandardMetricsReporting(merged_op, teacher_workers, config) \
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
