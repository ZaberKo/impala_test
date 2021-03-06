import math
import copy
import pickle
from ref.impala import *

from ray.rllib.execution.common import _get_shared_metrics, _get_global_vars
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.utils import merge_dicts
from pathlib import Path

from ray.rllib.utils.debug import update_global_seed_if_necessary
from collections import defaultdict
from typing import DefaultDict, Set
from ray.actor import ActorHandle

STUDENT_DEFAULT_CONFIG = merge_dicts(DEFAULT_CONFIG, {
    "teacher_checkpoint": "",
    "teacher_workers_frac": 1.0
})


def gather_experiences_directly(workers, config):
    from ref.rollout_ops import ParallelRollouts
    rollouts = ParallelRollouts(
        workers,
        mode="async",
        num_async=config["max_sample_requests_in_flight_per_worker"])

    # Augment with replay and concat to desired train batch size.
    train_batches = rollouts \
        .for_each(lambda batch: batch.decompress_if_needed()) \
        .for_each(MixInReplay(
            num_slots=config["replay_buffer_num_slots"],
            replay_proportion=config["replay_proportion"])) \
        .flatten() \
        .combine(
            ConcatBatches(
                min_batch_size=config["train_batch_size"],
                count_steps_by=config["multiagent"]["count_steps_by"],
            ))

    return train_batches


class BroadcastUpdateLearnerWeights:
    def __init__(self, learner_thread, workers,exclude_workers_set, broadcast_interval):
        self.learner_thread = learner_thread
        self.steps_since_broadcast = 0
        self.broadcast_interval = broadcast_interval
        self.workers = workers
        self.weights = workers.local_worker().get_weights()
        self.exclude_workers_set=exclude_workers_set

    def __call__(self, item):
        actor, batch = item
        self.steps_since_broadcast += 1
        if (self.steps_since_broadcast >= self.broadcast_interval
                and self.learner_thread.weights_updated
                and actor not in self.exclude_workers_set):
            self.weights = ray.put(self.workers.local_worker().get_weights())
            self.steps_since_broadcast = 0
            self.learner_thread.weights_updated = False
            # Update metrics.
            metrics = _get_shared_metrics()
            metrics.counters["num_weight_broadcasts"] += 1
        actor.set_weights.remote(self.weights, _get_global_vars())
        # Also update global vars of the local worker.
        self.workers.local_worker().set_global_vars(_get_global_vars())


class ImpalaStudentTrainer(ImpalaTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return STUDENT_DEFAULT_CONFIG

    def teacher_setup(self):
        # create teacher workers

        # self.teacher_config = merge_dicts(self.config, copy.deepcopy(self.config["teacher_config"]))

        # restore teacher workers from checkpoint
        teacher_checkpoint = Path(
            self.config["teacher_checkpoint"]).expanduser()
        with teacher_checkpoint.open("rb") as f:
            teacher_state = pickle.load(f)

        # contains model weights
        teacher_worker_state = teacher_state["worker"]
        # self.teacher_workers.local_worker().restore(teacher_worker_state)
        teacher_worker_state_ref = ray.put(teacher_worker_state)

        self.num_teacher_workers = math.floor(
            len(self.workers.remote_workers())*self.config["teacher_workers_frac"])

        self.teacher_workers_set = {
            w
            for w in self.workers.remote_workers()[:self.num_teacher_workers]
        }

        for r in self.teacher_workers_set:
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
        update_global_seed_if_necessary(
            self.config["framework"], self.config["seed"])

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
            # TODO: Uncommenting line above breaks tf2+eager_tracing for A3C.
            pass
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

            env_id = self._register_if_needed(
                eval_config.get("env"), eval_config)
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
        kwargs = super()._kwargs_for_execution_plan()
        kwargs["teacher_workers_set"] = self.teacher_workers_set
        return kwargs

    @staticmethod
    def execution_plan(workers, config, **kwargs):
        teacher_workers_set = kwargs["teacher_workers_set"]

        train_batches = gather_experiences_directly(workers, config)

        # Start the learner thread.
        learner_thread = make_learner_thread(workers.local_worker(), config)
        learner_thread.start()

        # This sub-flow sends experiences to the learner.
        enqueue_op = train_batches \
            .for_each(Enqueue(learner_thread.inqueue))

        # !!!No need to update teacher workers.
        if workers.remote_workers():
            enqueue_op = enqueue_op.zip_with_source_actor() \
                .for_each(BroadcastUpdateLearnerWeights(
                    learner_thread, workers, 
                    exclude_workers_set=teacher_workers_set,
                    broadcast_interval=config["broadcast_interval"]))

        # update timestep for schedulers
        # enqueue_op = enqueue_op.for_each(UpdateGlobalVars(workers))

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
            metrics.counters[STEPS_TRAINED_THIS_ITER_COUNTER] = 0
            return item

        return StandardMetricsReporting(merged_op, workers, config) \
            .for_each(learner_thread.add_learner_metrics) \
            .for_each(reset_trained_timesteps)
