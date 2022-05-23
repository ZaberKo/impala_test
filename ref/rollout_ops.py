import logging
import time
from typing import Callable, Container, List, Optional, Tuple, TYPE_CHECKING

import ray
from ray.rllib.evaluation.rollout_worker import get_global_worker
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.common import (
    AGENT_STEPS_SAMPLED_COUNTER,
    STEPS_SAMPLED_COUNTER,
    SAMPLE_TIMER,
    GRAD_WAIT_TIMER,
    _check_sample_batch_type,
    _get_shared_metrics,
)
from ray.rllib.policy.sample_batch import (
    SampleBatch,
    DEFAULT_POLICY_ID,
    MultiAgentBatch,
)
from ray.rllib.utils.annotations import ExperimentalAPI
from ray.rllib.utils.metrics.learner_info import LEARNER_INFO, LEARNER_STATS_KEY
from ray.rllib.utils.sgd import standardized
from ray.rllib.utils.typing import PolicyID, SampleBatchType, ModelGradients
from ray.util.iter import from_actors, LocalIterator
from ray.util.iter_metrics import SharedMetrics

def ParallelRollouts(
    workers: WorkerSet, *, mode="bulk_sync", num_async=1
) -> LocalIterator[SampleBatch]:
    """Operator to collect experiences in parallel from rollout workers.

    If there are no remote workers, experiences will be collected serially from
    the local worker instance instead.

    Args:
        workers (WorkerSet): set of rollout workers to use.
        mode (str): One of 'async', 'bulk_sync', 'raw'. In 'async' mode,
            batches are returned as soon as they are computed by rollout
            workers with no order guarantees. In 'bulk_sync' mode, we collect
            one batch from each worker and concatenate them together into a
            large batch to return. In 'raw' mode, the ParallelIterator object
            is returned directly and the caller is responsible for implementing
            gather and updating the timesteps counter.
        num_async (int): In async mode, the max number of async
            requests in flight per actor.

    Returns:
        A local iterator over experiences collected in parallel.

    Examples:
        >>> rollouts = ParallelRollouts(workers, mode="async")
        >>> batch = next(rollouts)
        >>> print(batch.count)
        50  # config.rollout_fragment_length

        >>> rollouts = ParallelRollouts(workers, mode="bulk_sync")
        >>> batch = next(rollouts)
        >>> print(batch.count)
        200  # config.rollout_fragment_length * config.num_workers

    Updates the STEPS_SAMPLED_COUNTER counter in the local iterator context.
    """

    # disable it 
    # workers.sync_weights()

    def report_timesteps(batch):
        metrics = _get_shared_metrics()
        metrics.counters[STEPS_SAMPLED_COUNTER] += batch.count
        if isinstance(batch, MultiAgentBatch):
            metrics.counters[AGENT_STEPS_SAMPLED_COUNTER] += batch.agent_steps()
        else:
            metrics.counters[AGENT_STEPS_SAMPLED_COUNTER] += batch.count
        return batch

    if not workers.remote_workers():
        # Handle the `num_workers=0` case, in which the local worker
        # has to do sampling as well.
        def sampler(_):
            while True:
                yield workers.local_worker().sample()

        return LocalIterator(sampler, SharedMetrics()).for_each(report_timesteps)

    # Create a parallel iterator over generated experiences.
    rollouts = from_actors(workers.remote_workers())

    if mode == "bulk_sync":
        return (
            rollouts.batch_across_shards()
            .for_each(lambda batches: SampleBatch.concat_samples(batches))
            .for_each(report_timesteps)
        )
    elif mode == "async":
        return rollouts.gather_async(num_async=num_async).for_each(report_timesteps)
    elif mode == "raw":
        return rollouts
    else:
        raise ValueError(
            "mode must be one of 'bulk_sync', 'async', 'raw', got '{}'".format(mode)
        )
