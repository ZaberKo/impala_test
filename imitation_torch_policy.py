from ref.vtrace_torch_policy import *



ImitationTorchPolicy = build_policy_class(
    name="ImitationTorchPolicy",
    framework="torch",
    loss_fn=build_vtrace_loss,
    get_default_config=lambda: ray.rllib.agents.impala.impala.DEFAULT_CONFIG,
    stats_fn=stats,
    extra_grad_process_fn=apply_grad_clipping,
    optimizer_fn=choose_optimizer,
    before_init=setup_mixins,
    mixins=[LearningRateSchedule, EntropyCoeffSchedule],
    get_batch_divisibility_req=lambda p: p.config["rollout_fragment_length"],
)