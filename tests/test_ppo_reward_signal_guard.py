import torch

from modelBased.policy_training.PPO import PPO


def _make_agent():
    torch.manual_seed(7)
    return PPO(
        state_dim=8,
        action_dim=6,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        K_epochs=2,
        eps_clip=0.2,
        has_continuous_action_space=False,
        entropy_coef=0.01,
        normalize_advantages=True,
        normalize_returns=False,
        freeze_actor_until_first_reward=True,
    )


def _collect_parallel(agent, rewards, terminal=True):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    states = torch.randn(rewards.numel(), 8)
    _, saved_states, actions, logprobs, values = agent.select_action_batch(states)
    agent.save_buffer_batch(
        saved_states,
        actions,
        logprobs,
        values,
        rewards,
        torch.full((rewards.numel(),), terminal, dtype=torch.bool),
    )


def test_zero_reward_rollout_freezes_actor_but_trains_critic():
    agent = _make_agent()
    _collect_parallel(agent, torch.zeros(256))

    metrics = agent.update(bootstrap_value=torch.ones(256))

    assert metrics["updated"]
    assert not metrics["reward_signal_in_rollout"]
    assert not metrics["actor_update_enabled"]
    assert metrics["actor_parameter_delta"] == 0.0
    assert metrics["critic_parameter_delta"] > 0.0


def test_first_reward_unlocks_actor_and_future_updates():
    agent = _make_agent()
    rewards = torch.zeros(512)
    rewards[0] = 1.0
    _collect_parallel(agent, rewards)

    rewarded = agent.update()

    assert rewarded["reward_signal_in_rollout"]
    assert rewarded["has_seen_reward_signal"]
    assert rewarded["actor_update_enabled"]
    assert rewarded["actor_parameter_delta"] > 0.0

    _collect_parallel(agent, torch.zeros(512), terminal=False)
    subsequent = agent.update(bootstrap_value=torch.full((512,), 0.25))

    assert not subsequent["reward_signal_in_rollout"]
    assert subsequent["has_seen_reward_signal"]
    assert subsequent["actor_update_enabled"]


def test_behavior_cloning_updates_only_actor_and_keeps_reward_guard():
    agent = _make_agent()
    states = torch.eye(8)[:6].repeat(8, 1)
    actions = torch.arange(6).repeat(8)
    critic_before = {
        name: value.detach().clone()
        for name, value in agent.policy.critic.named_parameters()
    }

    metrics = agent.behavior_clone(
        states,
        actions,
        epochs=400,
        batch_size=32,
        learning_rate=3e-3,
        target_accuracy=1.0,
        target_probability=0.95,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["mean_target_probability"] >= 0.95
    assert metrics["actor_parameter_delta"] > 0.0
    assert not agent.has_seen_reward_signal
    for name, value in agent.policy.critic.named_parameters():
        assert torch.equal(value, critic_before[name])
    for current, rollout in zip(
        agent.policy.actor.parameters(), agent.policy_old.actor.parameters()
    ):
        assert torch.equal(current, rollout)
