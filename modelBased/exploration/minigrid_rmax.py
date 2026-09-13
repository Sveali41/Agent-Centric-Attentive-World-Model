"""Persistent feedforward spatial PPO explorer for fully observed MiniGrid."""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX, STATE_TO_IDX
from torch.distributions import Categorical
from modelBased.exploration.count_based import MINIGRID_PLAYER_ID

_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_ARCH = "minigrid_spatial_feedforward_position_v1"
_DQN_ARCH = "minigrid_spatial_dqn_position_v1"


def _pos(image):
    found = np.argwhere(np.asarray(image)[..., 0] == MINIGRID_PLAYER_ID)
    if len(found) != 1:
        raise ValueError(f"Expected exactly one MiniGrid agent, found {len(found)}")
    return tuple(map(int, found[0]))


class _Buffer:
    def __init__(self):
        self.states, self.actions, self.logprobs, self.values = [], [], [], []
        self.rewards, self.terminals, self.boundaries, self.bootstraps = [], [], [], []

    @property
    def is_terminals(self):
        return self.terminals

    def transition_count(self):
        return len(self.rewards)

    def append(self, *, state, action, logprob, value, reward, terminal, bootstrap):
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu().reshape(()))
        self.logprobs.append(logprob.detach().cpu().reshape(()))
        self.values.append(value.detach().cpu().reshape(()))
        self.rewards.append(float(reward)); self.terminals.append(bool(terminal)); self.boundaries.append(bool(terminal)); self.bootstraps.append(float(bootstrap))

    def clear(self):
        self.__init__()


class _SpatialActorCritic(nn.Module):
    def __init__(self, feature_dim, actions, object_dim=8, color_dim=4, state_dim=4, channels=64):
        super().__init__()
        self.feature_dim, self.spatial_pool_size = int(feature_dim), 3
        self.object_embedding = nn.Embedding(max(OBJECT_TO_IDX.values()) + 1, object_dim)
        self.color_embedding = nn.Embedding(len(COLOR_TO_IDX), color_dim)
        self.state_embedding = nn.Embedding(max(STATE_TO_IDX.values()) + 1, state_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(object_dim + color_dim + state_dim + 3, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.encoder = nn.Sequential(nn.Linear(channels * (3 + self.spatial_pool_size ** 2) + 7, self.feature_dim), nn.Tanh())
        self.actor_head = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.Tanh(), nn.Linear(self.feature_dim, actions))
        self.critic_head = nn.Sequential(nn.Linear(self.feature_dim, self.feature_dim), nn.Tanh(), nn.Linear(self.feature_dim, 1))

    def encode(self, states):
        grid = states.to(_DEVICE)[:, :6].long()
        inv = states.to(_DEVICE)[:, 6:13, 0, 0].float()
        parts = (
            self.object_embedding(grid[:, 0].clamp(0, self.object_embedding.num_embeddings - 1)).permute(0, 3, 1, 2),
            self.color_embedding(grid[:, 1].clamp(0, self.color_embedding.num_embeddings - 1)).permute(0, 3, 1, 2),
            self.state_embedding(grid[:, 2].clamp(0, self.state_embedding.num_embeddings - 1)).permute(0, 3, 1, 2), grid[:, 3:].float())
        f = self.conv(torch.cat(parts, 1)); agent = grid[:, 3:4].float()
        return self.encoder(torch.cat(((f * agent).sum((2, 3)), nn.functional.adaptive_avg_pool2d(f, 1).flatten(1), nn.functional.adaptive_max_pool2d(f, 1).flatten(1), nn.functional.adaptive_avg_pool2d(f, self.spatial_pool_size).flatten(1), inv), 1))

    def forward(self, states):
        feature = self.encode(states)
        return Categorical(logits=self.actor_head(feature)), self.critic_head(feature).flatten()


class _MiniGridPPO:
    def __init__(self, *, action_dim, feature_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, entropy_coef, normalize_advantages, normalize_returns, max_grad_norm, minibatch_transitions, object_dim, color_dim, state_dim, channels):
        self.gamma, self.K_epochs, self.eps_clip, self.entropy_coef = float(gamma), int(K_epochs), float(eps_clip), float(entropy_coef)
        self.normalize_advantages, self.normalize_returns, self.max_grad_norm = bool(normalize_advantages), bool(normalize_returns), float(max_grad_norm)
        self.minibatch_transitions = int(minibatch_transitions)
        if self.minibatch_transitions <= 0: raise ValueError("MiniGrid ppo_minibatch_transitions must be positive")
        kw = dict(object_dim=object_dim, color_dim=color_dim, state_dim=state_dim, channels=channels)
        self.policy = _SpatialActorCritic(feature_dim, action_dim, **kw).to(_DEVICE)
        self.policy_old = _SpatialActorCritic(feature_dim, action_dim, **kw).to(_DEVICE)
        self.policy_old.load_state_dict(self.policy.state_dict())
        encoder = list(self.policy.object_embedding.parameters()) + list(self.policy.color_embedding.parameters()) + list(self.policy.state_embedding.parameters()) + list(self.policy.conv.parameters()) + list(self.policy.encoder.parameters())
        self.optimizer = torch.optim.Adam([{"params": encoder, "lr": float(lr_actor)}, {"params": self.policy.actor_head.parameters(), "lr": float(lr_actor)}, {"params": self.policy.critic_head.parameters(), "lr": float(lr_critic)}])
        self.buffer, self.update_count = _Buffer(), 0

    def select_action(self, state, generator):
        with torch.no_grad():
            dist, value = self.policy_old(state.unsqueeze(0)); action = torch.multinomial(dist.probs, 1, generator=generator).squeeze(0)
        return int(action.item()), state.detach(), action.detach(), dist.log_prob(action).detach(), value.detach()

    def estimate(self, state):
        with torch.no_grad(): return float(self.policy_old(state.unsqueeze(0))[1].item())

    def save(self, state, action, logprob, value, reward, terminal, bootstrap):
        self.buffer.append(state=state, action=action, logprob=logprob, value=value, reward=reward, terminal=terminal, bootstrap=bootstrap)

    def update(self, generator):
        n = self.buffer.transition_count()
        if not n: return {"updated": False, "update_count": self.update_count, "transition_count": 0, "rollout_size": 0, "optimizer_steps": 0}
        # The collector calls mark_rollout_boundary after each map.  Explicit
        # boundary flags prevent returns from crossing that map; terminal uses 0.
        rewards, terminals, boundaries, bootstraps = self.buffer.rewards, self.buffer.terminals, self.buffer.boundaries, self.buffer.bootstraps
        returns = np.zeros(n, dtype=np.float32); running = 0.0
        for i in range(n - 1, -1, -1):
            if terminals[i]: running = 0.0
            elif boundaries[i]: running = bootstraps[i]
            running = rewards[i] + self.gamma * running; returns[i] = running
        states, actions = torch.stack(self.buffer.states).to(_DEVICE), torch.stack(self.buffer.actions).long().to(_DEVICE)
        old_logprobs, old_values = torch.stack(self.buffer.logprobs).to(_DEVICE), torch.stack(self.buffer.values).to(_DEVICE)
        raw_returns = torch.as_tensor(returns, device=_DEVICE); advantages, targets = raw_returns - old_values, raw_returns
        mean, scale = torch.zeros((), device=_DEVICE), torch.ones((), device=_DEVICE)
        if self.normalize_returns and n > 1: mean, scale = targets.mean(), targets.std(unbiased=False) + 1e-7; targets = (targets - mean) / scale
        if self.normalize_advantages and n > 1: advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-7)
        steps = transitions = 0; kl_sum = clip_sum = entropy_sum = 0.0
        for _ in range(self.K_epochs):
            for ind in torch.randperm(n, generator=generator, device=_DEVICE).split(self.minibatch_transitions):
                dist, values = self.policy(states[ind]); ratio = torch.exp(dist.log_prob(actions[ind]) - old_logprobs[ind]); log_ratio = torch.log(ratio)
                critic = (values - mean) / scale if self.normalize_returns and n > 1 else values
                policy_loss = -torch.min(ratio * advantages[ind], torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages[ind]).mean()
                loss = policy_loss + .5 * nn.functional.mse_loss(critic, targets[ind]) - self.entropy_coef * dist.entropy().mean()
                self.optimizer.zero_grad(set_to_none=True); loss.backward()
                if self.max_grad_norm > 0: nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step(); steps += 1; transitions += len(ind)
                kl_sum += float(((ratio - 1) - log_ratio).detach().sum()); clip_sum += float((torch.abs(ratio - 1) > self.eps_clip).float().sum()); entropy_sum += float(dist.entropy().detach().sum())
        self.policy_old.load_state_dict(self.policy.state_dict()); self.buffer.clear(); self.update_count += 1
        return {"updated": True, "update_count": self.update_count, "transition_count": n, "rollout_size": n, "optimizer_steps": steps, "approx_kl": kl_sum/max(transitions,1), "clip_fraction": clip_sum/max(transitions,1), "entropy": entropy_sum/max(transitions,1)}


class _SpatialQNetwork(_SpatialActorCritic):
    """The DQN shares the established size-independent spatial encoder."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.critic_head

    def forward(self, states):
        return self.actor_head(self.encode(states))


class _ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        if self.capacity <= 0: raise ValueError("MiniGrid dqn_replay_capacity must be positive")
        self.states, self.actions, self.rewards, self.next_states, self.terminals = [], [], [], [], []
        self._next = 0

    def __len__(self): return len(self.states)

    def add(self, state, action, reward, next_state, terminal):
        item = (state.detach().cpu(), int(action), float(reward), None if next_state is None else next_state.detach().cpu(), bool(terminal))
        if len(self) < self.capacity:
            self.states.append(item[0]); self.actions.append(item[1]); self.rewards.append(item[2]); self.next_states.append(item[3]); self.terminals.append(item[4])
        else:
            self.states[self._next], self.actions[self._next], self.rewards[self._next], self.next_states[self._next], self.terminals[self._next] = item
        self._next = (self._next + 1) % self.capacity

    def sample(self, batch_size, generator):
        indices = torch.randint(len(self), (int(batch_size),), generator=generator).tolist()
        states = torch.stack([self.states[i] for i in indices]).to(_DEVICE)
        actions = torch.as_tensor([self.actions[i] for i in indices], device=_DEVICE, dtype=torch.long)
        rewards = torch.as_tensor([self.rewards[i] for i in indices], device=_DEVICE, dtype=torch.float32)
        terminals = torch.as_tensor([self.terminals[i] for i in indices], device=_DEVICE, dtype=torch.float32)
        # A transition without a next state is necessarily terminal.
        next_states = torch.stack([self.states[i] if self.next_states[i] is None else self.next_states[i] for i in indices]).to(_DEVICE)
        return states, actions, rewards, next_states, terminals

    def state_dict(self):
        return {"capacity": self.capacity, "states": self.states, "actions": self.actions, "rewards": self.rewards, "next_states": self.next_states, "terminals": self.terminals, "next": self._next}

    def load_state_dict(self, state):
        if int(state["capacity"]) != self.capacity: raise ValueError("DQN replay capacity is incompatible with this checkpoint")
        self.states=list(state["states"]); self.actions=list(state["actions"]); self.rewards=list(state["rewards"]); self.next_states=list(state["next_states"]); self.terminals=list(state["terminals"]); self._next=int(state["next"])


class _MiniGridDQN:
    def __init__(self, *, action_dim, feature_dim, lr, gamma, replay_capacity, batch_size, warmup_transitions, updates_per_iteration, target_update_interval, epsilon_start, epsilon_end, epsilon_decay_steps, max_grad_norm, object_dim, color_dim, state_dim, channels):
        self.action_dim, self.gamma = int(action_dim), float(gamma)
        self.batch_size, self.warmup_transitions = int(batch_size), int(warmup_transitions)
        self.updates_per_iteration, self.target_update_interval = int(updates_per_iteration), int(target_update_interval)
        self.epsilon_start, self.epsilon_end, self.epsilon_decay_steps = float(epsilon_start), float(epsilon_end), int(epsilon_decay_steps)
        self.max_grad_norm = float(max_grad_norm)
        if self.batch_size <= 0 or self.warmup_transitions < self.batch_size or self.updates_per_iteration <= 0 or self.target_update_interval <= 0: raise ValueError("Invalid MiniGrid DQN batch, warmup, update, or target interval")
        if not 0.0 <= self.epsilon_end <= self.epsilon_start <= 1.0 or self.epsilon_decay_steps <= 0: raise ValueError("Invalid MiniGrid DQN epsilon schedule")
        kw = dict(object_dim=object_dim, color_dim=color_dim, state_dim=state_dim, channels=channels)
        self.policy = _SpatialQNetwork(feature_dim, action_dim, **kw).to(_DEVICE)
        self.target = _SpatialQNetwork(feature_dim, action_dim, **kw).to(_DEVICE)
        self.target.load_state_dict(self.policy.state_dict()); self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=float(lr))
        self.replay = _ReplayBuffer(replay_capacity)
        self.transition_count_total = self.update_count = self.optimizer_steps = self._iteration_transitions = 0

    def epsilon(self):
        progress = min(self.transition_count_total / self.epsilon_decay_steps, 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, generator, *, explore):
        if explore and float(torch.rand((), generator=generator).item()) < self.epsilon(): return int(torch.randint(self.action_dim, (), generator=generator).item())
        with torch.no_grad(): return int(self.policy(state.unsqueeze(0)).argmax(1).item())

    def save(self, state, action, reward, next_state, terminal):
        self.replay.add(state, action, reward, next_state, terminal); self.transition_count_total += 1; self._iteration_transitions += 1

    def update(self, generator):
        n = self._iteration_transitions
        if len(self.replay) < self.warmup_transitions:
            return {"updated": False, "update_count": self.update_count, "transition_count": n, "rollout_size": n, "optimizer_steps": 0, "epsilon": self.epsilon(), "replay_size": len(self.replay)}
        losses = []
        for _ in range(self.updates_per_iteration):
            states, actions, rewards, next_states, terminals = self.replay.sample(self.batch_size, generator)
            with torch.no_grad(): targets = rewards + self.gamma * (1.0 - terminals) * self.target(next_states).max(1).values
            values = self.policy(states).gather(1, actions[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(values, targets)
            self.optimizer.zero_grad(set_to_none=True); loss.backward()
            if self.max_grad_norm > 0: nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step(); self.optimizer_steps += 1; losses.append(float(loss.detach()))
            if self.optimizer_steps % self.target_update_interval == 0: self.target.load_state_dict(self.policy.state_dict())
        self.update_count += 1
        return {"updated": True, "update_count": self.update_count, "transition_count": n, "rollout_size": n, "optimizer_steps": self.updates_per_iteration, "epsilon": self.epsilon(), "replay_size": len(self.replay), "dqn_loss": float(np.mean(losses))}


class MiniGridRMaxExplorer:
    expects_raw_obs = True
    def __init__(self, cfg):
        d, r = cfg.domains.minigrid, cfg.domains.minigrid.rmax_like; shape = tuple(map(int, d.grid_shape))
        if len(shape) != 3 or shape[0] != 3 or min(shape[1:]) <= 0: raise ValueError(f"MiniGrid grid_shape must be (3,H,W), got {shape}")
        if int(d.action_norm) != 6: raise ValueError(f"MiniGrid spatial explorer requires six compact actions, got {int(d.action_norm)}")
        self.shape = shape[1:]; self.grid_height, self.grid_width = self.shape; self.raw_shape = (self.grid_width, self.grid_height); self.inventory_classes = 7; self.obs_norm_values = tuple(map(float, d.obs_norm))
        self.ppo = _MiniGridPPO(action_dim=6, feature_dim=int(getattr(r, "feature_dim", 128)), lr_actor=float(r.lr_actor), lr_critic=float(r.lr_critic), gamma=float(r.gamma), K_epochs=int(r.K_epochs), eps_clip=float(r.eps_clip), entropy_coef=float(r.entropy_coef), normalize_advantages=bool(r.normalize_advantages), normalize_returns=bool(r.normalize_returns), max_grad_norm=float(r.max_grad_norm), minibatch_transitions=int(getattr(r, "ppo_minibatch_transitions", 512)), object_dim=int(getattr(r,"object_embedding_dim",8)), color_dim=int(getattr(r,"color_embedding_dim",4)), state_dim=int(getattr(r,"state_embedding_dim",4)), channels=int(getattr(r,"conv_channels",64)))
        self.ppo_updates_per_rollout = int(getattr(r, "ppo_updates_per_rollout", 1))
        if self.ppo_updates_per_rollout <= 0:
            raise ValueError("MiniGrid ppo_updates_per_rollout must be positive")
        self.death_penalty = float(getattr(r, "death_penalty", 0.0))
        if self.death_penalty < 0.0:
            raise ValueError("MiniGrid death_penalty must be non-negative")
        self._visited_positions = np.zeros(self.raw_shape, np.float32); self._pending_transition = None; self.training_enabled = True; self._iteration_active = False; self.completed_iteration = 0; self.current_inventory_token = self.next_inventory_token = 0
        self._rollout_transition_count = 0; self._rollout_update_boundaries = []; self._next_rollout_update = 0; self._iteration_updates = []
        self._rng = torch.Generator(device=_DEVICE); self._rng.manual_seed(int(getattr(cfg,"seed",0))); self.checkpoint_path = getattr(r,"checkpoint_path",None); self._reset_metrics()

    def _reset_metrics(self):
        self._rollout_transitions=self._rollout_changed=self._rollout_moved=self._rollout_rewarded=self._rollout_no_effect=0; self._rollout_deaths=0; self._rollout_reward_sum=0.; self._rollout_action_count=np.zeros(6,np.int64); self._rollout_interaction_successes={"pickup":0,"toggle":0,"drop":0}
    def set_training(self, enabled): self.training_enabled=bool(enabled); self._pending_transition=None
    def begin_iteration(self):
        if self._iteration_active or self.ppo.buffer.transition_count() or self._pending_transition is not None: raise RuntimeError("Cannot begin MiniGrid iteration with pending transitions")
        self.ppo.policy_old.load_state_dict(self.ppo.policy.state_dict()); self._iteration_updates=[]; self._iteration_active=True
    def end_iteration(self):
        if not self._iteration_active: raise RuntimeError("end_iteration requires begin_iteration")
        if self.ppo_updates_per_rollout == 1:
            result=self.ppo.update(self._rng)
        else:
            if self.ppo.buffer.transition_count():
                raise RuntimeError("MiniGrid rollout ended with an unflushed PPO segment")
            result=self._aggregate_iteration_updates()
        self._iteration_active=False; self.completed_iteration+=1; return result
    def begin_rollout(self, rollout_budget):
        budget=int(rollout_budget)
        if budget <= 0: raise ValueError("MiniGrid rollout budget must be positive")
        self._visited_positions=np.zeros(self.raw_shape,np.float32); self._pending_transition=None; self._reset_metrics()
        self._rollout_transition_count=0
        # Boundaries split one continuous map rollout into equal-sized PPO
        # segments.  The final segment may be one transition shorter.
        self._rollout_update_boundaries=[(i * budget + self.ppo_updates_per_rollout - 1) // self.ppo_updates_per_rollout for i in range(1, self.ppo_updates_per_rollout + 1)]
        self._next_rollout_update=0
    def reset_episode(self): self._pending_transition=None
    def set_carrying_token(self, token):
        if not 0<=int(token)<self.inventory_classes: raise ValueError(f"Invalid MiniGrid inventory token: {token}")
        self.current_inventory_token=int(token)
    def set_next_carrying_token(self, token):
        if not 0<=int(token)<self.inventory_classes: raise ValueError(f"Invalid MiniGrid inventory token: {token}")
        self.next_inventory_token=int(token)
    def _check(self,image):
        if np.asarray(image).shape != (*self.raw_shape,3): raise ValueError(f"MiniGrid observation shape changed within explorer run: expected {(*self.raw_shape,3)}, got {np.asarray(image).shape}")
    def _mark(self,image): self._check(image); x,y=_pos(image); self._visited_positions[x,y]=1
    def _frontier(self,image):
        v=self._visited_positions.astype(bool); free=image[...,0]!=OBJECT_TO_IDX["wall"]; door=image[...,0]==OBJECT_TO_IDX["door"]; free &= ~(door&(image[...,2]!=STATE_TO_IDX["open"])); adj=np.zeros_like(v); adj[1:]|=v[:-1]; adj[:-1]|=v[1:]; adj[:,1:]|=v[:,:-1]; adj[:,:-1]|=v[:,1:]; return (free&~v&adj).astype(np.float32)
    def _policy_state(self,obs,token):
        image=np.asarray(obs,dtype=np.int64); self._check(image); x,y=_pos(image); direction=int(image[x,y,2]); image=image.copy(); image[x,y,2]=0; grid=np.concatenate((image,(image[...,0]==MINIGRID_PLAYER_ID)[...,None],self._visited_positions[...,None],self._frontier(image)[...,None]),-1); side=max(grid.shape[:2]); padded=np.zeros((side,side,grid.shape[2]),dtype=grid.dtype); padded[...,0]=OBJECT_TO_IDX["wall"]; top,left=(side-grid.shape[0])//2,(side-grid.shape[1])//2; padded[top:top+grid.shape[0],left:left+grid.shape[1]]=grid; grid=np.rot90(padded,(-direction)%4).transpose(2,0,1); state=np.zeros((13,*grid.shape[1:]),np.float32); state[:6]=grid; state[6+int(token)]=1; return torch.as_tensor(state,device=_DEVICE)
    def select_action(self,obs):
        self._mark(obs); state=self._policy_state(obs,self.current_inventory_token); action,state,at,lp,value=self.ppo.select_action(state,self._rng); self._pending_transition=(state,at,lp,value); return action
    def intrinsic_reward(self,obs,action,next_obs):
        next_position=_pos(next_obs); reward=float(not self._visited_positions[next_position]); moved=_pos(obs)!=next_position; changed=not np.array_equal(np.asarray(obs),np.asarray(next_obs)) or self.current_inventory_token!=self.next_inventory_token; self._mark(next_obs); self._rollout_transitions+=1; self._rollout_moved+=int(moved); self._rollout_changed+=int(changed); self._rollout_no_effect+=int(not changed); self._rollout_rewarded+=int(reward>0); self._rollout_reward_sum+=reward; self._rollout_action_count[int(action)]+=1
        if changed and int(action) in {3,4,5}: self._rollout_interaction_successes[{3:"pickup",4:"toggle",5:"drop"}[int(action)]]+=1
        return reward
    def record_transition(self,reward,done,*,obs_next=None,terminated=None,truncated=None,env_reward=None,**_):
        if not self.training_enabled: self._pending_transition=None; return
        if not self._iteration_active: raise RuntimeError("Training transitions require begin_iteration before collection")
        if self._pending_transition is None: raise RuntimeError("MiniGrid transition has no sampled PPO action")
        terminal = (
            bool(done) if terminated is None and truncated is None
            else bool(terminated) if terminated is not None
            else bool(done) and not bool(truncated)
        )
        adjusted_reward = float(reward)
        # MiniGrid lava and other unsuccessful absorbing terminations expose
        # terminated=True with a non-positive native reward.  Truncations are
        # time/collection boundaries and must not receive this penalty.
        failed_termination = terminal and float(env_reward if env_reward is not None else 0.0) <= 0.0
        if failed_termination and self.death_penalty > 0.0:
            adjusted_reward -= self.death_penalty
            self._rollout_deaths += 1
            self._rollout_reward_sum -= self.death_penalty
        state,action,lp,value=self._pending_transition; bootstrap=0. if terminal or obs_next is None else self.ppo.estimate(self._policy_state(obs_next,self.next_inventory_token)); self.ppo.save(state,action,lp,value,adjusted_reward,terminal,bootstrap); self.ppo.buffer.boundaries[-1]=bool(done); self._pending_transition=None
        self._rollout_transition_count += 1
        if self.ppo_updates_per_rollout > 1 and self._next_rollout_update < len(self._rollout_update_boundaries) and self._rollout_transition_count >= self._rollout_update_boundaries[self._next_rollout_update]:
            self._flush_rollout_segment()
    def mark_rollout_boundary(self):
        if self.training_enabled and self.ppo.buffer.transition_count():
            self.ppo.buffer.boundaries[-1]=True
            if self.ppo_updates_per_rollout > 1: self._flush_rollout_segment()
        self._pending_transition=None

    def _flush_rollout_segment(self):
        """Update once without resetting the current MiniGrid environment."""
        if not self.ppo.buffer.transition_count(): return
        # The final transition bootstraps from policy_old, then this update
        # synchronizes policy_old for the next segment on the same map.
        self.ppo.buffer.boundaries[-1]=True
        self._iteration_updates.append(self.ppo.update(self._rng))
        self._next_rollout_update += 1

    def _aggregate_iteration_updates(self):
        updates=self._iteration_updates
        if not updates:
            return {"updated":False,"update_count":self.ppo.update_count,"transition_count":0,"rollout_size":0,"optimizer_steps":0,"rollout_updates":0}
        transitions=sum(int(m["transition_count"]) for m in updates)
        weighted=lambda key: sum(float(m.get(key,0.0))*int(m["transition_count"]) for m in updates)/max(transitions,1)
        return {"updated":True,"update_count":self.ppo.update_count,"transition_count":transitions,"rollout_size":transitions,"optimizer_steps":sum(int(m["optimizer_steps"]) for m in updates),"approx_kl":weighted("approx_kl"),"clip_fraction":weighted("clip_fraction"),"entropy":weighted("entropy"),"rollout_updates":len(updates)}
    @property
    def unique_semantic_states(self): return self.episodic_visited_positions
    @property
    def episodic_visited_positions(self): return int(np.count_nonzero(self._visited_positions))
    @property
    def visited_position_mask(self):
        """Copy of the map-local visited mask for coverage diagnostics."""
        return self._visited_positions.astype(bool).copy()
    @property
    def rollout_metrics(self):
        n=max(self._rollout_transitions,1); return {"unique_positions":float(self.episodic_visited_positions),"new_position_rate":self._rollout_rewarded/n,"movement_rate":self._rollout_moved/n,"no_effect_rate":self._rollout_no_effect/n,"intrinsic_reward_mean":self._rollout_reward_sum/n,"death_count":float(self._rollout_deaths),"action_distribution":(self._rollout_action_count/n).tolist(),"pickup_successes":float(self._rollout_interaction_successes["pickup"]),"toggle_successes":float(self._rollout_interaction_successes["toggle"]),"drop_successes":float(self._rollout_interaction_successes["drop"])}
    @property
    def update_count(self): return self.ppo.update_count
    @property
    def rollout_update_count(self): return self._next_rollout_update
    def _metadata(self):
        p=self.ppo.policy; return {"architecture":_ARCH,"feature_dim":p.feature_dim,"object_embedding_dim":p.object_embedding.embedding_dim,"color_embedding_dim":p.color_embedding.embedding_dim,"state_embedding_dim":p.state_embedding.embedding_dim,"conv_channels":p.conv[0].out_channels,"intrinsic_reward":"map_local_first_visit_position_v1","death_penalty":self.death_penalty,"count_scope":"map_local_v1","policy_spatial_pool_size":p.spatial_pool_size,"action_dim":p.actor_head[-1].out_features,"object_vocab_size":p.object_embedding.num_embeddings,"color_vocab_size":p.color_embedding.num_embeddings,"state_vocab_size":p.state_embedding.num_embeddings,"object_vocab":sorted(OBJECT_TO_IDX.items()),"color_vocab":sorted(COLOR_TO_IDX.items()),"state_vocab":sorted(STATE_TO_IDX.items()),"inventory_schema":["empty"]+[name for name,_ in sorted(COLOR_TO_IDX.items(),key=lambda item:item[1])],"action_schema":["left","right","forward","pickup","toggle","drop"]}
    def _checkpoint_file(self,path):
        resolved=path or self.checkpoint_path
        if not resolved: raise ValueError("Explorer checkpoint path is not configured")
        return Path(str(resolved)).expanduser()
    def save_checkpoint(self,path=None,completed_iteration=None):
        if self._iteration_active or self.ppo.buffer.transition_count() or self._pending_transition is not None: raise RuntimeError("Explorer checkpoints may only be saved at an iteration boundary")
        target=self._checkpoint_file(path); target.parent.mkdir(parents=True,exist_ok=True); tmp=target.with_suffix(target.suffix+".tmp"); torch.save({"metadata":self._metadata(),"policy":self.ppo.policy.state_dict(),"policy_old":self.ppo.policy_old.state_dict(),"optimizer":self.ppo.optimizer.state_dict(),"update_count":self.ppo.update_count,"completed_iteration":self.completed_iteration if completed_iteration is None else int(completed_iteration),"rng_state":self._rng.get_state()},tmp); os.replace(tmp,target)
    def load_checkpoint(self,path=None):
        source=self._checkpoint_file(path)
        if not source.is_file(): raise FileNotFoundError(f"Explorer checkpoint not found: {source}")
        ckpt=torch.load(source,map_location="cpu",weights_only=False)
        if ckpt.get("metadata")!=self._metadata(): raise ValueError("Explorer checkpoint is incompatible with the configured spatial policy")
        self.ppo.policy.load_state_dict(ckpt["policy"]); self.ppo.policy_old.load_state_dict(ckpt["policy_old"]); self.ppo.optimizer.load_state_dict(ckpt["optimizer"]); self.ppo.update_count=int(ckpt["update_count"]); self.completed_iteration=int(ckpt.get("completed_iteration",0)); self._rng.set_state(ckpt["rng_state"]); return self.completed_iteration
