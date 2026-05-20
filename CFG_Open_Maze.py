import gymnasium as gym
import minari
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.distributions import Bernoulli, Normal
import random
import wandb
from collections import defaultdict, deque
from typing import Self, List, Tuple, Dict, Iterator, Callable, Literal
from tqdm import trange, tqdm


class EnvWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        super(EnvWrapper, self).__init__(env)

    def observation(self, observation: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        return {k: v.astype(np.float32) for k, v in observation.items()}


class EpisodeDataset(Dataset):

    def __init__(self, data: List[Dict[str, torch.Tensor]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ep = self.data[idx]

        return {
            'observations': ep['observations'],
            'actions': ep['actions'],
            'rewards': ep['rewards'],
            'done': ep['done'],
            'goals': ep['goals']
        }

    def get_length(self, idx: int) -> int:
        return self.data[idx]['actions'].shape[0]

    @classmethod
    def create(cls, minari_dataset: minari.MinariDataset, max_length: int = 300, seed=42) -> Self:
        rng = random.Random(seed)
        data = []

        for ep_id in trange(minari_dataset.total_episodes):
            ep = minari_dataset[ep_id]

            observations = ep.observations['observation']
            actions = ep.actions
            rewards = ep.rewards
            done = ep.truncations
            goals = ep.observations['achieved_goal']

            if len(actions) > max_length:
                random_len = rng.randint(2, max_length)

                observations = observations[:random_len + 1]
                actions = actions[:random_len]
                rewards = rewards[:random_len]
                done = done[:random_len]
                goals = goals[:random_len + 1]

                rewards[-1] = 1.0
                done[-1] = True

            observations = torch.as_tensor(observations, dtype=torch.float32)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            done = torch.as_tensor(done, dtype=torch.float32)
            goals = torch.as_tensor(goals, dtype=torch.float32)

            data.append({
                'observations': observations,
                'actions': actions,
                'rewards': rewards,
                'done': done,
                'goals': goals
            })

        return cls(data)

    @classmethod
    def create_fixed_length(cls, minari_dataset: minari.MinariDataset, max_length: int = 32, stride: int | None = None, force_done: bool = True, force_reward: bool = True) -> Self:
        if stride is None:
            stride = max_length

        data = []

        for ep_id in trange(minari_dataset.total_episodes):
            ep = minari_dataset[ep_id]

            observations = ep.observations['observation']
            actions = ep.actions
            rewards = ep.rewards
            done = ep.truncations
            goals = ep.observations['achieved_goal']

            T = len(actions)

            if T < max_length:
                continue

            for start in range(0, T - max_length + 1, stride):
                end = start + max_length

                seg_observations = observations[start:end + 1]
                seg_actions = actions[start:end]
                seg_rewards = rewards[start:end].copy()
                seg_done = done[start:end].copy()
                seg_goals = goals[start:end + 1]

                if force_done:
                    seg_done[-1] = True

                if force_reward:
                    seg_rewards[-1] = 1.0

                seg_observations = torch.as_tensor(seg_observations, dtype=torch.float32)
                seg_actions = torch.as_tensor(seg_actions, dtype=torch.float32)
                seg_rewards = torch.as_tensor(seg_rewards, dtype=torch.float32)
                seg_done = torch.as_tensor(seg_done, dtype=torch.float32)
                seg_goals = torch.as_tensor(seg_goals, dtype=torch.float32)

                data.append({
                    'observations': seg_observations,
                    'actions': seg_actions,
                    'rewards': seg_rewards,
                    'done': seg_done,
                    'goals': seg_goals
                })

        return cls(data)


class HierarchicalDataset(Dataset):

    def __init__(self, data: List[Dict[str, torch.Tensor]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.data[idx]

    def get_length(self, idx: int) -> Tuple[int, int]:
        high_length = self.data[idx]['high_observations'].shape[0] - 1
        low_length = self.data[idx]['low_observations'].shape[0]
        return high_length, low_length

    @classmethod
    def create(cls, minari_dataset: minari.MinariDataset, chunk_size: int = 4, max_length: int = 300, seed: int = 42) -> Self:
        rng = random.Random(seed)
        data = []

        for ep_id in trange(minari_dataset.total_episodes):
            ep = minari_dataset[ep_id]

            observations = ep.observations['observation']
            actions = ep.actions
            rewards = ep.rewards
            done = ep.truncations
            goals = ep.observations['achieved_goal']

            T = len(actions)

            if T < chunk_size:
                continue

            if T > max_length:
                random_len = rng.randint(chunk_size, max_length)

                observations = observations[:random_len + 1]
                actions = actions[:random_len]
                rewards = rewards[:random_len]
                done = done[:random_len]
                goals = goals[:random_len + 1]

                rewards[-1] = 1.0
                done[-1] = True

                T = random_len

            for offset in range(chunk_size):
                remaining_T = T - offset

                if remaining_T < chunk_size:
                    continue

                usable_T = (remaining_T // chunk_size) * chunk_size

                high_observations = observations[offset : offset + usable_T + 1 : chunk_size]
                high_goals = goals[offset : offset + usable_T + 1 : chunk_size]

                low_observations = []
                low_goals = []
                low_actions = []

                for t in range(offset, offset + usable_T - chunk_size + 1):
                    low_observations.append(observations[t])
                    low_goals.append(goals[t + chunk_size])
                    low_actions.append(actions[t:t + chunk_size])

                if len(low_observations) == 0:
                    continue

                data.append({
                    'high_observations': torch.as_tensor(high_observations, dtype=torch.float32),
                    'high_goals': torch.as_tensor(high_goals, dtype=torch.float32),

                    'low_observations': torch.as_tensor(np.asarray(low_observations), dtype=torch.float32),
                    'low_goals': torch.as_tensor(np.asarray(low_goals), dtype=torch.float32),
                    'low_actions': torch.as_tensor(np.asarray(low_actions), dtype=torch.float32),

                    'rewards': torch.as_tensor(rewards[offset:offset + usable_T], dtype=torch.float32),
                    'done': torch.as_tensor(done[offset:offset + usable_T], dtype=torch.float32),
                })

        return cls(data)

    @classmethod
    def create_fixed_length(cls, minari_dataset: minari.MinariDataset, chunk_size: int = 4, max_length: int = 32, stride: int | None = None, force_done: bool = True, force_reward: bool = True) -> Self:
        if stride is None:
            stride = max_length

        data = []

        for ep_id in trange(minari_dataset.total_episodes):
            ep = minari_dataset[ep_id]

            observations = ep.observations['observation']
            actions = ep.actions
            rewards = ep.rewards
            done = ep.truncations
            goals = ep.observations['achieved_goal']

            T = len(actions)

            if T < max_length:
                continue

            for start in range(0, T - max_length + 1, stride):
                for offset in range(chunk_size):
                    real_start = start + offset
                    real_end = real_start + max_length

                    if real_end > T:
                        continue

                    seg_observations = observations[real_start:real_end + 1]
                    seg_actions = actions[real_start:real_end]
                    seg_rewards = rewards[real_start:real_end].copy()
                    seg_done = done[real_start:real_end].copy()
                    seg_goals = goals[real_start:real_end + 1]

                    if force_done:
                        seg_done[-1] = True

                    if force_reward:
                        seg_rewards[-1] = 1.0

                    high_observations = seg_observations[0:max_length + 1:chunk_size]
                    high_goals = seg_goals[0:max_length + 1:chunk_size]

                    low_observations = []
                    low_goals = []
                    low_actions = []

                    for t in range(0, max_length - chunk_size + 1):
                        low_observations.append(seg_observations[t])
                        low_goals.append(seg_goals[t + chunk_size])
                        low_actions.append(seg_actions[t:t + chunk_size])

                    data.append({
                        'high_observations': torch.as_tensor(high_observations, dtype=torch.float32),
                        'high_goals': torch.as_tensor(high_goals, dtype=torch.float32),

                        'low_observations': torch.as_tensor(np.asarray(low_observations), dtype=torch.float32),
                        'low_goals': torch.as_tensor(np.asarray(low_goals), dtype=torch.float32),
                        'low_actions': torch.as_tensor(np.asarray(low_actions), dtype=torch.float32),

                        'rewards': torch.as_tensor(seg_rewards, dtype=torch.float32),
                        'done': torch.as_tensor(seg_done, dtype=torch.float32),
                    })

        return cls(data)


class GroupByLengthSampler(Sampler):

    def __init__(self, all_batches: List[List[int]]) -> None:
        self.all_batches = all_batches

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.all_batches

    def __len__(self) -> int:
        return len(self.all_batches)

    @staticmethod
    def estimate_batch_size(length: int, base_memory: int, bucket_size: int, max_batch_size: int | None = None) -> int:
        batch_size = min(max((base_memory // (length ** 3 * 4)), 1), bucket_size)

        if max_batch_size is not None:
            batch_size = min(batch_size, max_batch_size)

        return batch_size

    @classmethod
    def create(cls, dataset: EpisodeDataset | HierarchicalDataset,
               base_memory: int = 28 * 1024 * 1024,
               max_batch_size: int | None = 16,
               shuffle: bool = False,
               drop_last: bool = False) -> Self:
        length_to_idx = defaultdict(list)
        all_batches = []

        for i in range(len(dataset)):
            length = dataset.get_length(i)
            length_to_idx[length].append(i)

        for length, idx in length_to_idx.items():
            if isinstance(length, int):
                batch_size = cls.estimate_batch_size(length, base_memory, len(idx), max_batch_size)
            else:
                batch_size = cls.estimate_batch_size(max(length[0], length[1]), base_memory, len(idx), max_batch_size)

            if shuffle:
                random.shuffle(idx)

            for i in range(0, len(idx), batch_size):
                batch = idx[i:i + batch_size]
                if drop_last and len(batch) < batch_size:
                    continue
                if len(batch) > 0:
                    all_batches.append(batch)

        if shuffle:
            random.shuffle(all_batches)

        return cls(all_batches)


def episode_random_future_goal_collate_fn(batch) -> Dict[str, torch.Tensor]:
    obs_batch = []
    goal_batch = []
    action_batch = []
    reward_batch = []
    done_batch = []

    for ep in batch:
        observations = ep['observations']
        actions = ep['actions']
        goals = ep['goals']
        rewards = ep['rewards']
        done = ep['done']

        T = actions.shape[0]
        obs = observations[:T]
        future_goal_indices = torch.empty(T, dtype=torch.int64)

        for i in range(T):
            future_goal_indices[i] = torch.randint(low=i + 1, high=T + 1, size=(1,))
        goals = goals[future_goal_indices]

        obs_batch.append(obs)
        goal_batch.append(goals)
        action_batch.append(actions)
        reward_batch.append(rewards)
        done_batch.append(done)

    return {
        'observations': torch.stack(obs_batch, dim=0),
        'goals': torch.stack(goal_batch, dim=0),
        'actions': torch.stack(action_batch, dim=0),
        'rewards': torch.stack(reward_batch, dim=0),
        'done': torch.stack(done_batch, dim=0),
    }


def hierarchical_random_future_goal_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    high_obs_batch = []
    high_goal_batch = []
    low_obs_batch = []
    low_goal_batch = []
    low_action_batch = []
    reward_batch = []
    done_batch = []

    for ep in batch:
        high_observations = ep['high_observations']  # (H + 1, obs_dim)
        high_goals = ep['high_goals']                # (H + 1, goal_dim)
        H = high_observations.shape[0] - 1
        high_obs = high_observations[:H]
        high_goal_indices = torch.empty(H, dtype=torch.long)

        for i in range(H):
            high_goal_indices[i] = torch.randint(low=i + 1, high=H + 1, size=(1,))

        sampled_high_goals = high_goals[high_goal_indices]
        low_observations = ep['low_observations']  # (L, obs_dim)
        low_goals = ep['low_goals']                # (L, goal_dim)
        low_actions = ep['low_actions']            # (L, chunk_size, action_dim)
        L = low_observations.shape[0]
        low_goal_indices = torch.empty(L, dtype=torch.int64)

        for i in range(L):
            low_goal_indices[i] = torch.randint(low=i, high=L, size=(1,))

        sampled_low_goals = low_goals[low_goal_indices]
        high_obs_batch.append(high_obs)
        high_goal_batch.append(sampled_high_goals)
        low_obs_batch.append(low_observations)
        low_goal_batch.append(sampled_low_goals)
        low_action_batch.append(low_actions)
        reward_batch.append(ep['rewards'])
        done_batch.append(ep['done'])

    return {
        'high_observations': torch.stack(high_obs_batch, dim=0),
        'high_goals': torch.stack(high_goal_batch, dim=0),
        'low_observations': torch.stack(low_obs_batch, dim=0),
        'low_goals': torch.stack(low_goal_batch, dim=0),
        'low_actions': torch.stack(low_action_batch, dim=0),
        'rewards': torch.stack(reward_batch, dim=0),
        'done': torch.stack(done_batch, dim=0),
    }


class MLP(nn.Module):

    def __init__(self, dims: List[int],
                 activation_fn: Callable[[], nn.Module] | None = nn.ReLU,
                 output_activation_fn: Callable[[], nn.Module] | None = None,
                 dropout: float | None = None) -> None:
        super(MLP, self).__init__()

        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ShouldSplitNet(nn.Module):

    def __init__(self, dims,
                 activation_fn: Callable[[], nn.Module] = nn.ReLU,
                 output_activation_fn: Callable[[], nn.Module] = None,
                 dropout: float = None) -> None:
        super(ShouldSplitNet, self).__init__()

        self.net = MLP(
            dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )

    def forward(self, observations: torch.Tensor, goals: torch.Tensor) -> Bernoulli:
        delta = goals - observations[..., :goals.shape[-1]]
        dist = torch.norm(delta, dim=-1, keepdim=True)
        logits = self.net(torch.cat([observations, goals, delta, dist], dim=-1)).squeeze(-1)
        return Bernoulli(logits=logits)

    def compute_terminal_likelihood(self, observations: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        B, n = observations.shape[0], observations.shape[1] - 1

        s_i = observations[:, :n].unsqueeze(2).expand(B, n, n, -1)
        g_jp1 = goals[:, 1:].unsqueeze(1).expand(B, n, n, -1)
        terminal = torch.eye(n, dtype=torch.float32, device=observations.device).unsqueeze(0)

        dist = self(s_i, g_jp1)
        T = dist.log_prob(terminal)  # shape: (B, n, n)

        T_mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=observations.device))
        T = T.masked_fill(~T_mask.unsqueeze(0), -float('inf'))
        return T

    @classmethod
    def create(cls, dims,
               activation_fn: Callable[[], nn.Module] = nn.ReLU,
               output_activation_fn: Callable[[], nn.Module] = None,
               dropout: float = None) -> Self:

        return cls(
            dims=dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )


class MidGoalPredictor(nn.Module):

    def __init__(self, dims, activation_fn: Callable[[], nn.Module] = nn.ReLU,
                 output_activation_fn: Callable[[], nn.Module] = None,
                 dropout: float = None) -> None:
        super(MidGoalPredictor, self).__init__()

        self.net = MLP(
            dims=dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )
        self.log_std = nn.Parameter(torch.zeros(dims[-1], dtype=torch.float32))

    def forward(self, observations: torch.Tensor, goals: torch.Tensor) -> Normal:
        mean = self.net(torch.cat([observations, goals], dim=-1))
        std = torch.exp(self.log_std.clamp(min=0.0, max=0.0))
        return Normal(mean, std)

    def compute_planed_likelihood(self, observations: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        B, n = observations.shape[0], observations.shape[1] - 1

        s_i = observations[:, :n-1].unsqueeze(2).unsqueeze(3).expand(B, n-1, n-1, n-1, -1)            # shape: (B, n-1, n-1, n-1, obs_dim)
        g_jp2 = goals[:, 2:n+1].unsqueeze(1).unsqueeze(3).expand(B, n-1, n-1, n-1, -1)                # shape: (B, n-1, n-1, n-1, goal_dim)
        s_kp1 = observations[:, 1:n].unsqueeze(1).unsqueeze(1)[..., :2].expand(B, n-1, n-1, n-1, -1)  # shape: (B, n-1, n-1, n-1, goal_dim)

        dist = self(s_i, g_jp2)
        P = dist.log_prob(s_kp1).sum(dim=-1)  # shape: (B, n-1, n-1, n-1)

        idx = torch.arange((n - 1), dtype=torch.int64, device=observations.device)
        k, i, j = torch.meshgrid(idx, idx, idx, indexing='ij')
        P_mask = (i <= k) & (k <= j)
        P = P.masked_fill(~P_mask.unsqueeze(0), -float('inf'))
        return P

    @classmethod
    def create(cls, dims,
               activation_fn: Callable[[], nn.Module] = nn.ReLU,
               output_activation_fn: Callable[[], nn.Module] = None,
               dropout: float = None) -> Self:

        return cls(
            dims=dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )


class PolicyNet(nn.Module):

    def __init__(self, dims,
                 activation_fn: Callable[[], nn.Module] = nn.ReLU,
                 output_activation_fn: Callable[[], nn.Module] = nn.Tanh,
                 dropout: float = None) -> None:
        super(PolicyNet, self).__init__()

        self.net = MLP(
            dims=dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )
        self.log_std = nn.Parameter(torch.zeros(dims[-1], dtype=torch.float32))

    def forward(self, observations: torch.Tensor, goals: torch.Tensor) -> Normal:
        mean = self.net(torch.cat([observations, goals], dim=-1))
        std = torch.exp(self.log_std.clamp(min=0.0, max=0.0))
        return Normal(mean, std)

    def compute_actor_likelihood(self, observations: torch.Tensor, goals: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        B, n = observations.shape[0], observations.shape[1] - 1

        s_i = observations[:, :n]  # shape: (B, n, obs_dim)
        g_ip1 = goals[:, 1:]       # shape: (B, n, goal_dim)
        a_i = actions[:, :n]       # shape: (B, n, action_dim)

        dist = self(s_i, g_ip1)
        A = dist.log_prob(a_i).sum(dim=-1)  # shape: (B, n)
        return A

    @classmethod
    def create(cls, dims,
               activation_fn: Callable[[], nn.Module] = nn.ReLU,
               output_activation_fn: Callable[[], nn.Module] = nn.Tanh,
               dropout: float = None) -> Self:

        return cls(
            dims=dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )


class LowLevelPolicyNet(nn.Module):

    def __init__(self, dims,
                 chunk_size: int,
                 activation_fn: Callable[[], nn.Module] = nn.ReLU,
                 output_activation_fn: Callable[[], nn.Module] = nn.Tanh,
                 dropout: float = None) -> None:
        super(LowLevelPolicyNet, self).__init__()

        self.chunk_size = chunk_size
        self.action_dim = dims[-1] // chunk_size

        self.net = MLP(
            dims=dims,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )
        self.log_std = nn.Parameter(torch.zeros(chunk_size, self.action_dim, dtype=torch.float32))

    def forward(self, observations: torch.Tensor, goals: torch.Tensor) -> Normal:
        mean = self.net(torch.cat([observations, goals], dim=-1))
        mean = mean.view(*mean.shape[:-1], self.chunk_size, self.action_dim)
        std = torch.exp(self.log_std.clamp(min=0.0, max=0.0))
        return Normal(mean, std)

    def compute_low_level_likelihood(self, observations: torch.Tensor, goals: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        dist = self(observations, goals)
        log_prob = dist.log_prob(actions)
        low_A = log_prob.sum(dim=(-1, -2))
        return low_A

    @classmethod
    def create(cls, dims,
               chunk_size: int,
               activation_fn: Callable[[], nn.Module] = nn.ReLU,
               output_activation_fn: Callable[[], nn.Module] = nn.Tanh,
               dropout: float = None) -> Self:

        return cls(
            dims=dims,
            chunk_size=chunk_size,
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )


class PlanningAgent(nn.Module):

    def __init__(self, terminal_dims: List,
                 mid_goal_dims: List,
                 actor_dims: List,
                 chunk_size: int | None = None,
                 max_split_steps: int = 50) -> None:
        super(PlanningAgent, self).__init__()

        self.terminal = ShouldSplitNet.create(terminal_dims)
        self.mid_goal = MidGoalPredictor.create(mid_goal_dims)
        if chunk_size is None:
            self.actor = PolicyNet.create(actor_dims)
        else:
            self.actor = LowLevelPolicyNet(actor_dims, chunk_size)
        self.max_split_steps = max_split_steps

    def act_agent(self, observations: torch.Tensor, goals: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        split_step = 1
        mid_goal_list = [goals]

        while self.terminal(observations, goals).mean < 0.5:  # where P_terminal < 0.5
            if split_step >= self.max_split_steps:
                break

            split_step += 1
            goals = self.mid_goal(observations, goals).mean
            mid_goal_list.append(goals)

        mid_goal_list.reverse()

        return self.actor(observations, goals).mean, mid_goal_list

    def act_baseline(self, observations: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
        return self.actor(observations, goals).mean


def compute_total_likelihood(T: torch.Tensor,  # shape: (B, n, n)
                             P: torch.Tensor,  # shape: (B, n-1, n-1, n-1)
                             A: torch.Tensor | None = None,  # shape: (B, n)
                             use_optimal_path: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, n, _ = T.shape
    L = torch.full((B, n, n), -float('inf'), dtype=torch.float32, device=T.device)
    optimal_idx = torch.full((B, n, n), -1, dtype=torch.int64, device=T.device) if use_optimal_path else None
    tree_depth = torch.full((B, n, n), 1, dtype=torch.int64, device=T.device)

    diag = torch.arange(n, dtype=torch.int64, device=T.device)
    if A is None:
        L[:, diag, diag] = T[:, diag, diag]
    else:
        L[:, diag, diag] = A + T[:, diag, diag]

    for d in range(1, n):
        i = torch.arange(n - d, dtype=torch.int64, device=T.device)
        j = i + d
        k = i[:, None] + torch.arange(d, dtype=torch.int64, device=T.device)[None, :]  # shape: (n-d, d)

        p_mid = P[:, k, i[:, None], j[:, None] - 1]  # shape: (B, n-d, d)
        l1 = L[:, i[:, None], k]                     # shape: (B, n-d, d)
        l2 = L[:, k + 1, j[:, None]]                 # shape: (B, n-d, d)

        log_terms = p_mid + l1 + l2                  # shape: (B, n-d, d)

        if use_optimal_path:
            optimal, argmax_k = torch.max(log_terms, dim=2)
            L[:, i, j] = T[:, i, j] + optimal

            abs_k = i.unsqueeze(0) + argmax_k        # shape: (B, n-d)
            optimal_idx[:, i, j] = abs_k

            b_idx = torch.arange(B, dtype=torch.int64, device=T.device)[:, None]     # shape: (B, 1)
            i_idx = i[None, :]
            j_idx = j[None, :]
            left_depth = tree_depth[b_idx, i_idx, abs_k]
            right_depth = tree_depth[b_idx, abs_k + 1, j_idx]
            depth = torch.maximum(left_depth, right_depth) + 1
            tree_depth[:, i, j] = depth
        else:
            L[:, i, j] = T[:, i, j] + torch.logsumexp(log_terms, dim=2)

    return L, optimal_idx, tree_depth                # shape: (B, n, n)


def recover_optimal_trajectory(goals: torch.Tensor, optimal_idx: torch.Tensor) -> Tuple[List[torch.Tensor], List[Dict]]:
    B, n = goals.shape[0], goals.shape[1] - 1
    optimal_paths = []
    tree_structures = []

    def build_tree(idx: torch.Tensor, i: int, j: int) -> None | Dict:
        k = idx[i, j].item()
        if k == -1:
            return None
        node = {
            'idx': k + 1,
            'left': build_tree(idx, i, k) if k >= i else None,
            'right': build_tree(idx, k + 1, j) if k + 1 <= j else None
        }
        return node

    def inorder_traverse(tree: None | Dict) -> List[int]:
        if tree is None:
            return []
        return inorder_traverse(tree['left']) + [tree['idx']] + inorder_traverse(tree['right'])

    for b in range(B):
        idx = optimal_idx[b]
        tree = build_tree(idx, 0, n - 1)
        tree_structures.append(tree)

        mid_goal_idx = inorder_traverse(tree)
        mid_goals = [goals[b, m] for m in mid_goal_idx]
        trajectory = torch.stack([goals[b, 0]] + mid_goals + [goals[b, -1]], dim=0)
        optimal_paths.append(trajectory)

    return optimal_paths, tree_structures


def compute_mid_goal_achieved(trajectory_info: List[Dict[str, np.ndarray]],
                              mid_goal_list_info: List[List[torch.Tensor]],
                              max_split_step: int = 50,
                              threshold: float = 0.45) -> Tuple[List[int], List[int]]:
    n_level_goal_achieved = [0] * max_split_step
    n_level_goal_counts = [0] * max_split_step

    for i, mid_goal_list in enumerate(mid_goal_list_info[:-1]):
        mid_goal = [mg.detach().cpu().numpy() for mg in mid_goal_list]
        trajectory_achieved_goal = [step['achieved_goal'] for step in trajectory_info[i + 1:]]

        for j in range(len(mid_goal_list)):
            n_level_goal_counts[j] += 1

        if float(np.linalg.norm(trajectory_achieved_goal[0] - mid_goal[0])) < threshold:
            n_level_goal_achieved[0] += 1

            target_idx = 1
            for achieved_goal in trajectory_achieved_goal[1:]:
                if target_idx >= len(mid_goal):
                    break

                if float(np.linalg.norm(achieved_goal - mid_goal[target_idx])) < threshold:
                    n_level_goal_achieved[target_idx] += 1
                    target_idx += 1

    return n_level_goal_achieved, n_level_goal_counts


def evaluate(agent: nn.Module,
             env: gym.Env,
             chunk_size: int | None = None,
             execute_mode: Literal['chunk', 'first'] = 'chunk',
             epoches: int = 10,
             use_baseline: bool = False,
             device: str = 'cpu') -> Tuple[List[float], List[int], List[int], List[List[float]], List[float]]:

    agent.eval()

    epoch = 0
    roll_step_info = []
    reward_info = []
    distance_info = []
    success_info = []
    success_rate_info = []

    pbar = tqdm(total=epoches, leave=True, dynamic_ncols=True)

    while epoch < epoches:
        trajectory_info = []
        mid_goal_list_info = []

        obs_dict, info = env.reset()
        observations = torch.as_tensor(obs_dict['observation'], device=device)
        goals = torch.as_tensor(obs_dict['desired_goal'], device=device)
        done = info['success']
        distance_info.append(torch.norm(goals - observations[:goals.shape[-1]]).item())
        roll_reward = 0.0
        roll_step = 0

        while not done:
            with torch.no_grad():
                if not use_baseline:
                    actions, mid_goal_list = agent.act_agent(observations, goals)
                else:
                    actions = agent.act_baseline(observations, goals)
                    mid_goal_list = None

                if chunk_size is None:
                    actions_to_execute = actions.unsqueeze(0)
                else:
                    if execute_mode == 'chunk':
                        actions_to_execute = actions[:chunk_size]
                    else:
                        actions_to_execute = actions[:1]

                for action in actions_to_execute:
                    obs_dict, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                    observations = torch.as_tensor(obs_dict['observation'], device=device)
                    goals = torch.as_tensor(obs_dict['desired_goal'], device=device)
                    done = info['success'] or truncated
                    roll_reward += reward
                    roll_step += 1

                    if roll_step >= 300:
                        done = True

                    trajectory_info.append(obs_dict)
                    if mid_goal_list is not None:
                        mid_goal_list_info.append(mid_goal_list)

                    if done:
                        break

        if not use_baseline:
            n_level_goal_achieved, n_level_goal_counts = compute_mid_goal_achieved(trajectory_info, mid_goal_list_info)
            success_rate = [float(g / c) if c != 0 else 0 for g, c in zip(n_level_goal_achieved, n_level_goal_counts)]
        else:
            success_rate = []

        roll_step_info.append(roll_step)
        success_info.append(int(float(reward)))
        success_rate_info.append(success_rate)
        reward_info.append(roll_reward)

        epoch += 1
        pbar.update(1)
        pbar.set_description(f'Testing Epoch: {epoch} / {epoches}, Reward: {roll_reward}, Mean = {np.nanmean(reward_info) if reward_info else 0:.9f}, Std = {np.nanstd(reward_info) if reward_info else 0:.9f}')

    pbar.close()

    return distance_info, success_info, roll_step_info, success_rate_info, reward_info


def train(agent: PlanningAgent,
          env: gym.Env,
          dataloader: DataLoader,
          max_length: int,
          batch_size: int,
          hidden_dims: int,
          use_fixed_length: bool,
          optimizer: torch.optim.Optimizer | None = None,
          use_average_model: bool = True,
          chunk_size: int | None = None,
          execute_mode: Literal['chunk', 'first'] = 'chunk',
          low_actor_alpha: float = 1.0,
          use_wandb: bool = True,
          lr: float = 1e-5,
          weight_decay: float = 1e-5,
          train_steps: int = 200_000,
          eval_interval: int = 10_000,
          log_interval: int = 500,
          use_baseline: bool = False,
          planner_save_path: str = 'open_maze_planner_1.pt',
          l_save_path: str = 'log_l_1.pt',
          optimal_idx_save_path: str = 'log_optimal_idx_1.pt',
          tree_depth_save_path: str = 'log_tree_depth_1.pt',
          device: str = 'cpu') -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[List[float]], List[List[int]], List[List[int]]]:

    is_hierarchical = chunk_size is not None

    if use_wandb:
        wandb.init(
            mode='online',
            project='Open Maze Training Test',
            name=f'Use Fixed Length: {use_fixed_length} Length: {max_length}, Batch Size: {batch_size}, Hidden Dims: {hidden_dims}, lr: {lr}, Use Average Model: {use_average_model}, Chunk Size: {chunk_size}, Execute Mode: {execute_mode}, Use Baseline {use_baseline}',
            config={
                'lr': lr,
                'batch_size': batch_size,
                'hidden_dims': hidden_dims,
                'use_fixed_length': use_fixed_length,
                'length': max_length,
                'chunk_size': chunk_size,
                'low_actor_alpha': low_actor_alpha,
                'is_hierarchical': is_hierarchical,
                'use_average_model': use_average_model,
                'use_baseline': use_baseline,
                'weight_decay': weight_decay,
                'train_steps': train_steps,
                'eval_interval': eval_interval,
                'log_interval': log_interval
            }
        )

    if optimizer is None:
        params = agent.actor.parameters() if use_baseline else agent.parameters()
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    if use_average_model:
        average_agent = AveragedModel(agent).to(device)
        average_agent.train()
    else:
        average_agent = None

    agent.train()

    step = 0
    eval_count = 0
    recent_loss = deque(maxlen=100)
    log_L = []
    log_optimal_idx = []
    log_tree_depth = []
    distance = []
    success = []
    roll = []
    success_rate = []

    pbar = tqdm(total=train_steps, leave=True, dynamic_ncols=True)

    while step < train_steps:
        for batch in dataloader:
            if step >= train_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}

            if not use_baseline:
                if is_hierarchical:
                    high_observations = batch['high_observations']
                    high_goals = batch['high_goals']

                    low_observations = batch['low_observations']
                    low_goals = batch['low_goals']
                    low_actions = batch['low_actions']

                    B, n = high_observations.shape[0], high_observations.shape[1] - 1

                    low_A = agent.actor.compute_low_level_likelihood(low_observations, low_goals, low_actions)

                    T = agent.terminal.compute_terminal_likelihood(high_observations, high_goals)
                    P = agent.mid_goal.compute_planed_likelihood(high_observations, high_goals)

                    L, optimal_idx, tree_depth = compute_total_likelihood(T, P, A=None)     # shape: (B, n, n)
                    n = L.shape[1]
                    high_loss = -L[:, 0, n - 1].mean() / n

                    actor_loss = -low_A.mean()

                    loss = high_loss + low_actor_alpha * actor_loss

                else:
                    observations = batch['observations']  # shape: (B, n+1, obs_dim)
                    actions = batch['actions']  # shape: (B, n, action_dim)
                    goals = batch['goals']  # shape: (B, n+1, goal_dim)
                    B, n = observations.shape[0], observations.shape[1] - 1

                    A = agent.actor.compute_actor_likelihood(observations, goals, actions)  # shape: (B, n)
                    T = agent.terminal.compute_terminal_likelihood(observations, goals)     # shape: (B, n, n)
                    P = agent.mid_goal.compute_planed_likelihood(observations, goals)       # shape: (B, n-1, n-1, n-1)

                    actor_loss = -A.mean()
                    L, optimal_idx, tree_depth = compute_total_likelihood(T, P, A=A)        # shape: (B, n, n)

                    n = L.shape[1]
                    loss = -L[:, 0, n - 1].mean() / n  # L[0, n-1] maximize likelihood
                    high_loss = loss

                T_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=device)).unsqueeze(0)  # shape: (1, n, n)
                T_ref = T[T_mask.expand(B, -1, -1)]
                terminal_loss = -T_ref.mean()

                log_L.append(L.detach().cpu())
                log_optimal_idx.append(optimal_idx.detach().cpu() if optimal_idx is not None else None)
                log_tree_depth.append(tree_depth.detach().cpu())

            else:
                if is_hierarchical:
                    low_observations = batch['low_observations']
                    low_goals = batch['low_goals']
                    low_actions = batch['low_actions']

                    low_A = agent.actor.compute_low_level_likelihood(low_observations, low_goals, low_actions)

                    actor_loss = -low_A.mean()

                    loss = actor_loss

                else:
                    observations = batch['observations']  # shape: (B, n+1, obs_dim)
                    actions = batch['actions']            # shape: (B, n, action_dim)
                    goals = batch['goals']                # shape: (B, n+1, goal_dim)

                    A = agent.actor.compute_actor_likelihood(observations, goals, actions)  # shape: (B, n)

                    actor_loss = -A.mean()

                    loss = actor_loss

                high_loss = torch.tensor(0.0, device=device)
                terminal_loss = torch.tensor(0.0, device=device)

            recent_loss.append(loss.item())
            average_loss = sum(recent_loss) / len(recent_loss)

            optimizer.zero_grad()
            loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(parameters=agent.actor.parameters(), max_norm=1, norm_type=2)
            if not use_baseline:
                terminal_grad = nn.utils.clip_grad_norm_(parameters=agent.terminal.parameters(), max_norm=1, norm_type=2)
                mid_goal_grad = nn.utils.clip_grad_norm_(parameters=agent.mid_goal.parameters(), max_norm=1, norm_type=2)
            else:
                terminal_grad = torch.tensor(0.0)
                mid_goal_grad = torch.tensor(0.0)
            optimizer.step()
            if use_average_model:
                average_agent.update_parameters(agent)

            step += 1
            pbar.update(1)
            pbar.set_description(f'Train Step: {step} / {train_steps}, Current Loss = {loss.item():.9f}, Average loss = {average_loss:.9f}')

            if use_wandb and step % log_interval == 0:
                wandb.log({
                    'Train/Current Loss': loss.item(),
                    'Train/High Loss': high_loss.item(),
                    'Train/Average Loss': average_loss,
                    'Train/Actor Loss': actor_loss.item(),
                    'Train/Terminal Loss': terminal_loss.item(),
                    'Train/Actor Grad': actor_grad,
                    'Train/Terminal Grad': terminal_grad,
                    'Train/Mid Goal Grad': mid_goal_grad
                }, step=step)

            if step % eval_interval == 0 or step == train_steps:
                eval_count += 1
                eval_agent = average_agent.module if use_average_model else agent
                distance_info, success_info, roll_step_info, success_rate_info, reward_info = evaluate(eval_agent, env, chunk_size=chunk_size, execute_mode=execute_mode, use_baseline=use_baseline, device=device)
                if use_average_model:
                    average_agent.train()
                agent.train()

                plot_distance_roll_scatter([distance_info], [roll_step_info], [success_info], subtitle=eval_count, eval_count=eval_count, save_path=f'Distance Roll Scatter Evaluate Epoch_{eval_count}')

                distance.append(distance_info)
                success.append(success_info)
                roll.append(roll_step_info)
                success_rate.append(success_rate_info)

                if use_wandb:
                    wandb.log({
                        'Evaluate/Distance Mean': np.mean(distance_info),
                        'Evaluate/Distance Std': np.std(distance_info),
                        'Evaluate/Success Mean': np.mean(success_info),
                        'Evaluate/Success Std': np.std(success_info),
                        'Evaluate/Roll Step Mean': np.mean(roll_step_info),
                        'Evaluate/Roll Step Std': np.std(roll_step_info),
                        'Evaluate/Total Reward Mean': np.mean(reward_info),
                        'Evaluate/Total Reward Std': np.std(reward_info)
                    }, step=step)

    pbar.close()

    if use_wandb:
        wandb.finish()

    agent.eval()
    torch.save(agent.state_dict(), planner_save_path)
    if use_average_model:
        torch.save(average_agent.module.state_dict(), planner_save_path.replace('.pt', '_average.pt'))
    print(f'Saved planner model to {planner_save_path}')

    torch.save(log_L, l_save_path)
    torch.save(log_optimal_idx, optimal_idx_save_path)
    torch.save(log_tree_depth, tree_depth_save_path)
    print('Saved all log files.')

    return log_L, log_optimal_idx, log_tree_depth, distance, success, roll


def plot_trajectory_length_histogram(dataset: EpisodeDataset | HierarchicalDataset, max_length: int, save_path: str | None = None) -> None:
    length = []

    if isinstance(dataset, EpisodeDataset):
        for trajectory in dataset:
            length.append(trajectory['actions'].shape[0])
    else:
        for trajectory in dataset:
            length.append(trajectory['high_observations'].shape[0])

    plt.figure(figsize=(25, 5))
    plt.hist(length, bins=max_length, edgecolor='black')
    plt.title('Trajectory Sample Distribution')
    plt.xlabel('Trajectory Sample')
    plt.ylabel('Count')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_distance_histogram(dataset: EpisodeDataset | HierarchicalDataset, save_path: str | None = None) -> None:
    distance = []

    if isinstance(dataset, EpisodeDataset):
        for trajectory in dataset:
            start = trajectory['observations'][0]
            end = trajectory['observations'][-1]
            distance.append(torch.norm(end - start, dim=-1).item())
    else:
        for trajectory in dataset:
            start = trajectory['high_goals'][0]
            end = trajectory['high_goals'][-1]
            distance.append(torch.norm(end - start, dim=-1).item())

    plt.figure(figsize=(25, 5))
    plt.hist(distance, bins=30, edgecolor='black')
    plt.title('Dataset Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_evaluate_distance_histogram(distance: List[List[float]], save_path: str | None = None) -> None:
    distance = [x for sublist in distance for x in sublist]

    plt.figure(figsize=(25, 5))
    plt.hist(distance, bins=30, edgecolor='black')
    plt.title('Evaluate Distance Distribution')
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_distance_roll_scatter(distance: List[List[float]], roll: List[List[int]], success: List[List[int]], subtitle: int | None = None, eval_count: int | None = None, save_path: str | None = None) -> None:
    plt.figure(figsize=(25, 5))

    for eval_idx, (d_list, r_list, s_list) in enumerate(zip(distance, roll, success)):
        for d, r, s in zip(d_list, r_list, s_list):
            plt.scatter(d, r, c='green' if s else 'red', alpha=1.0)
            if s:
                plt.text(d, r, f'E{eval_idx + 1}' if eval_count is None else f'E{eval_count}', fontsize=12)

    title = 'Distance vs Roll Step Scatter' if subtitle is None else f'Distance vs Roll Step Scatter - Evaluate Epoch_{subtitle}'
    plt.title(title)
    plt.xlabel('Distance')
    plt.ylabel('Roll Steps')
    plt.grid(True)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_distance_success_rate(distance: List[List[float]], success: List[List[float]], bins: int = 15, save_path: str | None = None) -> None:
    distance = [x for sublist in distance for x in sublist]
    success = [x for sublist in success for x in sublist]
    distance = np.asarray(distance)
    success = np.asarray(success)

    bin_edges = np.linspace(distance.min(), distance.max(), bins + 1)

    bin_centers = []
    success_rates = []

    for i in range(len(bin_edges) - 1):
        left = bin_edges[i]
        right = bin_edges[i + 1]

        mask = (distance >= left) & (distance < right)
        if np.sum(mask) == 0:
            continue

        rate = success[mask].mean()
        bin_centers.append((left + right) / 2)
        success_rates.append(rate)

    plt.figure(figsize=(25, 5))
    plt.plot(bin_centers, success_rates, marker='o')
    plt.title('Distance vs Success Rate')
    plt.xlabel('Distance')
    plt.ylabel('Success Rate')
    plt.grid(True)
    plt.ylim(0, 1)

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def main():
    dataset_id: str = 'D4RL/pointmaze/open-v2'
    wandb.login(key='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')

    minari_dataset: minari.MinariDataset = minari.load_dataset(dataset_id)
    env: gym.Env = EnvWrapper(minari_dataset.recover_environment(render_mode='rgb_array'))
    env.reset()

    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    chunk_size = None
    execute_mode = 'chunk'
    lr = 1e-5
    max_length = 32
    batch_size = 64
    hidden_dims = 64
    use_average_model = True
    use_baseline = False
    use_fixed_length = True
    file_no = 7

    is_hierarchical = chunk_size is not None
    print(f'is_hierarchical: {is_hierarchical}')

    planner_save_path = f'open_maze_planner_{file_no}.pt'
    l_save_path = f'log_l_{file_no}.pt'
    optimal_idx_save_path = f'log_optimal_idx_{file_no}.pt'
    tree_depth_save_path = f'log_tree_depth_{file_no}.pt'

    print(f'Device: {device}')
    print(f'Original Trajectory: {len(minari_dataset)}')

    if is_hierarchical:
        if use_fixed_length:
            hierarchical_dataset: HierarchicalDataset = HierarchicalDataset.create_fixed_length(minari_dataset, chunk_size=chunk_size, max_length=max_length)
        else:
            hierarchical_dataset: HierarchicalDataset = HierarchicalDataset.create(minari_dataset, chunk_size=chunk_size, max_length=max_length)

        if not use_baseline:
            hierarchical_data_loader: DataLoader = DataLoader(hierarchical_dataset, batch_sampler=GroupByLengthSampler.create(hierarchical_dataset, drop_last=True, base_memory=(24 * 1024 * 1024), max_batch_size=batch_size))
        else:
            hierarchical_data_loader: DataLoader = DataLoader(hierarchical_dataset, batch_sampler=GroupByLengthSampler.create(hierarchical_dataset, drop_last=True, base_memory=(24 * 1024 * 1024), max_batch_size=batch_size), collate_fn=hierarchical_random_future_goal_collate_fn)

        print(f'Clip Trajectory: {len(hierarchical_dataset)}')

        obs_dim = hierarchical_dataset[0]['high_observations'].shape[-1]
        action_dim = hierarchical_dataset[0]['low_actions'].shape[-1]
        goal_dim = hierarchical_dataset[0]['high_goals'].shape[-1]

        agent: PlanningAgent = PlanningAgent([(obs_dim + goal_dim + goal_dim + 1), hidden_dims, hidden_dims, 1], [(obs_dim + goal_dim), hidden_dims, hidden_dims, goal_dim], [(obs_dim + goal_dim), hidden_dims, hidden_dims, (action_dim * chunk_size)], chunk_size=chunk_size).to(device=device)
        log_L, log_optimal_idx, log_tree_depth, distance, success, roll = train(agent, env, hierarchical_data_loader, max_length, batch_size, hidden_dims, use_fixed_length, chunk_size=chunk_size, execute_mode=execute_mode, lr=lr, use_average_model=use_average_model, use_baseline=use_baseline, planner_save_path=planner_save_path, l_save_path=l_save_path, optimal_idx_save_path=optimal_idx_save_path, tree_depth_save_path=tree_depth_save_path, device=device)

        plot_trajectory_length_histogram(hierarchical_dataset, max_length // chunk_size, save_path='Trajectory Length Histogram')
        plot_distance_histogram(hierarchical_dataset, save_path='Trajectory Distance Histogram')

    else:
        if use_fixed_length:
            episode_dataset: EpisodeDataset = EpisodeDataset.create_fixed_length(minari_dataset, max_length=max_length)
        else:
            episode_dataset: EpisodeDataset = EpisodeDataset.create(minari_dataset, max_length=max_length)

        if not use_baseline:
            episode_data_loader: DataLoader = DataLoader(episode_dataset, batch_sampler=GroupByLengthSampler.create(episode_dataset, drop_last=True, base_memory=(24 * 1024 * 1024), max_batch_size=batch_size))
        else:
            episode_data_loader: DataLoader = DataLoader(episode_dataset, batch_sampler=GroupByLengthSampler.create(episode_dataset, drop_last=True, base_memory=(24 * 1024 * 1024), max_batch_size=batch_size), collate_fn=episode_random_future_goal_collate_fn)

        print(f'Clip Trajectory: {len(episode_dataset)}')

        obs_dim = episode_dataset[0]['observations'].shape[-1]
        action_dim = episode_dataset[0]['actions'].shape[-1]
        goal_dim = episode_dataset[0]['goals'].shape[-1]

        agent: PlanningAgent = PlanningAgent([(obs_dim + goal_dim + goal_dim + 1), hidden_dims, hidden_dims, 1], [(obs_dim + goal_dim), hidden_dims, hidden_dims, goal_dim], [(obs_dim + goal_dim), hidden_dims, hidden_dims, action_dim]).to(device=device)
        log_L, log_optimal_idx, log_tree_depth, distance, success, roll = train(agent, env, episode_data_loader, max_length, batch_size, hidden_dims, use_fixed_length, lr=lr, use_average_model=use_average_model, use_baseline=use_baseline, planner_save_path=planner_save_path, l_save_path=l_save_path, optimal_idx_save_path=optimal_idx_save_path, tree_depth_save_path=tree_depth_save_path, device=device)

        plot_trajectory_length_histogram(episode_dataset, max_length, save_path='Trajectory Length Histogram')
        plot_distance_histogram(episode_dataset, save_path='Trajectory Distance Histogram')

    plot_evaluate_distance_histogram(distance, save_path='Evaluate Distance Histogram')
    plot_distance_roll_scatter(distance, roll, success, save_path='Distance Roll Scatter')
    plot_distance_success_rate(distance, success, save_path='Distance Success Rate')


if __name__ == '__main__':
    main()
