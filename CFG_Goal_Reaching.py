import gymnasium as gym
import minari
import matplotlib.pyplot as plt
import numpy as np
import random
import tyro
import wandb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Bernoulli, Normal
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
from tqdm import trange, tqdm
from pathlib import Path


@dataclass
class CFGGoalReachingConfig:
    device: str = 'cpu'

    env_id: str = 'PointMaze_UMaze-v3'
    ref_min_score: float = 23.85
    ref_max_score: float = 161.86

    dataset_id: str = 'D4RL/pointmaze/umaze-v2'
    download_dataset: bool = True
    concat_goal: bool = False
    save_dataset_before_used: bool = False
    episode_data_path: str = 'episode_dataset.pt'

    base_memory: int = 28 * 1024 * 1024
    shuffle: bool = False
    drop_last: bool = False

    obs_dim: int = 4
    goal_dim: int = 2
    action_dim: int = 2
    hidden_dim: int = 32
    n_hidden: int = 2
    max_split_step: int = 50
    threshold: float = 0.45

    log_std_min: float = -5
    log_std_max: float = 2.0

    evaluate_epochs: int = 10
    use_optimal_path: bool = True

    lr: float = 1e-4
    weight_decay: float = 1e-5
    train_epochs: int = 10
    log_interval: int = 100
    plan_path: str = 'test_planner_full.pt'
    project_name: str = 'test_planner_training_full'
    run_name: str = 'test_planner_training_full_default_run'
    use_wandb: bool = True


class EnvWrapper(gym.ObservationWrapper):
    def __init__(
            self,
            env: gym.Env
        ) -> None:

        super(EnvWrapper, self).__init__(env)

    def observation(
            self,
            observation: Dict[str, np.ndarray]
        ) -> Dict[str, np.ndarray]:

        return {k: v.astype(np.float32) for k, v in observation.items()}


class EpisodeDataset(Dataset):
    def __init__(
            self,
            minari_dataset: minari.MinariDataset,
            config: CFGGoalReachingConfig
        ) -> None:

        self.device = config.device

        self.data = []

        for ep_id in trange(minari_dataset.total_episodes):
            ep = minari_dataset[ep_id]

            obs = ep.observations['observation']
            action = ep.actions
            reward = ep.rewards
            done = ep.truncations
            goal = ep.observations['achieved_goal']

            obs = torch.tensor(np.array(obs), dtype=torch.float32)
            action = torch.tensor(np.array(action), dtype=torch.float32)
            reward = torch.tensor(np.array([reward]), dtype=torch.float32)
            done = torch.tensor(np.array([done]), dtype=torch.float32)
            goal = torch.tensor(np.array(goal), dtype=torch.float32)

            if config.concat_goal:
                obs = torch.cat([obs, goal], dim=-1)

            self.data.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'done': done,
                'goal': goal
            })

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
            self,
            idx: int
        ) -> Dict[str, torch.Tensor]:

        ep = self.data[idx]

        return {
            'obs': ep['obs'].to(self.device),
            'action': ep['action'].to(self.device),
            'reward': ep['reward'].to(self.device),
            'done': ep['done'].to(self.device),
            'goal': ep['goal'].to(self.device)
        }


class EpisodeTensorDataset(Dataset):
    def __init__(
            self,
            config: CFGGoalReachingConfig
        ) -> None:

        self.data = torch.load(config.episode_data_path)
        self.concat_goal = config.concat_goal
        self.device = config.device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(
            self,
            idx: int
        ) -> Dict[str, torch.Tensor]:

        ep = self.data[idx]

        if self.concat_goal:
            ep['obs'] = torch.cat([ep['obs'], ep['goal']], dim=-1)

        return {
            'obs': ep['obs'].to(self.device),
            'action': ep['action'].to(self.device),
            'reward': ep['reward'].to(self.device),
            'done': ep['done'].to(self.device),
            'goal': ep['goal'].to(self.device)
        }


class GroupByLengthBatchSampler:
    def __init__(
            self,
            episode_dataset: EpisodeDataset | EpisodeTensorDataset,
            config: CFGGoalReachingConfig
        ) -> None:

        self.all_batch = []

        length_to_idx = defaultdict(list)

        for i in range(len(episode_dataset)):
            trajectory = episode_dataset[i]
            length = trajectory['action'].shape[0]
            length_to_idx[length].append(i)

        for length, idx in length_to_idx.items():
            batch_size = min(max((config.base_memory // (length ** 3 * 4)), 1), len(idx))
            if config.shuffle:
                random.shuffle(idx)

            for i in range(0, len(idx), batch_size):
                batch = idx[i:i + batch_size]
                if config.drop_last and len(batch) < batch_size:
                    continue

                if len(batch) > 0:
                    self.all_batch.append(batch)

        if config.shuffle:
            random.shuffle(self.all_batch)

    def __iter__(self):
        yield from self.all_batch

    def __len__(self):
        return len(self.all_batch)


class MLP(nn.Module):
    def __init__(
            self,
            dim: List[int],
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            dropout: float = None
        ) -> None:

        super(MLP, self).__init__()

        layer = []
        for i in range(len(dim) - 2):
            layer.append(nn.Linear(dim[i], dim[i + 1]))
            layer.append(activation_fn())

            if dropout is not None:
                layer.append(nn.Dropout(dropout))

        layer.append(nn.Linear(dim[-2], dim[-1]))

        if output_activation_fn is not None:
            layer.append(output_activation_fn())

        self.net = nn.Sequential(*layer)

    def forward(
            self,
            x: torch.Tensor
        ) -> torch.Tensor:

        return self.net(x)


class ShouldSplitNet(nn.Module):
    def __init__(
            self,
            config: CFGGoalReachingConfig,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            dropout: float = None
        ) -> None:

        super(ShouldSplitNet, self).__init__()

        self.net = MLP(
            dim=[(config.obs_dim + config.goal_dim), *([config.hidden_dim] * config.n_hidden), 1],
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )
        self.device = config.device

    def forward(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor
        ) -> Bernoulli:

        enhance_obs = torch.cat([obs, goal], dim=-1)
        logits = self.net(enhance_obs).squeeze(-1)

        return Bernoulli(logits=logits)

    def get_terminal_likelihood(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor,
            terminal: torch.Tensor
        ) -> torch.Tensor:

        T = self(obs, goal)

        return T.log_prob(terminal)

    def compute_terminal_likelihood(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor
        ) -> torch.Tensor:

        B, n = obs.shape[0], obs.shape[1] - 1

        s_i = obs[:, :n].unsqueeze(2).expand(B, n, n, -1)     # (B, n, n, obs_dim)
        g_jp1 = goal[:, 1:].unsqueeze(1).expand(B, n, n, -1)  # (B, n, n, goal_dim)
        terminal = torch.eye(n, dtype=torch.float32, device=self.device).unsqueeze(0)  # (B, n, n)

        T = self.get_terminal_likelihood(s_i, g_jp1, terminal)  # shape: (B, n, n)

        T_mask = torch.triu(torch.ones((n, n), dtype=torch.bool, device=self.device))
        T = T.masked_fill(~T_mask.unsqueeze(0), -float('inf'))

        return T  # (B, n, n)


class MidGoalPredictor(nn.Module):
    def __init__(
            self,
            config: CFGGoalReachingConfig,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            dropout: float = None
        ) -> None:

        super(MidGoalPredictor, self).__init__()

        self.net = MLP(
            dim=[(config.obs_dim + config.goal_dim), *([config.hidden_dim] * config.n_hidden), config.goal_dim],
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )
        self.log_std = nn.Parameter(torch.zeros(config.goal_dim, dtype=torch.float32))
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max
        self.device = config.device

    def forward(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor
        ) -> Normal:

        enhance_obs = torch.cat([obs, goal], dim=-1)
        mean = self.net(enhance_obs)
        std = torch.exp(self.log_std.clamp(min=self.log_std_min, max=self.log_std_max))

        return Normal(mean, std)

    def get_planed_likelihood(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor,
            mid_goal: torch.Tensor
        ) -> torch.Tensor:

        P = self(obs, goal)

        return P.log_prob(mid_goal).sum(dim=-1)

    def compute_planed_likelihood(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor
        ) -> torch.Tensor:

        B, n = obs.shape[0], obs.shape[1] - 1

        s_i = obs[:, :n-1].unsqueeze(2).unsqueeze(3).expand(B, n-1, n-1, n-1, -1)            # shape: (B, n-1, n-1, n-1, obs_dim)
        g_jp2 = goal[:, 2:n+1].unsqueeze(1).unsqueeze(3).expand(B, n-1, n-1, n-1, -1)        # shape: (B, n-1, n-1, n-1, goal_dim)
        s_kp1 = obs[:, 1:n].unsqueeze(1).unsqueeze(1)[..., :2].expand(B, n-1, n-1, n-1, -1)  # shape: (B, n-1, n-1, n-1, goal_dim)

        P = self.get_planed_likelihood(s_i, g_jp2, s_kp1)  # shape: (B, n-1, n-1, n-1)

        idx = torch.arange((n - 1), dtype=torch.int32, device=self.device)
        k, i, j = torch.meshgrid(idx, idx, idx, indexing='ij')
        P_mask = (i <= k) & (k <= j)
        P = P.masked_fill(~P_mask.unsqueeze(0), -float('inf'))

        return P


class PolicyNet(nn.Module):
    def __init__(
            self,
            config: CFGGoalReachingConfig,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = nn.Tanh,
            dropout: float = None
        ) -> None:

        super(PolicyNet, self).__init__()

        self.net = MLP(
            dim=[(config.obs_dim + config.goal_dim), *([config.hidden_dim] * config.n_hidden), config.action_dim],
            activation_fn=activation_fn,
            output_activation_fn=output_activation_fn,
            dropout=dropout
        )
        self.log_std = nn.Parameter(torch.zeros(config.action_dim, dtype=torch.float32))
        self.log_std_min = config.log_std_min
        self.log_std_max = config.log_std_max

    def forward(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor
        ) -> Normal:

        enhance_obs = torch.cat([obs, goal], dim=-1)
        mean = self.net(enhance_obs)
        std = torch.exp(self.log_std.clamp(min=self.log_std_min, max=self.log_std_max))

        return Normal(mean, std)

    def get_actor_likelihood(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor,
            action: torch.Tensor
        ) -> torch.Tensor:

        A = self(obs, goal)

        return A.log_prob(action).sum(dim=-1)

    def compute_actor_likelihood(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor,
            action: torch.Tensor
        ) -> torch.Tensor:

        B, n = obs.shape[0], obs.shape[1] - 1

        s_i = obs[:, :n]     # shape: (B, n, obs_dim)
        g_ip1 = goal[:, 1:]  # shape: (B, n, goal_dim)
        a_i = action[:, :n]  # shape: (B, n, action_dim)

        A = self.get_actor_likelihood(s_i, g_ip1, a_i)  # shape: (B, n)

        return A


class PlanningAgent(nn.Module):
    def __init__(
            self,
            config: CFGGoalReachingConfig
        ) -> None:

        super(PlanningAgent, self).__init__()

        self.max_split_step = config.max_split_step

        self.terminal = ShouldSplitNet(config)
        self.mid_goal = MidGoalPredictor(config)
        self.actor = PolicyNet(config)

    def act_agent(
            self,
            obs: torch.Tensor,
            goal: torch.Tensor
        ) -> Tuple[torch.Tensor, List[torch.Tensor]]:

        split_step = 1
        mid_goal_list = [goal]

        while self.terminal(obs, goal).mean < 0.5:
            if split_step >= self.max_split_step:
                break

            split_step += 1
            goal = self.mid_goal(obs, goal).mean
            mid_goal_list.append(goal)

        mid_goal_list.reverse()

        return self.actor(obs, goal).mean, mid_goal_list


def save_episode_dataset(
        minari_dataset: minari.MinariDataset,
        config: CFGGoalReachingConfig,
    ) -> None:

    data = []

    for ep_id in trange(minari_dataset.total_episodes):
        ep = minari_dataset[ep_id]

        obs = ep.observations['observation']
        action = ep.actions
        reward = ep.rewards
        done = ep.truncations
        goal = ep.observations['achieved_goal']

        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        action = torch.tensor(np.array(action), dtype=torch.float32)
        reward = torch.tensor(np.array([reward]), dtype=torch.float32)
        done = torch.tensor(np.array([done]), dtype=torch.float32)
        goal = torch.tensor(np.array(goal), dtype=torch.float32)

        if config.concat_goal:
            obs = torch.cat([obs, goal], dim=-1)

        data.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done,
            'goal': goal
        })

    torch.save(data, config.episode_data_path)
    print(f'Saved data to {config.episode_data_path}, total episodes: {len(data)}')


def compute_total_likelihood(
        A: torch.Tensor,   # shape: (B, n)
        T: torch.Tensor,   # shape: (B, n, n)
        P: torch.Tensor,   # shape: (B, n-1, n-1, n-1)
        config: CFGGoalReachingConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    B, n = A.shape
    L = torch.full((B, n, n), -float('inf'), dtype=torch.float32, device=config.device)
    optimal_idx = torch.full((B, n, n), -1, dtype=torch.int64, device=config.device) if config.use_optimal_path else None

    diag = torch.arange(n, dtype=torch.int32, device=config.device)
    L[:, diag, diag] = A + T[:, diag, diag]

    for d in range(1, n):
        i = torch.arange(n - d, dtype=torch.int32, device=config.device)
        j = i + d
        k = i[:, None] + torch.arange(d, dtype=torch.int32, device=config.device)[None, :]  # shape: (n-d, d)

        # Shape: (B, n-d, d)
        p_mid = P[:, k, i[:, None], j[:, None] - 1]           # shape: (B, n-d, d)
        l1 = L[:, i[:, None], k]                              # shape: (B, n-d, d)
        l2 = L[:, k + 1, j[:, None]]                          # shape: (B, n-d, d)

        log_terms = p_mid + l1 + l2                           # shape: (B, n-d, d)

        if config.use_optimal_path:
            optimal, argmax_k = torch.max(log_terms, dim=2)
            L[:, i, j] = T[:, i, j] + optimal

            abs_k = i.unsqueeze(0) + argmax_k
            optimal_idx[:, i, j] = abs_k
        else:
            L[:, i, j] = T[:, i, j] + torch.logsumexp(log_terms, dim=2)

    return L, optimal_idx                                     # shape: (B, n, n)


def recover_optimal_trajectory(
        goal: torch.Tensor,
        optimal_idx: torch.Tensor
    ) -> List[torch.Tensor]:

    B, n = goal.shape[0], goal.shape[1] - 1
    optimal_path = []

    def extract_optimal_path(
            idx: torch.Tensor,
            i: int,
            j: int
        ) -> List[int]:

        k = idx[i, j].item()
        if k == -1 or k == i or k == j:
            return []
        return extract_optimal_path(idx, i, k) + [k + 1] + extract_optimal_path(idx, k + 1, j)

    for b in range(B):
        idx = optimal_idx[b]
        mid_goal_idx = extract_optimal_path(idx, 0, n - 1)
        mid_goal = [goal[b, idx] for idx in mid_goal_idx]
        optimal_trajectory = torch.stack([goal[b, 0]] + mid_goal + [goal[b, -1]], dim=0)
        optimal_path.append(optimal_trajectory)

    return optimal_path


def compute_mid_goal_achieved(
        trajectory_info: List[Dict[str, np.ndarray]],
        mid_goal_list_info: List[List[torch.Tensor]],
        config: CFGGoalReachingConfig
    ) -> Tuple[List[int], List[int]]:

    n_level_goal_achieved = [0] * config.max_split_step
    n_level_goal_counts = [0] * config.max_split_step

    for i, mid_goal_list in enumerate(mid_goal_list_info[:-1]):
        mid_goal = [mg.detach().cpu().numpy() for mg in mid_goal_list]
        trajectory_achieved_goal = [step['achieved_goal'] for step in trajectory_info[i + 1:]]

        for j in range(len(mid_goal_list)):
            n_level_goal_counts[j] += 1

        if float(np.linalg.norm(trajectory_achieved_goal[0] - mid_goal[0])) < config.threshold:
            n_level_goal_achieved[0] += 1

            target_idx = 1
            for achieved_goal in trajectory_achieved_goal[1:]:
                if target_idx >= len(mid_goal):
                    break

                if float(np.linalg.norm(achieved_goal - mid_goal[target_idx])) < config.threshold:
                    n_level_goal_achieved[target_idx] += 1
                    target_idx += 1

    return n_level_goal_achieved, n_level_goal_counts


def evaluate(
        agent: PlanningAgent,
        env: gym.Env,
        config: CFGGoalReachingConfig
    ) -> Tuple[np.floating, np.floating, np.floating, np.floating, List[np.floating], List[np.floating]]:

    agent.eval()

    step_info = []
    reward_info = []
    norm_total_reward_info = []
    success_rate_info = []

    with tqdm(iterable=range(config.evaluate_epochs), leave=False, dynamic_ncols=True) as pbar:
        for epoch in pbar:
            trajectory_info = []
            mid_goal_list_info = []

            obs_dict, info = env.reset()
            obs = torch.tensor(obs_dict['observation'], device=config.device)
            goal = torch.tensor(obs_dict['desired_goal'], device=config.device)
            done = info['success']
            total_reward = 0.0
            roll_step = 0

            while not done:
                with torch.no_grad():
                    action, mid_goal_list = agent.act_agent(obs, goal)
                    obs_dict, reward, terminated, truncated, info = env.step(action.cpu().numpy())
                    obs = torch.tensor(obs_dict['observation'], device=config.device)
                    goal = torch.tensor(obs_dict['desired_goal'], device=config.device)
                    done = info['success'] or truncated
                    total_reward += reward
                    roll_step += 1

                    trajectory_info.append(obs_dict)
                    mid_goal_list_info.append(mid_goal_list)

            norm_total_reward = (total_reward - config.ref_min_score) / (config.ref_max_score - config.ref_min_score)
            n_level_goal_achieved, n_level_goal_counts = compute_mid_goal_achieved(trajectory_info, mid_goal_list_info, config)
            success_rate = [float(g / c) if c != 0 else 0 for g, c in zip(n_level_goal_achieved, n_level_goal_counts)]
            success_rate_info.append(success_rate)

            if config.use_wandb:
                wandb.log({
                    **{
                        'Test/Roll Step': roll_step,
                        'Test/Total Reward': total_reward,
                        'Test/Norm Total Reward': norm_total_reward
                    },
                    **{f'Test/Success Rate/Depth_{i + 1}': v for i, v in enumerate(success_rate)}
                })

            pbar.set_description(f'Testing Epoch: {epoch + 1} / {config.evaluate_epochs}, Mean = {np.nanmean(reward_info) if reward_info else 0:.9f}, Std = {np.nanstd(reward_info) if reward_info else 0:.9f}')

            step_info.append(roll_step)
            reward_info.append(total_reward)
            norm_total_reward_info.append(norm_total_reward)

    return np.nanmean(reward_info), np.nanstd(reward_info), np.nanmean(norm_total_reward_info), np.nanstd(norm_total_reward_info), np.nanmean(success_rate_info, axis=0), np.nanstd(success_rate_info, axis=0)


def train(
        agent: PlanningAgent,
        env: gym.Env,
        dataloader: DataLoader,
        config: CFGGoalReachingConfig,
        optimizer: torch.optim.Optimizer = None
    ) -> None:

    if config.use_wandb:
        wandb.init(
            mode='online',
            project=config.project_name,
            name=config.run_name,
            config={
                'lr': config.lr,
                'weight_decay': config.weight_decay,
                'epochs': config.train_epochs,
                'log_interval': config.log_interval
            }
        )

    if optimizer is None:
        optimizer = torch.optim.Adam(
            agent.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

    agent.train()

    step = 0
    total_loss = 0.0
    total_trajectory = 0

    for epoch in range(config.train_epochs):
        with tqdm(iterable=dataloader, leave=False, dynamic_ncols=True) as pbar:
            for batch in pbar:

                obs = batch['obs']          # shape: (B, n+1, obs_dim)
                action = batch['action']    # shape: (B, n, action_dim)
                goal = batch['goal']        # shape: (B, n+1, goal_dim)
                B, n = obs.shape[0], obs.shape[1] - 1

                A = agent.actor.compute_actor_likelihood(obs, goal, action)         # shape: (B, n)
                T = agent.terminal.compute_terminal_likelihood(obs, goal)           # shape: (B, n, n)
                P = agent.mid_goal.compute_planed_likelihood(obs, goal)             # shape: (B, n-1, n-1, n-1)

                actor_loss = -A.mean()

                T_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=config.device)).unsqueeze(0)  # shape: (1, n, n)
                T_ref = T[T_mask.expand(B, -1, -1)]
                terminal_loss = -T_ref.mean()

                L, optimal_idx = compute_total_likelihood(A, T, P, config)          # shape: (B, n, n)

                n = L.shape[1]
                loss = -L[:, 0, n-1].mean() / n                                     # L[0, n-1] maximize likelihood

                total_loss += loss.item() * B
                total_trajectory += B
                average_loss = total_loss / total_trajectory

                optimizer.zero_grad()
                loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(parameters=agent.actor.parameters(), max_norm=1, norm_type=2)
                terminal_grad = nn.utils.clip_grad_norm_(parameters=agent.terminal.parameters(), max_norm=1, norm_type=2)
                mid_goal_grad = nn.utils.clip_grad_norm_(parameters=agent.mid_goal.parameters(), max_norm=1, norm_type=2)
                optimizer.step()

                pbar.set_description(f'Training Epoch: {epoch + 1} / {config.train_epochs}, Step {step}: Current Loss = {loss.item():.9f}, Average loss = {average_loss:.9f}')
                if config.use_wandb and step % config.log_interval == 0:
                    wandb.log({
                        'Train/Current Loss': loss.item(),
                        'Train/Average Loss': average_loss,
                        'Train/Actor Loss': actor_loss.item(),
                        'Train/Terminal Loss': terminal_loss.item(),
                        'Train/Actor Grad': actor_grad,
                        'Train/Terminal Grad': terminal_grad,
                        'Train/Mid Goal Grad': mid_goal_grad
                    }, step=step)
                step += 1

            mean, std, norm_mean, norm_std, success_mean, success_std = evaluate(agent, env, config)
            if config.use_wandb:
                wandb.log({
                    **{
                        'Test/Mean': mean,
                        'Test/Std': std,
                        'Test/Norm Mean': norm_mean,
                        'Test/Norm Std': norm_std
                    },
                    **{f'Test/Success Mean/Depth_{i + 1}': v for i, v in enumerate(success_mean)},
                    **{f'Test/Success Std/Depth_{i + 1}': v for i, v in enumerate(success_std)}
                }, step=step)

    wandb.finish()

    agent.eval()
    torch.save(agent.state_dict(), config.plan_path)
    print(f'Saved planner model to {config.plan_path}')


def plot_trajectory_length_histogram(episode_dataset: EpisodeDataset | EpisodeTensorDataset):
    length = []

    for trajectory in episode_dataset:
        length.append(trajectory['action'].shape[0])

    plt.figure(figsize=(20, 5))
    plt.hist(length, bins=200, edgecolor='black')
    plt.title('Trajectory Length Distribution')
    plt.xlabel('Trajectory Length')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()


def main():
    config: CFGGoalReachingConfig = tyro.cli(CFGGoalReachingConfig)
    config.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    dataset: minari.MinariDataset = minari.load_dataset(config.dataset_id, download=config.download_dataset)
    dataset.recover_environment()
    env: gym.Env = EnvWrapper(gym.make(config.env_id, render_mode='rgb_array'))
    env.reset()

    if config.save_dataset_before_used:
        save_episode_dataset(dataset, config)
        episode_dataset: EpisodeTensorDataset = EpisodeTensorDataset(config)
    elif Path(config.episode_data_path).exists():
        episode_dataset: EpisodeTensorDataset = EpisodeTensorDataset(config)
    else:
        episode_dataset: EpisodeDataset = EpisodeDataset(dataset, config)

    batch_sampler: GroupByLengthBatchSampler = GroupByLengthBatchSampler(episode_dataset, config)
    episode_dataloader: DataLoader = DataLoader(episode_dataset, batch_sampler=batch_sampler)

    config.obs_dim = episode_dataset[0]['obs'].shape[-1]
    config.action_dim = episode_dataset[0]['action'].shape[-1]
    config.goal_dim = episode_dataset[0]['goal'].shape[-1]

    agent: PlanningAgent = PlanningAgent(config).to(device=config.device)

    train(agent, env, episode_dataloader, config)


if __name__ == '__main__':
    main()
