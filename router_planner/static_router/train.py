"""
Training script for the Static Router using PPO.

This script orchestrates the training loop, including:
- Batch episode collection with thread-based parallelism
- PPO policy/value updates with configurable schedules
- Periodic evaluation on a held-out test set
- Weights & Biases logging for experiment tracking

Usage:
    python -m router_planner.static_router.train --query_file <path> ...

The wandb API key should be set via the WANDB_API_KEY environment variable.
"""

import os
import argparse
import time
import threading
import copy
import gc
import random
import warnings

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.dummy import Pool as ThreadPool

from .network import RouteAgent
from .route_env import Env_route

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
#                        Episode execution functions                           #
# --------------------------------------------------------------------------- #

def operate_global(request):
    """
    Run a single training episode. Updates the historical memory table
    after completion.
    """
    agent = request["agent"]
    env = request["env"]
    state = request["state"]
    agent_llm_emb = request["agent_llm_emb"]
    query_id_embed = request["query_id_embed"]
    LLM_role_id_embed = request["LLM_role_id_embed"]
    local_memory_table = request["local_memory_table"]
    his_index = request["his_index"]
    his_memory_table = request["his_memory_table"]
    max_timesteps = request["max_timesteps"]
    memory_table_obj = request["memory_table_obj"]

    state = torch.stack([state]).to(device)
    his_tables = {
        "memory_table": his_memory_table,
        "query_table": query_id_embed,
        "LLM_table": LLM_role_id_embed,
    }
    local_tables = {
        "memory_table": local_memory_table,
        "query_table": query_id_embed,
        "LLM_table": LLM_role_id_embed,
    }
    episode_reward = 0
    episode_entropies = []
    transitions = []

    for t in range(max_timesteps):
        # Select action (training mode: sample from distribution)
        action, log_prob, value, entropy = agent.select_action(
            state, local_tables, his_tables, his_index, greedy=False
        )
        episode_entropies.append(entropy)

        # Environment step
        next_state, done, cost, reward, next_agent_llm_emb, local_memory_table, his_index = (
            env.next_step(action=action)
        )

        next_state_tensor = torch.stack([next_state]).to(device)
        local_tables = {
            "memory_table": local_memory_table,
            "query_table": query_id_embed,
            "LLM_table": LLM_role_id_embed,
        }

        transitions.append((
            state, action, reward, log_prob, value, done,
            local_memory_table, his_index,
        ))
        episode_reward += reward

        state = next_state_tensor
        agent_llm_emb = next_agent_llm_emb

        if done:
            break

    episode_length = t + 1

    # Update global historical memory table with this episode's records
    if memory_table_obj is not None:
        update_record = {}
        for key in local_memory_table:
            if key in ["query.id", "LLM_role.id", "score", "cost", "n_i", "n_o"]:
                update_record[key] = local_memory_table[key]
        if update_record and any(
            len(v) > 0 if isinstance(v, list) else v is not None
            for v in update_record.values()
        ):
            memory_table_obj.update(update_record)

    return {
        "reward": episode_reward,
        "entropy": episode_entropies,
        "transitions": transitions,
        "episode_length": episode_length,
    }


def test_agent_on_dataset_subprocess(request):
    """Run a single test episode (no memory table update)."""
    agent = request["agent"]
    test_env = request["test_env"]
    state = request["state"]
    agent_llm_emb = request["agent_llm_emb"]
    query_id_embed = request["query_id_embed"]
    LLM_role_id_embed = request["LLM_role_id_embed"]
    local_memory_table = request["local_memory_table"]
    his_memory_table = request["his_memory_table"]
    his_index = request["his_index"]
    max_timesteps = request["max_timesteps"]

    with torch.no_grad():
        state = torch.stack([state]).to(device)
        his_tables = {
            "memory_table": his_memory_table,
            "query_table": query_id_embed,
            "LLM_table": LLM_role_id_embed,
        }
        local_tables = {
            "memory_table": local_memory_table,
            "query_table": query_id_embed,
            "LLM_table": LLM_role_id_embed,
        }
        episode_reward = 0
        episode_cost = 0

        for t in range(max_timesteps):
            # Greedy action selection for evaluation
            action, _, _, _ = agent.select_action(
                state, local_tables, his_tables, his_index, greedy=True
            )

            next_state, done, cost, reward, next_agent_llm_emb, local_memory_table, his_index = (
                test_env.next_step(action=action)
            )

            next_state_tensor = torch.stack([next_state]).to(device)
            local_tables = {
                "memory_table": local_memory_table,
                "query_table": query_id_embed,
                "LLM_table": LLM_role_id_embed,
            }
            episode_reward += reward
            episode_cost += cost
            state = next_state_tensor

            if done:
                break

    return episode_reward, t + 1, episode_cost


def test_agent_on_dataset(agent, test_query_file, llm_file, agent_file,
                          width=3, depth=2, max_timesteps=500, batch_size=2,
                          his_memory=None):
    """Evaluate the agent on the entire test dataset."""
    test_start_time = time.time()
    his_memory_table = his_memory.get_table()

    test_data = pd.read_csv(test_query_file)
    num_test_samples = len(test_data)
    print(f"Testing on {num_test_samples} samples from test dataset...")

    test_env = Env_route(
        query_file=test_query_file,
        test_query_file=test_query_file,
        llm_file=llm_file,
        agent_file=agent_file,
        width=width,
        depth=depth,
        device=device,
    )
    test_env.enable_test_mode()

    test_rewards = []
    test_episode_lengths = []
    test_costs = []

    if hasattr(agent, "eval"):
        agent.eval()

    # Process test samples in batches
    for ii in tqdm(range(0, num_test_samples, batch_size), desc="Testing Batches"):
        test_envs = []
        states = []
        initial_embs = []
        local_mems = []
        his_indexs = []

        for jj in range(ii, min(ii + batch_size, num_test_samples)):
            state, agent_llm_emb, query_id_embed, LLM_role_id_embed, local_memory_table, his_index = (
                test_env.reset()
            )
            test_envs.append(copy.copy(test_env))
            states.append(state)
            initial_embs.append(agent_llm_emb)
            local_mems.append(local_memory_table)
            his_indexs.append(his_index)

        # Share the agent across test threads (no weight updates)
        agents = [agent] * len(test_envs)
        for a in agents:
            if hasattr(a, "eval"):
                a.eval()

        requests = [{
            "agent": agents[i],
            "test_env": test_envs[i],
            "state": states[i],
            "agent_llm_emb": initial_embs[i],
            "query_id_embed": query_id_embed,
            "LLM_role_id_embed": LLM_role_id_embed,
            "local_memory_table": local_mems[i],
            "his_memory_table": his_memory_table,
            "his_index": his_indexs[i],
            "max_timesteps": max_timesteps,
        } for i in range(len(test_envs))]

        results = []
        with torch.no_grad():
            with ThreadPool(batch_size) as executor:
                for r in tqdm(
                    executor.imap_unordered(test_agent_on_dataset_subprocess, requests),
                    total=len(requests), desc="Processing", ncols=100,
                ):
                    results.append(r)

        for res in results:
            test_rewards.append(res[0])
            test_episode_lengths.append(res[1])
            test_costs.append(res[2])

        # Free batch-specific objects
        del test_envs, states, initial_embs, agents, requests, results
        gc.collect()
        torch.cuda.empty_cache()

    test_env.disable_test_mode()

    if hasattr(agent, "train"):
        agent.train()

    test_duration = time.time() - test_start_time

    return {
        "test_avg_reward": np.mean(test_rewards),
        "test_std_reward": np.std(test_rewards),
        "test_max_reward": np.max(test_rewards),
        "test_min_reward": np.min(test_rewards),
        "test_avg_episode_length": np.mean(test_episode_lengths),
        "test_avg_cost": np.mean(test_costs),
        "test_total_cost": np.sum(test_costs),
        "test_samples": num_test_samples,
        "test_duration": test_duration,
    }


# --------------------------------------------------------------------------- #
#                          Historical memory table                             #
# --------------------------------------------------------------------------- #

class MemoryTable:
    """Thread-safe historical memory table that accumulates episode records."""

    def __init__(self):
        self.his_memory_table = {
            "query.id": [],
            "LLM_role.id": [],
            "score": [],
            "cost": [],
            "n_i": [],
            "n_o": [],
        }
        self.lock = threading.Lock()

    def update(self, record: dict):
        """
        Add records to the memory table.

        Args:
            record: Either a single record (scalar values) or batch record
                    (list values). All lists must have the same length.
        """
        is_batch = any(isinstance(v, list) for v in record.values())

        if is_batch:
            lengths = [len(v) for v in record.values() if isinstance(v, list)]
            if len(set(lengths)) > 1:
                raise ValueError("All list values must have the same length")
            n = lengths[0]
            with self.lock:
                for key in self.his_memory_table:
                    values = record.get(key, [None] * n)
                    if not isinstance(values, list):
                        values = [values] * n
                    self.his_memory_table[key].extend(values)
        else:
            with self.lock:
                for key in self.his_memory_table:
                    self.his_memory_table[key].append(record.get(key, None))

    def get_table(self, align=True, freeze=False):
        """
        Return a snapshot of the memory table.

        Args:
            align: If True, truncate all columns to the minimum length.
            freeze: If True, convert lists to tuples (immutable).
        """
        with self.lock:
            snap = {k: list(v) for k, v in self.his_memory_table.items()}

        if align:
            keys = ["query.id", "LLM_role.id", "score", "cost", "n_i", "n_o"]
            n = min(len(snap[k]) for k in keys)
            for k in keys:
                snap[k] = snap[k][:n]

        if freeze:
            for k in snap:
                snap[k] = tuple(snap[k])

        return snap

    def __repr__(self):
        return str(self.his_memory_table)


# --------------------------------------------------------------------------- #
#                            Main training loop                                #
# --------------------------------------------------------------------------- #

def train_ppo(query_file, llm_file, agent_file, test_query_file=None,
              width=3, depth=2, embed_dim=768 * 2, max_episodes=1000,
              max_timesteps=500, update_timestep=2000,
              wandb_project="ppo_routing", experiment_name=None,
              gpu_id=None, test_every_n_episodes=10, batch_size=1,
              test_batch_size=1, test_before_train=True,
              gnn_backbone="dblayer"):
    """Train a PPO agent with embedding-based state representation."""

    if experiment_name is None:
        experiment_name = f"ppo_routing_embed_{embed_dim}"

    config = {
        "embed_dim": embed_dim,
        "max_episodes": max_episodes,
        "max_timesteps": max_timesteps,
        "width": width,
        "buffer_size": 50,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "eps_clip": 0.2,
        "k_epochs": 4,
        "entropy_coef": 0.00,
        "hidden_dim": 16,
        "candidate_embed_dim": 768 * 2,
        "device": str(device),
        "test_every_n_episodes": test_every_n_episodes,
        "test_query_file": test_query_file,
    }

    if test_query_file:
        test_data = pd.read_csv(test_query_file)
        config["test_dataset_size"] = len(test_data)

    wandb.init(project=wandb_project, name=experiment_name, config=config)

    # Create agent and environment
    agent = RouteAgent(
        embed_dim=768, candidate_embed_dim=768 * 2, hidden_dim=32,
        buffer_size=100, device=device, gnn_backbone=gnn_backbone,
    )
    env = Env_route(
        query_file=query_file, test_query_file=test_query_file,
        llm_file=llm_file, agent_file=agent_file,
        width=width, depth=depth, device=device,
    )
    his_memory_table = MemoryTable()

    # Global embeddings (populated on first reset)
    global_query_id_embed = None
    global_LLM_role_id_embed = None

    # Optional: test before training starts
    if test_before_train == 1 and test_query_file:
        print("=" * 60)
        print("INITIAL TESTING BEFORE TRAINING STARTS")
        print("=" * 60)
        with torch.no_grad():
            initial_test_stats = test_agent_on_dataset(
                agent=agent, test_query_file=test_query_file,
                llm_file=llm_file, agent_file=agent_file,
                width=width, depth=depth, max_timesteps=max_timesteps,
                batch_size=test_batch_size, his_memory=his_memory_table,
            )
        print(f"Initial Test Results (Before Training):")
        print(f"  Tested on {initial_test_stats['test_samples']} samples")
        print(f"  Average Reward: {initial_test_stats['test_avg_reward']:.2f} "
              f"+/- {initial_test_stats['test_std_reward']:.2f}")
        print(f"  Max Reward: {initial_test_stats['test_max_reward']:.2f}")
        print(f"  Min Reward: {initial_test_stats['test_min_reward']:.2f}")
        print(f"  Average Episode Length: {initial_test_stats['test_avg_episode_length']:.1f}")
        print(f"  Average Cost per Sample: {initial_test_stats['test_avg_cost']:.4f}")
        print(f"  Total Test Cost: {initial_test_stats['test_total_cost']:.4f}")
        print(f"  Test Duration: {initial_test_stats['test_duration']:.2f}s")

        wandb.log({
            "episode": -1,
            "initial_test_avg_reward": initial_test_stats["test_avg_reward"],
            "initial_test_std_reward": initial_test_stats["test_std_reward"],
            "initial_test_max_reward": initial_test_stats["test_max_reward"],
            "initial_test_min_reward": initial_test_stats["test_min_reward"],
            "initial_test_avg_episode_length": initial_test_stats["test_avg_episode_length"],
            "initial_test_avg_cost": initial_test_stats["test_avg_cost"],
            "initial_test_total_cost": initial_test_stats["test_total_cost"],
            "initial_test_samples": initial_test_stats["test_samples"],
        })
        print("=" * 60)

    # Training statistics
    episode_rewards = []
    running_reward = 0
    first_training_done = False  # Whether the first PPO update has occurred

    print(f"Starting PPO training with embedding dimension: {embed_dim}")
    print(f"Buffer size: 50, Training schedule: first train when buffer full, then every 3 episodes")
    print(f"Device: {device}")
    if test_query_file:
        test_data = pd.read_csv(test_query_file)
        print(f"Testing every {test_every_n_episodes} episodes on {len(test_data)} samples")
    print("-" * 50)

    def batch_operate(initial_agent, initial_env, batch_size=1, his_memory_table=None):
        """Collect a batch of episodes in parallel."""
        nonlocal global_query_id_embed, global_LLM_role_id_embed

        envs = []
        reset_results = []
        for _ in range(batch_size):
            reset_result = initial_env.reset()
            envs.append(copy.copy(initial_env))
            reset_results.append(reset_result)

        states = [r[0] for r in reset_results]
        initial_embs = [r[1] for r in reset_results]
        query_id_embeds = [r[2] for r in reset_results]
        LLM_role_id_embeds = [r[3] for r in reset_results]
        local_memory_tables = [r[4] for r in reset_results]
        his_indexs = [r[5] for r in reset_results]

        # Store global embeddings on first run
        if global_query_id_embed is None:
            global_query_id_embed = query_id_embeds[0]
            global_LLM_role_id_embed = LLM_role_id_embeds[0]

        # Create independent agent copies to avoid shared state across threads
        agents = [copy.copy(initial_agent) for _ in range(batch_size)]

        requests = [{
            "agent": agents[i],
            "env": envs[i],
            "state": states[i],
            "agent_llm_emb": initial_embs[i],
            "query_id_embed": query_id_embeds[i],
            "LLM_role_id_embed": LLM_role_id_embeds[i],
            "local_memory_table": local_memory_tables[i],
            "his_index": his_indexs[i],
            "his_memory_table": his_memory_table.get_table(),
            "max_timesteps": max_timesteps,
            "memory_table_obj": his_memory_table,
        } for i in range(batch_size)]

        results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(operate_global, req) for req in requests]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                results.append(future.result())
        del futures
        torch.cuda.empty_cache()

        avg_episode_entropy = (
            sum(sum(r["entropy"]) for r in results)
            / sum(len(r["entropy"]) for r in results)
        )
        reward = sum(r["reward"] for r in results) / batch_size
        avg_episode_length = sum(r["episode_length"] for r in results) / batch_size

        # Store all transitions into the main agent's buffer
        for result in results:
            for transition in result["transitions"]:
                initial_agent.store_transition(*transition)

        return avg_episode_entropy, reward, avg_episode_length

    # Main training loop
    for episode in range(max_episodes):
        episode_start_time = time.time()

        # Collect batch of episodes
        avg_episode_entropy, episode_reward, avg_episode_length = batch_operate(
            agent, env, batch_size, his_memory_table
        )

        # Determine whether to run a PPO update
        should_train = False
        training_reason = ""

        if agent.is_buffer_full() and not first_training_done:
            should_train = True
            first_training_done = True
            training_reason = "Buffer is full (first training)"
        elif first_training_done and (episode % 3 == 0 or batch_size > 1):
            should_train = True
            training_reason = "Scheduled training"

        if should_train:
            policy_loss, value_loss, training_entropy = agent.update(
                global_query_id_embed, global_LLM_role_id_embed,
                his_memory_table, 32,
            )
            print(f"Episode {episode}: {training_reason}")
        else:
            policy_loss, value_loss, training_entropy = 0, 0, 0
            if not first_training_done and episode < 50:
                print(f"Episode {episode}: Buffer filling ({len(agent.buffer)}/{agent.buffer_size})")

        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        episode_rewards.append(episode_reward)

        episode_duration = time.time() - episode_start_time

        # Log metrics
        log_dict = {
            "episode": episode,
            "episode_reward": episode_reward,
            "running_reward": running_reward,
            "episode_entropy": avg_episode_entropy,
            "buffer_size": len(agent.buffer),
            "episode_length": avg_episode_length,
            "episode_duration": episode_duration,
            "training_occurred": should_train,
            "memory_table_size": len(his_memory_table.get_table()["query.id"]),
        }

        if should_train:
            log_dict.update({
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "training_entropy": training_entropy,
            })

        # Periodic evaluation
        if test_query_file and (episode + 1) % test_every_n_episodes == 0:
            print(f"Testing at episode {episode + 1}...")
            with torch.no_grad():
                test_stats = test_agent_on_dataset(
                    agent=agent, test_query_file=test_query_file,
                    llm_file=llm_file, agent_file=agent_file,
                    width=width, depth=depth, max_timesteps=max_timesteps,
                    batch_size=test_batch_size, his_memory=his_memory_table,
                )
            log_dict.update(test_stats)

            print(f"Test Results at Episode {episode + 1}:")
            print(f"  Tested on {test_stats['test_samples']} samples")
            print(f"  Average Reward: {test_stats['test_avg_reward']:.2f} "
                  f"+/- {test_stats['test_std_reward']:.2f}")
            print(f"  Max Reward: {test_stats['test_max_reward']:.2f}")
            print(f"  Min Reward: {test_stats['test_min_reward']:.2f}")
            print(f"  Average Episode Length: {test_stats['test_avg_episode_length']:.1f}")
            print(f"  Average Cost per Sample: {test_stats['test_avg_cost']:.4f}")
            print(f"  Total Test Cost: {test_stats['test_total_cost']:.4f}")
            print(f"  Test Duration: {test_stats['test_duration']:.2f}s")

        wandb.log(log_dict)

        # Progress report
        if episode % 100 == 0:
            status = "Training started" if first_training_done else "Filling buffer"
            print(f"Episode {episode}, Reward: {episode_reward:.2f}, "
                  f"Running Reward: {running_reward:.2f}, "
                  f"Duration: {episode_duration:.2f}s, Status: {status}")
            print(f"  Episode Entropy: {avg_episode_entropy:.4f}")
            print(f"  Memory Table Size: {len(his_memory_table.get_table()['query.id'])}")
            if should_train:
                print(f"  Policy Loss: {policy_loss:.4f}, "
                      f"Value Loss: {value_loss:.4f}, "
                      f"Training Entropy: {training_entropy:.4f}")

        # Early stopping
        if running_reward > 10:
            print(f"Problem solved! Episode {episode}, Running Reward: {running_reward:.2f}")
            break

    wandb.finish()
    return agent, episode_rewards


# --------------------------------------------------------------------------- #
#                          Argument parsing                                     #
# --------------------------------------------------------------------------- #

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PPO training for the static router")

    # Data file paths
    parser.add_argument("--query_file", type=str,
                        default="data/router_data_train.csv",
                        help="Training query file path")
    parser.add_argument("--llm_file", type=str,
                        default="config/llm_descriptions_with_embeddings.json",
                        help="LLM descriptions file path")
    parser.add_argument("--agent_file", type=str,
                        default="config/agent_roles_with_embeddings.json",
                        help="Agent roles file path")
    parser.add_argument("--test_query_file", type=str,
                        default="data/router_data_test.csv",
                        help="Test query file path")

    # Model parameters
    parser.add_argument("--width", type=int, default=3,
                        help="Decomposition width (number of sub-queries)")
    parser.add_argument("--depth", type=int, default=1,
                        help="Decomposition depth")
    parser.add_argument("--embed_dim", type=int, default=16,
                        help="Embedding dimension")
    parser.add_argument("--gnn_backbone", type=str, default="dblayer",
                        choices=["dblayer", "homo", "hetero"],
                        help="GNN backbone type")

    # Training parameters
    parser.add_argument("--max_episodes", type=int, default=1000,
                        help="Maximum number of training episodes")
    parser.add_argument("--max_timesteps", type=int, default=500,
                        help="Maximum timesteps per episode")
    parser.add_argument("--update_timestep", type=int, default=2000,
                        help="Update timestep interval")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Training batch size (parallel episodes)")
    parser.add_argument("--test_batch_size", type=int, default=20,
                        help="Test batch size")
    parser.add_argument("--test_before_train", type=int, default=0,
                        help="Run initial test before training (1=yes, 0=no)")
    parser.add_argument("--test_every_n_episodes", type=int, default=50,
                        help="Test every N episodes")

    # Experiment parameters
    parser.add_argument("--wandb_project", type=str, default="router_planner",
                        help="Weights & Biases project name")
    parser.add_argument("--experiment_name", type=str, default="experiment_v1",
                        help="Experiment name for logging")
    parser.add_argument("--gpu_id", type=int, default=1,
                        help="GPU device ID")

    return parser.parse_args()


def main():
    """Parse arguments and start training."""
    args = parse_args()

    print("Training parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-" * 50)

    agent, rewards = train_ppo(
        query_file=args.query_file,
        llm_file=args.llm_file,
        agent_file=args.agent_file,
        test_query_file=args.test_query_file,
        width=args.width,
        depth=args.depth,
        embed_dim=args.embed_dim,
        max_episodes=args.max_episodes,
        max_timesteps=args.max_timesteps,
        update_timestep=args.update_timestep,
        wandb_project=args.wandb_project,
        experiment_name=args.experiment_name,
        gpu_id=args.gpu_id,
        test_every_n_episodes=args.test_every_n_episodes,
        batch_size=args.batch_size,
        test_batch_size=args.test_batch_size,
        test_before_train=args.test_before_train,
        gnn_backbone=args.gnn_backbone,
    )

    return agent, rewards
