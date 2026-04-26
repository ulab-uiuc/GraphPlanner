"""
Training script for the Dynamic Router using PPO.

Orchestrates the full training loop including:
- Batch episode collection with thread-parallel environment interaction
- PPO policy updates with action masking
- Periodic evaluation on a held-out test set
- Model checkpointing (best + latest + final)
- Weights & Biases experiment tracking
- Historical memory table persistence across episodes

Usage:
    python -m router_planner.dynamic_router.train --query_file <path> --llm_file <path> ...
"""

import os
import argparse
import time
import threading
from .network import RouteAgent
from .route_env import Env_route
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing.dummy import Pool as ThreadPool
import copy
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import random
import warnings
import gc
import pickle

# Authenticate with wandb using environment variable (set WANDB_API_KEY in your environment)
wandb_key = os.environ.get("WANDB_API_KEY")
if wandb_key:
    wandb.login(key=wandb_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def operate_global(request):
    """Run a single training episode with action masking."""
    agent, env, state, agent_llm_emb, query_id_embed, LLM_role_id_embed, \
        local_memory_table, his_index, his_memory_table, max_timesteps, memory_table_obj, action_mask, force_planner_first = (
        request['agent'],
        request['env'],
        request['state'],
        request['agent_llm_emb'],
        request['query_id_embed'],
        request['LLM_role_id_embed'],
        request['local_memory_table'],
        request['his_index'],
        request['his_memory_table'],
        request['max_timesteps'],
        request['memory_table_obj'],
        request['action_mask'],
        request.get('force_planner_first', False)
    )

    state = torch.stack([state]).to(device)

    his_tables = {
        "memory_table": his_memory_table,
        "query_table": query_id_embed,
        "LLM_table": LLM_role_id_embed
    }
    local_tables = {
        "memory_table": local_memory_table,
        "query_table": query_id_embed,
        "LLM_table": LLM_role_id_embed
    }
    episode_reward = 0
    episode_entropies = []
    transitions = []

    for t in range(max_timesteps):
        his_index = torch.tensor(his_index).to(device)
        # On first timestep, optionally force planner action
        current_action_mask = action_mask
        if t == 0 and force_planner_first:
            planner_mask = np.zeros_like(action_mask)
            for action_idx, action_info in env.action_map.items():
                if action_info["role"] == "planner":
                    planner_mask[action_idx] = 1.0

            # Only use planner mask if there are valid planner actions available
            if planner_mask.sum() > 0:
                current_action_mask = planner_mask

        # Select action with masking
        action, log_prob, value, entropy = agent.select_action(
            state, local_tables, his_tables, his_index,
            action_mask=current_action_mask, greedy=False
        )
        episode_entropies.append(entropy)

        # Environment step (returns new action mask)
        next_state, done, cost, reward, next_agent_llm_emb, local_memory_table, his_index, next_action_mask = env.next_step(
            action=action)

        next_state_tensor = torch.stack([next_state]).to(device)
        local_tables = {
            "memory_table": local_memory_table,
            "query_table": query_id_embed,
            "LLM_table": LLM_role_id_embed
        }

        # Store transition with the action mask that was actually used
        transitions.append(
            (state, action, reward, log_prob, value, done, local_memory_table, his_index, current_action_mask))
        episode_reward += reward

        state = next_state_tensor
        agent_llm_emb = next_agent_llm_emb
        action_mask = next_action_mask

        if done:
            break

    episode_length = t + 1

    # Update historical memory table if in training mode
    if memory_table_obj is not None:
        update_record = {}
        for key in local_memory_table:
            if key in ["query.id", "LLM_role.id", "score", "cost", "n_i", "n_o"]:
                update_record[key] = local_memory_table[key]

        if update_record and any(len(v) > 0 if isinstance(v, list) else v is not None for v in update_record.values()):
            memory_table_obj.update(update_record)

    return {
        "reward": episode_reward,
        "entropy": episode_entropies,
        "transitions": transitions,
        "episode_length": episode_length
    }


def test_agent_on_dataset_subprocess(request):
    """Run a single test episode with greedy action selection."""
    agent, test_env, state, agent_llm_emb, query_id_embed, LLM_role_id_embed, \
        local_memory_table, his_memory_table, his_index, max_timesteps, action_mask = (
        request['agent'], request['test_env'], request['state'], request['agent_llm_emb'],
        request['query_id_embed'], request['LLM_role_id_embed'], request['local_memory_table'],
        request['his_memory_table'], request['his_index'], request['max_timesteps'],
        request['action_mask']
    )

    with torch.no_grad():
        state = torch.stack([state]).to(device)
        his_tables = {
            "memory_table": his_memory_table,
            "query_table": query_id_embed,
            "LLM_table": LLM_role_id_embed
        }
        local_tables = {
            "memory_table": local_memory_table,
            "query_table": query_id_embed,
            "LLM_table": LLM_role_id_embed
        }
        episode_reward = 0
        episode_cost = 0

        for t in range(max_timesteps):
            his_index = torch.tensor(his_index).to(device)
            action, _, _, _ = agent.select_action(
                state, local_tables, his_tables, his_index,
                action_mask=action_mask, greedy=True
            )

            # Environment step
            next_state, done, cost, reward, next_agent_llm_emb, local_memory_table, his_index, next_action_mask = test_env.next_step(
                action=action)

            next_state_tensor = torch.stack([next_state]).to(device)
            local_tables = {
                "memory_table": local_memory_table,
                "query_table": query_id_embed,
                "LLM_table": LLM_role_id_embed
            }
            episode_reward += reward
            episode_cost += cost
            state = next_state_tensor
            action_mask = next_action_mask

            if done:
                break

    return episode_reward, t + 1, episode_cost


def test_agent_on_dataset(agent, test_query_file, llm_file, agent_file, max_planner_calls=5,
                          max_timesteps=500, batch_size=2, his_memory=None):
    """Test trained agent on the full test dataset."""

    test_start_time = time.time()
    his_memory_table = his_memory.get_table()

    test_data = pd.read_csv(test_query_file)
    num_test_samples = len(test_data)
    print(f"Testing on {num_test_samples} samples from test dataset...")

    # Create test environment
    test_env = Env_route(
        query_file=test_query_file,
        test_query_file=test_query_file,
        llm_file=llm_file,
        agent_file=agent_file,
        max_planner_calls=max_planner_calls,
        device=device
    )
    test_env.enable_test_mode()

    test_rewards = []
    test_episode_lengths = []
    test_costs = []

    if hasattr(agent, 'eval'):
        agent.eval()

    for ii in tqdm(range(0, num_test_samples, batch_size), desc="Testing Batches"):
        test_envs = []
        states = []
        initial_embs = []
        local_mems = []
        his_indexs = []
        action_masks = []

        for jj in range(ii, min(ii + batch_size, num_test_samples)):
            # Reset returns action mask
            state, agent_llm_emb, query_id_embed, LLM_role_id_embed, local_memory_table, his_index, action_mask = test_env.reset()
            test_envs.append(copy.copy(test_env))
            states.append(state)
            initial_embs.append(agent_llm_emb)
            local_mems.append(local_memory_table)
            his_indexs.append(his_index)
            action_masks.append(action_mask)

        agents = [agent] * len(test_envs)
        [i.eval() if hasattr(i, 'eval') else None for i in agents]

        requests = [{
            'agent': agents[i],
            'test_env': test_envs[i],
            'state': states[i],
            'agent_llm_emb': initial_embs[i],
            'query_id_embed': query_id_embed,
            'LLM_role_id_embed': LLM_role_id_embed,
            'local_memory_table': local_mems[i],
            'his_memory_table': his_memory_table,
            'his_index': his_indexs[i],
            'max_timesteps': max_timesteps,
            'action_mask': action_masks[i]
        } for i in range(len(test_envs))]

        results = []
        with torch.no_grad():
            with ThreadPool(batch_size) as executor:
                for r in tqdm(
                        executor.imap_unordered(test_agent_on_dataset_subprocess, requests),
                        total=len(requests),
                        desc="Processing",
                        ncols=100
                ):
                    results.append(r)

        for res in results:
            test_rewards.append(res[0])
            test_episode_lengths.append(res[1])
            test_costs.append(res[2])

        del test_envs
        del states
        del initial_embs
        del agents
        del requests
        del results

        gc.collect()
        torch.cuda.empty_cache()

    test_env.disable_test_mode()

    if hasattr(agent, 'train'):
        agent.train()

    test_end_time = time.time()
    test_duration = test_end_time - test_start_time

    test_stats = {
        'test_avg_reward': np.mean(test_rewards),
        'test_std_reward': np.std(test_rewards),
        'test_max_reward': np.max(test_rewards),
        'test_min_reward': np.min(test_rewards),
        'test_avg_episode_length': np.mean(test_episode_lengths),
        'test_avg_cost': np.mean(test_costs),
        'test_total_cost': np.sum(test_costs),
        'test_samples': num_test_samples,
        'test_duration': test_duration
    }

    return test_stats


class MemoryTable:
    """Thread-safe historical memory table for cross-episode experience storage."""

    def __init__(self):
        self.his_memory_table = {
            "query.id": [],
            "LLM_role.id": [],
            "score": [],
            "cost": [],
            "n_i": [],
            "n_o": []
        }
        self.lock = threading.Lock()

    def update(self, record: dict):
        """Add one or more records to the memory table."""
        is_batch = any(isinstance(v, list) for v in record.values())

        if is_batch:
            lengths = [len(v) for v in record.values() if isinstance(v, list)]
            if len(set(lengths)) > 1:
                raise ValueError("All lists must have same length")
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
        """Get a snapshot of the memory table."""
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

    def save_to_file(self, filepath):
        """Save memory table to file."""
        with self.lock:
            data_to_save = {k: list(v) for k, v in self.his_memory_table.items()}

        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        print(f"Memory table saved to: {filepath}")

    def load_from_file(self, filepath):
        """Load memory table from file."""
        try:
            with open(filepath, 'rb') as f:
                loaded_data = pickle.load(f)

            with self.lock:
                self.his_memory_table = loaded_data
            print(f"Memory table loaded from: {filepath}")
            print(f"Loaded {len(self.his_memory_table['query.id'])} entries")
        except FileNotFoundError:
            print(f"Memory table file not found: {filepath}")
            print("Starting with empty memory table")
        except Exception as e:
            print(f"Error loading memory table: {e}")
            print("Starting with empty memory table")

    def get_size(self):
        """Get the number of entries in memory table."""
        with self.lock:
            return len(self.his_memory_table["query.id"])

    def __repr__(self):
        return str(self.his_memory_table)


def train_ppo(query_file, llm_file, agent_file, test_query_file=None,
              max_planner_calls=5, embed_dim=768 * 2, max_episodes=1000,
              max_timesteps=500, update_timestep=2000, wandb_project="ppo_routing_flexible",
              experiment_name=None, gpu_id=None, test_every_n_episodes=10,
              batch_size=1, test_batch_size=1, test_before_train=True,
              save_dir="./checkpoints", load_best_model_path=None,
              force_planner_episodes=50, gnn_backbone="dblayer"):
    """Train PPO agent with flexible architecture and planner call limit."""

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    if experiment_name is None:
        experiment_name = f"ppo_routing_flexible_{embed_dim}"

    config = {
        "embed_dim": embed_dim,
        "max_episodes": max_episodes,
        "max_timesteps": max_timesteps,
        "max_planner_calls": max_planner_calls,
        "force_planner_episodes": force_planner_episodes,
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
        "save_dir": save_dir,
        "load_best_model_path": load_best_model_path
    }

    if test_query_file:
        test_data = pd.read_csv(test_query_file)
        config["test_dataset_size"] = len(test_data)

    wandb.init(
        project=wandb_project,
        name=experiment_name,
        config=config
    )

    # Best model tracking variables
    best_running_reward = float('-inf')
    best_model_path = os.path.join(save_dir, f"{experiment_name}_best_model.pth")
    best_memory_path = os.path.join(save_dir, f"{experiment_name}_best_memory.pkl")
    latest_model_path = os.path.join(save_dir, f"{experiment_name}_latest_model.pth")
    latest_memory_path = os.path.join(save_dir, f"{experiment_name}_latest_memory.pkl")

    print(f"Best model will be saved to: {best_model_path}")
    print(f"Best memory will be saved to: {best_memory_path}")
    print(f"Latest model will be saved to: {latest_model_path}")
    print(f"Latest memory will be saved to: {latest_memory_path}")

    # Create PPO agent
    agent = RouteAgent(embed_dim=768, candidate_embed_dim=768 * 2, hidden_dim=32,
                       buffer_size=50, device=device, gnn_backbone=gnn_backbone)

    # Load best model if provided
    if load_best_model_path and os.path.exists(load_best_model_path):
        print(f"Loading best model from: {load_best_model_path}")
        agent.load_model(load_best_model_path)

        # Try to load corresponding memory table
        memory_path = load_best_model_path.replace('_best_model.pth', '_best_memory.pkl')
        if os.path.exists(memory_path):
            print(f"Found corresponding memory table: {memory_path}")
        else:
            print(f"No corresponding memory table found at: {memory_path}")

    # Create environment with planner call limit
    env = Env_route(query_file=query_file, test_query_file=test_query_file,
                    llm_file=llm_file, agent_file=agent_file,
                    max_planner_calls=max_planner_calls, device=device)

    # Create and optionally load memory table
    his_memory_table = MemoryTable()

    if load_best_model_path:
        memory_path = load_best_model_path.replace('_best_model.pth', '_best_memory.pkl')
        his_memory_table.load_from_file(memory_path)
        best_running_reward = 0.0

    global_query_id_embed = None
    global_LLM_role_id_embed = None

    if test_before_train == 1:
        initial_test_stats = None
        if test_query_file:
            print("=" * 60)
            print("INITIAL TESTING BEFORE TRAINING STARTS")
            print("=" * 60)
            with torch.no_grad():
                initial_test_stats = test_agent_on_dataset(
                    agent=agent,
                    test_query_file=test_query_file,
                    llm_file=llm_file,
                    agent_file=agent_file,
                    max_planner_calls=max_planner_calls,
                    max_timesteps=max_timesteps,
                    batch_size=test_batch_size,
                    his_memory=his_memory_table
                )

            print(f"Initial Test Results (Before Training):")
            print(f"  Average Reward: {initial_test_stats['test_avg_reward']:.2f}")

            initial_log_dict = {
                "episode": -1,
                "initial_test_avg_reward": initial_test_stats['test_avg_reward'],
                "memory_table_size": his_memory_table.get_size()
            }
            wandb.log(initial_log_dict)
            print("=" * 60)

    episode_rewards = []
    running_reward = 0
    timestep = 0
    first_training_done = False

    print(f"Starting PPO training with flexible architecture")
    print(f"Max planner calls allowed: {max_planner_calls}")
    print(f"Force planner episodes: {force_planner_episodes}")
    print(f"Device: {device}")
    print(f"Initial memory table size: {his_memory_table.get_size()}")
    print("-" * 50)

    def batch_operate(initial_agent, initial_env, batch_size=1, his_memory_table=None, force_planner_first=False):
        """Run a batch of training episodes in parallel."""
        envs = []
        reset_results = []
        for _ in range(batch_size):
            reset_result = initial_env.reset()
            envs.append(copy.copy(initial_env))
            reset_results.append(reset_result)

        states = [result[0] for result in reset_results]
        initial_embs = [result[1] for result in reset_results]
        query_id_embeds = [result[2] for result in reset_results]
        LLM_role_id_embeds = [result[3] for result in reset_results]
        local_memory_tables = [result[4] for result in reset_results]
        his_indexs = [result[5] for result in reset_results]
        action_masks = [result[6] for result in reset_results]

        nonlocal global_query_id_embed, global_LLM_role_id_embed
        if global_query_id_embed is None:
            global_query_id_embed = query_id_embeds[0]
            global_LLM_role_id_embed = LLM_role_id_embeds[0]

        agents = [copy.copy(initial_agent) for _ in range(batch_size)]

        requests = [
            {
                'agent': agents[i],
                'env': envs[i],
                'state': states[i],
                'agent_llm_emb': initial_embs[i],
                'query_id_embed': query_id_embeds[i],
                'LLM_role_id_embed': LLM_role_id_embeds[i],
                'local_memory_table': local_memory_tables[i],
                'his_index': his_indexs[i],
                'his_memory_table': his_memory_table.get_table(),
                'max_timesteps': max_timesteps,
                'memory_table_obj': his_memory_table,
                'action_mask': action_masks[i],
                'force_planner_first': force_planner_first
            }
            for i in range(batch_size)
        ]

        results = []
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = [executor.submit(operate_global, request) for request in requests]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                results.append(future.result())
        del futures
        torch.cuda.empty_cache()

        avg_episode_entropy = sum(sum(result['entropy']) for result in results) / sum(
            len(result['entropy']) for result in results)
        reward = sum(result['reward'] for result in results) / batch_size
        avg_episode_length = sum(result['episode_length'] for result in results) / batch_size

        for result in results:
            for transition in result['transitions']:
                initial_agent.store_transition(*transition)

        return avg_episode_entropy, reward, avg_episode_length

    for episode in range(max_episodes):
        episode_start_time = time.time()

        # Determine if this episode should force planner first
        force_planner_first = episode < force_planner_episodes

        avg_episode_entropy, episode_reward, avg_episode_length = batch_operate(
            agent, env, batch_size, his_memory_table, force_planner_first=force_planner_first
        )

        should_train = False
        training_reason = ""

        if agent.is_buffer_full() and not first_training_done:
            should_train = True
            first_training_done = True
            training_reason = "Buffer is full (first training)"
        elif first_training_done and ((episode) % 3 == 0 or batch_size > 1):
            should_train = True
            training_reason = f"Scheduled training"

        if should_train:
            policy_loss, value_loss, training_entropy = agent.update(
                global_query_id_embed,
                global_LLM_role_id_embed,
                his_memory_table, 32
            )
            print(f"Episode {episode}: {training_reason}")
        else:
            policy_loss, value_loss, training_entropy = 0, 0, 0

        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        episode_rewards.append(episode_reward)

        # Check if this is the best model and save both model and memory
        if running_reward > best_running_reward:
            best_running_reward = running_reward
            agent.save_model(best_model_path)
            his_memory_table.save_to_file(best_memory_path)
            print(f"New best running reward: {best_running_reward:.4f} - Model and memory saved!")

            wandb.log({
                "best_running_reward": best_running_reward,
                "best_model_episode": episode
            })

        # Periodically save latest model and memory
        if episode % 50 == 0 and episode > 0:
            agent.save_model(latest_model_path)
            his_memory_table.save_to_file(latest_memory_path)
            print(f"Latest model and memory checkpoint saved at episode {episode}")

        episode_end_time = time.time()
        episode_duration = episode_end_time - episode_start_time

        log_dict = {
            "episode": episode,
            "episode_reward": episode_reward,
            "running_reward": running_reward,
            "best_running_reward": best_running_reward,
            "episode_entropy": avg_episode_entropy,
            "buffer_size": len(agent.buffer),
            "episode_length": avg_episode_length,
            "episode_duration": episode_duration,
            "training_occurred": should_train,
            "memory_table_size": his_memory_table.get_size(),
            "force_planner_first": force_planner_first
        }

        if should_train:
            log_dict.update({
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "training_entropy": training_entropy
            })

        if test_query_file and (episode + 1) % test_every_n_episodes == 0:
            print(f"Testing at episode {episode + 1}...")
            with torch.no_grad():
                test_stats = test_agent_on_dataset(
                    agent=agent,
                    test_query_file=test_query_file,
                    llm_file=llm_file,
                    agent_file=agent_file,
                    max_planner_calls=max_planner_calls,
                    max_timesteps=max_timesteps,
                    batch_size=test_batch_size,
                    his_memory=his_memory_table
                )

            log_dict.update(test_stats)
            print(f"Test Results: Avg Reward = {test_stats['test_avg_reward']:.2f}")

        wandb.log(log_dict)

        if episode % 100 == 0:
            print(
                f"Episode {episode}, Reward: {episode_reward:.2f}, Running: {running_reward:.2f}, Best: {best_running_reward:.2f}, Memory: {his_memory_table.get_size()}, Force Planner: {force_planner_first}")

        if running_reward > 10:
            print(f"Problem solved! Episode {episode}")
            final_model_path = os.path.join(save_dir, f"{experiment_name}_final_solved_model.pth")
            final_memory_path = os.path.join(save_dir, f"{experiment_name}_final_solved_memory.pkl")
            agent.save_model(final_model_path)
            his_memory_table.save_to_file(final_memory_path)
            print(f"Final solved model saved to: {final_model_path}")
            print(f"Final solved memory saved to: {final_memory_path}")
            break

    # Save final model and memory at end of training
    final_model_path = os.path.join(save_dir, f"{experiment_name}_final_model.pth")
    final_memory_path = os.path.join(save_dir, f"{experiment_name}_final_memory.pkl")
    agent.save_model(final_model_path)
    his_memory_table.save_to_file(final_memory_path)
    print(f"Final training model saved to: {final_model_path}")
    print(f"Final training memory saved to: {final_memory_path}")

    print(f"\nTraining completed:")
    print(f"Best running reward achieved: {best_running_reward:.4f}")
    print(f"Best model saved at: {best_model_path}")
    print(f"Best memory saved at: {best_memory_path}")
    print(f"Latest model saved at: {latest_model_path}")
    print(f"Latest memory saved at: {latest_memory_path}")
    print(f"Final model saved at: {final_model_path}")
    print(f"Final memory saved at: {final_memory_path}")
    print(f"Final memory table size: {his_memory_table.get_size()}")

    wandb.finish()
    return agent, episode_rewards, best_model_path, best_memory_path


def parse_args():
    parser = argparse.ArgumentParser(description='Flexible PPO training with action masking')

    parser.add_argument('--query_file', type=str,
                        default="data/router_data_train.csv")
    parser.add_argument('--llm_file', type=str,
                        default="config/llm_descriptions_with_embeddings.json")
    parser.add_argument('--agent_file', type=str,
                        default="config/agent_roles_with_embeddings.json")
    parser.add_argument('--test_query_file', type=str,
                        default="data/router_data_test.csv")

    parser.add_argument('--max_planner_calls', type=int, default=2,
                        help='Maximum number of planner calls allowed per episode')
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--max_episodes', type=int, default=1000)
    parser.add_argument('--max_timesteps', type=int, default=500)
    parser.add_argument('--update_timestep', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=20)
    parser.add_argument('--test_before_train', type=int, default=0)
    parser.add_argument('--test_every_n_episodes', type=int, default=20)
    parser.add_argument('--wandb_project', type=str, default="router_planner_flexible")
    parser.add_argument('--experiment_name', type=str, default="flexible_v1")
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default="./checkpoints_all",
                        help='Directory to save model checkpoints')
    parser.add_argument('--load_best_model_path', type=str, default=None,
                        help='Path to load best model from previous training (will also load corresponding memory table)')
    parser.add_argument('--force_planner_episodes', type=int, default=0,
                        help='Number of episodes to force planner action as first step')
    parser.add_argument("--gnn_backbone", type=str, default="dblayer", choices=["dblayer", "homo", "hetero"])

    return parser.parse_args()


def load_best_model(model_path, embed_dim=768, candidate_embed_dim=768 * 2, hidden_dim=32, device=None, gnn_backbone="dblayer"):
    """
    Load a saved best model.

    Args:
        model_path: Path to the model file.
        embed_dim: Embedding dimension.
        candidate_embed_dim: Candidate embedding dimension.
        hidden_dim: Hidden layer dimension.
        device: Target device.
        gnn_backbone: GNN backbone type.

    Returns:
        agent: Agent with loaded best weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = RouteAgent(
        embed_dim=embed_dim,
        candidate_embed_dim=candidate_embed_dim,
        hidden_dim=hidden_dim,
        buffer_size=100,
        device=device,
        gnn_backbone=gnn_backbone
    )

    agent.load_model(model_path)

    print(f"Best model loaded from: {model_path}")
    return agent


def load_best_model_with_memory(model_path, embed_dim=768, candidate_embed_dim=768 * 2, hidden_dim=32, device=None, gnn_backbone="dblayer"):
    """
    Load a saved best model and its corresponding memory table.

    Args:
        model_path: Path to the model file.
        embed_dim: Embedding dimension.
        candidate_embed_dim: Candidate embedding dimension.
        hidden_dim: Hidden layer dimension.
        device: Target device.
        gnn_backbone: GNN backbone type.

    Returns:
        agent: Agent with loaded best weights.
        memory_table: MemoryTable loaded with historical data.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = RouteAgent(
        embed_dim=embed_dim,
        candidate_embed_dim=candidate_embed_dim,
        hidden_dim=hidden_dim,
        buffer_size=100,
        device=device,
        gnn_backbone=gnn_backbone
    )

    agent.load_model(model_path)
    print(f"Best model loaded from: {model_path}")

    # Create and load memory table
    memory_table = MemoryTable()
    memory_path = model_path.replace('_best_model.pth', '_best_memory.pkl')
    memory_table.load_from_file(memory_path)

    return agent, memory_table


def main():
    args = parse_args()

    print("Training parameters:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("-" * 50)

    agent, rewards, best_model_path, best_memory_path = train_ppo(
        query_file=args.query_file,
        llm_file=args.llm_file,
        agent_file=args.agent_file,
        test_query_file=args.test_query_file,
        max_planner_calls=args.max_planner_calls,
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
        save_dir=args.save_dir,
        load_best_model_path=args.load_best_model_path,
        force_planner_episodes=args.force_planner_episodes,
        gnn_backbone=args.gnn_backbone
    )

    print(f"\nBest model can be loaded using:")
    print(f"best_agent = load_best_model('{best_model_path}')")
    print(f"best_agent, memory_table = load_best_model_with_memory('{best_model_path}')")
    return agent, rewards, best_model_path, best_memory_path


if __name__ == "__main__":
    main()
