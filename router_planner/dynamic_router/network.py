"""
Neural network components for the Dynamic Router.

Architecture:
- TableLayer: Linear transformation with optional skip connections
- DBLayer: Relational database-inspired aggregation layer using scatter operations
- HomoGCNDBLayer / HeteroGCNDBLayer: GNN-based alternatives to DBLayer
- PolicyNetwork: Dual-tower model (state + LLM candidate) with action masking
- ValueNetwork: State value estimation with local + historical table fusion
- RouteAgent: PPO agent with experience buffer and GAE advantage estimation

The dynamic router uses dual DBLayers to process both local (current episode)
and historical (cross-episode) memory tables, enabling experience transfer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_scatter import scatter
import numpy as np
from collections import deque, defaultdict
import random
import copy

from .gnn_baselines import HeteroGCNDBLayer, HomoGCNDBLayer


# ============================================================================
# Table Layer
# ============================================================================
class TableLayer(nn.Module):
    """
    Linear transformation layer with optional skip connection,
    layer normalization, and activation.
    """

    def __init__(self, dim_in, dim_out, dropout=0.0, norm=nn.LayerNorm, act=nn.ReLU, skip=True):
        super().__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.layer_post = nn.Sequential(
            *([nn.Dropout(dropout)] if dropout > 0 else [])
            + ([norm(dim_out)] if norm else [])
            + ([act()] if act else [])
        )
        self.skip = skip

    def forward(self, x):
        if self.skip and x.shape[1] == self.layer.out_features:
            return x + self.layer_post(self.layer(x))
        return self.layer_post(self.layer(x))


# ============================================================================
# Database Layer - Relational Aggregation
# ============================================================================
class DBLayer(nn.Module):
    """
    Database-inspired relational aggregation layer.

    Processes multiple tables (query, LLM, memory) through per-table linear
    layers, then performs scatter-based aggregation using foreign key
    relationships to propagate information across tables.
    """

    def __init__(self, table_dims: dict, dim_out: int, skip_ratio: bool = True, device=None):
        super().__init__()
        self.device = device
        self.layer = nn.ModuleDict()
        self.skip_ratio = nn.Parameter(torch.tensor(1.0, device=device)) if skip_ratio else 1.0

        for t_name in table_dims:
            self.layer[t_name] = TableLayer(table_dims[t_name], dim_out)
            if device is not None:
                self.layer[t_name] = self.layer[t_name].to(device)

    def forward(self, tables: dict, reduce: str = "mean"):
        # Determine table processing order (query -> LLM -> memory -> others)
        table_order = []
        for key in ["query_table", "LLM_table", "memory_table"]:
            if key in tables:
                table_order.append(key)
        for key in tables:
            if key not in table_order:
                table_order.append(key)

        # Collect global meta-key mappings for consistent scatter indexing
        global_meta_keys = defaultdict(set)
        for t_name in table_order:
            td = tables[t_name]
            if not td or len(list(td.values())[0]) == 0:
                continue
            for key, values in td.items():
                if ".id" in key:
                    global_meta_keys[key.split(".")[0]].update(values)

        global_mappings = {}
        global_sizes = {}
        for mk, vals in global_meta_keys.items():
            sorted_vals = sorted(list(vals))
            global_mappings[mk] = {v: i for i, v in enumerate(sorted_vals)}
            global_sizes[mk] = len(sorted_vals)

        key_feat = {}
        key_count = defaultdict(int)
        result_embeddings = {}

        # Step 1: Forward each table through its dedicated layer
        for t_name in table_order:
            td = tables[t_name]
            if not td or len(list(td.values())[0]) == 0:
                result_embeddings[t_name] = torch.empty(0, self.layer[t_name].layer.out_features, device=self.device)
                continue

            if t_name == "memory_table":
                n_i = torch.stack([t.to(self.device) for t in td["n_i"]])
                n_o = torch.stack([t.to(self.device) for t in td["n_o"]])
                scores = torch.tensor(td["score"], dtype=torch.float32, device=self.device).unsqueeze(1)
                costs = torch.tensor(td["cost"], dtype=torch.float32, device=self.device).unsqueeze(1)
                feat_tensor = torch.cat([n_i, n_o, scores, costs], dim=1)
            elif t_name == "query_table":
                feat_tensor = torch.stack([t.to(self.device) for t in td["query_embedding"]])
            elif t_name == "LLM_table":
                embs = []
                for emb in td["embedding"]:
                    if isinstance(emb, torch.Tensor):
                        embs.append(emb.to(self.device))
                    else:
                        embs.append(torch.tensor(emb, dtype=torch.float32, device=self.device))
                feat_tensor = torch.stack(embs)
            else:
                raise ValueError(f"Unknown table: {t_name}")

            new_feat = self.layer[t_name](feat_tensor)
            result_embeddings[t_name] = new_feat

            # Step 2: Scatter-aggregate by foreign keys
            for key, values in td.items():
                if ".id" in key:
                    mk = key.split(".")[0]
                    codes = [global_mappings[mk][v] for v in values]
                    k_tensor = torch.tensor(codes, dtype=torch.long, device=self.device)
                    scattered = scatter(new_feat, k_tensor, dim=0, reduce=reduce, dim_size=global_sizes[mk])
                    if mk not in key_feat:
                        key_feat[mk] = scattered
                    else:
                        key_feat[mk] += scattered
                    key_count[mk] += 1

        # Step 3: Normalize aggregated features
        for mk in key_feat:
            if key_count[mk] > 0:
                key_feat[mk] /= key_count[mk]

        # Step 4: Propagate aggregated features back to each table via skip connection
        for t_name in table_order:
            td = tables[t_name]
            if not td or len(list(td.values())[0]) == 0:
                continue
            feat = result_embeddings[t_name]
            for key, values in td.items():
                if ".id" in key:
                    mk = key.split(".")[0]
                    if mk in key_feat:
                        codes = [global_mappings[mk][v] for v in values]
                        indices = torch.tensor(codes, dtype=torch.long, device=self.device)
                        skip_val = self.skip_ratio if isinstance(self.skip_ratio, torch.Tensor) else torch.tensor(self.skip_ratio, device=self.device)
                        feat = feat + key_feat[mk][indices].to(feat.device) * skip_val.to(feat.device)
            result_embeddings[t_name] = feat

        return result_embeddings


# ============================================================================
# Policy Network
# ============================================================================
class PolicyNetwork(nn.Module):
    """
    Policy network with dual DBLayers for local and historical tables.

    Processes historical tables through his_dbl to obtain enriched LLM embeddings,
    then feeds them into local_dbl along with local episode data.
    Uses query embedding and task embedding for context-aware state representation.
    Supports multiple GNN backbones: 'dblayer', 'homo', 'hetero'.
    """

    def __init__(self, state_embed_dim, candidate_embed_dim=768, hidden_dim=256,
                 num_heads=8, device=None, gnn_backbone="dblayer"):
        super().__init__()
        self.device = device

        local_table_dims = {
            "query_table": state_embed_dim,
            "LLM_table": hidden_dim,
            "memory_table": state_embed_dim * 2 + 2,
        }
        his_table_dims = {
            "query_table": state_embed_dim,
            "LLM_table": state_embed_dim * 2,
            "memory_table": state_embed_dim * 2 + 2,
        }

        # Select backbone for both local and historical processing
        if gnn_backbone == "homo":
            self.local_dbl = HomoGCNDBLayer(local_table_dims, hidden_dim, True, device)
            self.his_dbl = HomoGCNDBLayer(his_table_dims, hidden_dim, True, device)
        elif gnn_backbone == "hetero":
            self.local_dbl = HeteroGCNDBLayer(local_table_dims, hidden_dim, True, device)
            self.his_dbl = HeteroGCNDBLayer(his_table_dims, hidden_dim, True, device)
        elif gnn_backbone == "dblayer":
            self.local_dbl = DBLayer(local_table_dims, hidden_dim, True, device)
            self.his_dbl = DBLayer(his_table_dims, hidden_dim, True, device)
        else:
            raise ValueError(f"Unknown GNN backbone: {gnn_backbone}")

        # State processing tower
        self.state_tower = nn.Sequential(
            nn.Linear(state_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Task type embedding (up to 11 task categories)
        self.class_embedding = nn.Embedding(11, hidden_dim)
        # Query embedding tower
        self.state_tower_query = nn.Sequential(
            nn.Linear(state_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # Fuse state + query + task into unified representation
        self.state_transform = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
        )

    def forward(self, batch_state_embeddings, local_tables, his_tables, his_index, action_mask=None):
        """
        Compute action probabilities with optional action masking.

        Args:
            batch_state_embeddings: List[Tensor] of state embeddings.
            local_tables: Local episode tables dict.
            his_tables: Historical memory tables dict.
            his_index: Tuple of (query_index, task_index) for embedding lookup.
            action_mask: Binary mask (1=valid, 0=invalid) or None.

        Returns:
            action_probs: Tensor [batch_size, num_actions].
        """
        # Extract query and task embeddings
        query_emb = copy.copy(local_tables["query_table"]["query_embedding"][his_index[0]])
        task_emb = nn.ReLU()(self.class_embedding(his_index[1]))
        query_emb = self.state_tower_query(query_emb)

        # Process historical tables to get enriched LLM embeddings
        his_table_embs = self.his_dbl(his_tables, reduce="mean")
        his_llm_embs = his_table_embs["LLM_table"]
        transform_list = list(torch.unbind(his_llm_embs, dim=0))

        # Replace local LLM embeddings with history-enriched ones
        local_copy = copy.copy(local_tables)
        llm_copy = copy.copy(local_tables["LLM_table"])
        llm_copy["embedding"] = transform_list
        local_copy["LLM_table"] = llm_copy

        # Process local tables
        local_embs = self.local_dbl(local_copy, reduce="mean")
        llm_role_embs = local_embs["LLM_table"]  # [num_actions, hidden_dim]

        # Build fused state representation
        state = self.state_tower(batch_state_embeddings[0])
        state = torch.cat([task_emb.reshape(1, -1), query_emb.reshape(1, -1), state.reshape(1, -1)], dim=-1)
        state = self.state_transform(state)

        # Compute state-action matching scores
        scores = state @ llm_role_embs.T

        # Apply action mask
        if action_mask is not None:
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            action_mask = action_mask.to(scores.device)
            scores = scores.masked_fill(action_mask == 0, -1e9)

        return F.softmax(scores, dim=-1)


# ============================================================================
# Value Network
# ============================================================================
class ValueNetwork(nn.Module):
    """
    State value estimation network with dual DBLayers.

    Fuses local table features, historical table features, state embedding,
    query embedding, and task embedding to predict a scalar state value.
    """

    def __init__(self, state_embed_dim, hidden_dim=256, device=None):
        super().__init__()
        self.device = device

        local_dims = {"query_table": state_embed_dim, "LLM_table": hidden_dim, "memory_table": state_embed_dim * 2 + 2}
        his_dims = {"query_table": state_embed_dim, "LLM_table": state_embed_dim * 2, "memory_table": state_embed_dim * 2 + 2}

        self.local_dbl = DBLayer(local_dims, hidden_dim, True, device)
        self.his_dbl = DBLayer(his_dims, hidden_dim, True, device)

        self.state_tower = nn.Sequential(nn.Linear(state_embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.state_tower_query = nn.Sequential(nn.Linear(state_embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.class_embedding = nn.Embedding(11, hidden_dim)
        self.state_transform = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU())

        self.table_aggregator = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.his_aggregator = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.fusion_layer = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, batch_state_embeddings, local_tables, his_tables, his_index):
        if isinstance(batch_state_embeddings, torch.Tensor):
            if len(batch_state_embeddings.shape) == 1:
                batch_state_embeddings = [batch_state_embeddings]
            else:
                batch_state_embeddings = [batch_state_embeddings[i] for i in range(batch_state_embeddings.shape[0])]

        # Historical processing
        his_embs = self.his_dbl(his_tables, reduce="mean")
        his_llm = his_embs["LLM_table"]
        transform_list = list(torch.unbind(his_llm, dim=0))

        local_copy = copy.copy(local_tables)
        llm_copy = copy.copy(local_tables["LLM_table"])
        llm_copy["embedding"] = transform_list
        local_copy["LLM_table"] = llm_copy

        # Local processing
        local_embs = self.local_dbl(local_copy, reduce="mean")

        # Aggregate local features
        local_feats = []
        for key in ["query_table", "LLM_table", "memory_table"]:
            if key in local_embs and local_embs[key].shape[0] > 0:
                local_feats.append(local_embs[key].mean(dim=0))
            else:
                local_feats.append(torch.zeros(self.local_dbl.layer[key].layer.out_features, device=self.device))
        local_agg = self.table_aggregator(torch.cat(local_feats, dim=-1))

        # Aggregate historical features
        his_feats = []
        for key in ["query_table", "LLM_table", "memory_table"]:
            if key in his_embs and his_embs[key].shape[0] > 0:
                his_feats.append(his_embs[key].mean(dim=0))
            else:
                his_feats.append(torch.zeros(self.his_dbl.layer[key].layer.out_features, device=self.device))
        his_agg = self.his_aggregator(torch.cat(his_feats, dim=-1))

        # Query and task embeddings
        query_emb = self.state_tower_query(copy.copy(local_tables["query_table"]["query_embedding"][his_index[0]]))
        task_emb = nn.ReLU()(self.class_embedding(his_index[1]))

        # State fusion
        state = self.state_tower(batch_state_embeddings[0])
        state_fused = self.state_transform(torch.cat([state.reshape(1, -1), query_emb.reshape(1, -1), task_emb.reshape(1, -1)], dim=-1))

        # Final fusion and value prediction
        fused = torch.cat([state_fused.reshape(1, -1), local_agg.reshape(1, -1), his_agg.reshape(1, -1)], dim=-1)
        return self.value_head(self.fusion_layer(fused))


# ============================================================================
# PPO Route Agent
# ============================================================================
class RouteAgent:
    """
    PPO-based routing agent for the dynamic router.

    Manages policy/value networks, experience buffer, and training updates
    with action masking and Generalized Advantage Estimation (GAE).
    """

    def __init__(self, embed_dim=384, candidate_embed_dim=768, hidden_dim=16,
                 lr=1e-4, gamma=0.98, eps_clip=0.2, k_epochs=1,
                 entropy_coef=0.0, num_heads=8, buffer_size=7,
                 device=None, networks=None, gnn_backbone="dblayer"):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.buffer_size = buffer_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if networks is not None:
            self.policy = networks["policy"]
            self.value = networks["value"]
            self.policy_old = networks["policy_old"]
        else:
            self.policy = PolicyNetwork(embed_dim, candidate_embed_dim, hidden_dim, device=self.device, gnn_backbone=gnn_backbone).to(self.device)
            self.value = ValueNetwork(embed_dim, hidden_dim, device=self.device).to(self.device)
            self.policy_old = PolicyNetwork(embed_dim, candidate_embed_dim, hidden_dim, device=self.device, gnn_backbone=gnn_backbone).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=2 * lr)
        self.buffer = deque(maxlen=self.buffer_size)

    def select_action(self, state_embeddings, local_tables, his_tables, his_index, action_mask=None, greedy=False):
        """Select an action using the old policy with optional action masking."""
        with torch.no_grad():
            if action_mask is not None and not isinstance(action_mask, torch.Tensor):
                action_mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            action_probs = self.policy_old(state_embeddings, local_tables, his_tables, his_index, action_mask)
            value = self.value(state_embeddings, local_tables, his_tables, his_index)

        if action_mask is not None:
            action_mask = action_mask.to(action_probs.device)
            masked_probs = action_probs * action_mask
            masked_probs = masked_probs / (masked_probs.sum(dim=-1, keepdim=True) + 1e-10)
        else:
            masked_probs = action_probs

        dist = Categorical(masked_probs)
        action = torch.argmax(masked_probs, dim=-1) if greedy else dist.sample()
        return action.cpu().item(), dist.log_prob(action).cpu().item(), value.cpu().item(), dist.entropy().cpu().item()

    def store_transition(self, state_embeddings, action, reward, log_prob, value, done, local_memory_table, his_index, action_mask):
        """Store a transition in the experience buffer."""
        if isinstance(state_embeddings, torch.Tensor):
            state_embeddings = state_embeddings.to(self.device)
        elif isinstance(state_embeddings, list):
            state_embeddings = [s.to(self.device) for s in state_embeddings]
        if not isinstance(action_mask, torch.Tensor):
            action_mask = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
        else:
            action_mask = action_mask.to(self.device)
        his_index = torch.tensor(his_index, device=self.device)
        self.buffer.append((state_embeddings, action, reward, log_prob, value, done, local_memory_table, his_index, action_mask))

    def compute_gae(self, rewards, values, dones, next_value, lambda_gae=0.85):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        values = values + [next_value]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * lambda_gae * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def is_buffer_full(self):
        return len(self.buffer) >= self.buffer_size

    def update(self, global_query_id_embed, global_LLM_role_id_embed, his_memory_table, batch_size=32):
        """PPO update with mini-batch sampling and action masking."""
        if not self.is_buffer_full():
            return 0, 0, 0

        states, actions, rewards, log_probs, values, dones, local_mems, his_idxs, masks = zip(*self.buffer)
        states = list(states)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards, values, dones = list(rewards), list(values), list(dones)
        masks = torch.stack([m if isinstance(m, torch.Tensor) else torch.tensor(m, dtype=torch.float32) for m in masks]).to(self.device)

        advantages = self.compute_gae(rewards, values, dones, 0)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        his_tables = {"memory_table": his_memory_table.get_table(), "query_table": global_query_id_embed, "LLM_table": global_LLM_role_id_embed}
        dataset_size = len(states)
        indices = list(range(dataset_size))
        total_pl, total_vl, total_ent, n_batches = 0, 0, 0, 0

        for _ in range(self.k_epochs):
            random.shuffle(indices)
            for start in range(0, dataset_size, batch_size):
                bi = indices[start : min(start + batch_size, dataset_size)]
                bs = len(bi)

                b_probs, b_vals = [], []
                for i in range(bs):
                    lt = {"memory_table": local_mems[bi[i]], "query_table": global_query_id_embed, "LLM_table": global_LLM_role_id_embed}
                    ap = self.policy([states[bi[i]]], lt, his_tables, his_idxs[bi[i]], masks[bi[i]].unsqueeze(0))
                    b_probs.append(ap.squeeze(0))
                    sv = self.value([states[bi[i]]], lt, his_tables, his_idxs[bi[i]])
                    b_vals.append(sv.squeeze())

                b_probs = torch.stack(b_probs)
                b_vals = torch.stack(b_vals)
                mp = b_probs * masks[bi]
                mp = mp / (mp.sum(dim=1, keepdim=True) + 1e-10)
                lp = torch.log(mp + 1e-10)
                new_lp = lp.gather(1, actions[bi].unsqueeze(1)).squeeze(1)
                ent = -(mp * lp).sum(dim=1)

                ratios = torch.exp(new_lp - old_log_probs[bi])
                s1 = ratios * advantages[bi]
                s2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[bi]
                pl = -torch.min(s1, s2).mean() - self.entropy_coef * ent.mean()
                vl = F.mse_loss(b_vals.reshape(-1), returns[bi].reshape(-1))

                self.policy_optimizer.zero_grad(); pl.backward(); self.policy_optimizer.step()
                self.value_optimizer.zero_grad(); vl.backward(); self.value_optimizer.step()

                total_pl += pl.item(); total_vl += vl.item(); total_ent += ent.mean().item(); n_batches += 1

        self.policy_old.load_state_dict(self.policy.state_dict())
        return (total_pl / max(n_batches, 1), total_vl / max(n_batches, 1), total_ent / max(n_batches, 1))

    def save_model(self, filepath):
        torch.save({"policy_state_dict": self.policy.state_dict(), "value_state_dict": self.value.state_dict(), "policy_optimizer_state_dict": self.policy_optimizer.state_dict(), "value_optimizer_state_dict": self.value_optimizer.state_dict()}, filepath)

    def load_model(self, filepath):
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value.load_state_dict(ckpt["value_state_dict"])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer_state_dict"])

    def eval(self):
        self.policy.eval(); self.policy_old.eval(); self.value.eval()

    def train(self):
        self.policy.train(); self.policy_old.train(); self.value.train()
