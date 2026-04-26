"""
Neural network components for the Static Router.

The static router uses a fixed decomposition structure (predetermined width
and depth) where the planner/executor/summarizer roles follow a DFS pattern.
The policy network selects which LLM to use for each role.

Key differences from the Dynamic Router:
- No action masking (roles are determined by the environment phase)
- Simpler value network (state-only, no table fusion)
- Role-based LLM selection via integer index (0=planner, 1=executor, 2=summarizer)
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

from .gnn_baselines import HomoGCNDBLayer, HeteroGCNDBLayer


class TableLayer(nn.Module):
    """Linear transformation layer with optional skip connection."""

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


class DBLayer(nn.Module):
    """Database-inspired relational aggregation layer (same as dynamic router)."""

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
        # Determine processing order: query -> LLM -> memory -> others
        table_order = []
        for key in ["query_table", "LLM_table", "memory_table"]:
            if key in tables:
                table_order.append(key)
        for key in tables:
            if key not in table_order:
                table_order.append(key)

        # Collect all unique meta-keys (foreign keys) and build global mappings
        global_meta_keys = defaultdict(set)
        for t_name in table_order:
            td = tables[t_name]
            if not td or len(list(td.values())[0]) == 0:
                continue
            for key, values in td.items():
                if ".id" in key:
                    global_meta_keys[key.split(".")[0]].update(values)

        global_mappings, global_sizes = {}, {}
        for mk, vals in global_meta_keys.items():
            sv = sorted(list(vals))
            global_mappings[mk] = {v: i for i, v in enumerate(sv)}
            global_sizes[mk] = len(sv)

        key_feat, key_count, result_embeddings = {}, defaultdict(int), {}

        # Step 1: Forward each table through its TableLayer
        for t_name in table_order:
            td = tables[t_name]

            # Handle empty tables
            if not td or len(list(td.values())[0]) == 0:
                result_embeddings[t_name] = torch.empty(0, self.layer[t_name].layer.out_features, device=self.device)
                continue

            # Build feature tensor depending on table type
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

            # Step 2: Aggregate features by foreign keys using scatter
            for key, values in td.items():
                if ".id" in key:
                    mk = key.split(".")[0]
                    codes = [global_mappings[mk][v] for v in values]
                    kt = torch.tensor(codes, dtype=torch.long, device=self.device)
                    scattered = scatter(new_feat, kt, dim=0, reduce=reduce, dim_size=global_sizes[mk])
                    if mk not in key_feat:
                        key_feat[mk] = scattered
                    else:
                        key_feat[mk] += scattered
                    key_count[mk] += 1

        # Step 3: Normalize aggregated features
        for mk in key_feat:
            if key_count[mk] > 0:
                key_feat[mk] /= key_count[mk]

        # Step 4: Map aggregated features back to each table via skip connections
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


class PolicyNetwork(nn.Module):
    """
    Policy network for the static router.

    Uses his_index as an integer to select which role's LLM embeddings
    to use for action scoring (0=planner, 1=executor, 2=summarizer).
    """

    def __init__(self, state_embed_dim, candidate_embed_dim=768, hidden_dim=256,
                 num_heads=8, device=None, gnn_backbone="dblayer"):
        super().__init__()
        self.device = device

        local_dims = {"query_table": state_embed_dim, "LLM_table": hidden_dim, "memory_table": state_embed_dim * 2 + 2}
        his_dims = {"query_table": state_embed_dim, "LLM_table": state_embed_dim * 2, "memory_table": state_embed_dim * 2 + 2}

        # Select backbone: DBLayer (default) or GNN variants
        if gnn_backbone == "homo":
            self.local_dbl = HomoGCNDBLayer(local_dims, hidden_dim, True, device)
            self.his_dbl = HomoGCNDBLayer(his_dims, hidden_dim, True, device)
        elif gnn_backbone == "hetero":
            self.local_dbl = HeteroGCNDBLayer(local_dims, hidden_dim, True, device)
            self.his_dbl = HeteroGCNDBLayer(his_dims, hidden_dim, True, device)
        elif gnn_backbone == "dblayer":
            self.local_dbl = DBLayer(local_dims, hidden_dim, True, device)
            self.his_dbl = DBLayer(his_dims, hidden_dim, True, device)
        else:
            raise ValueError(f"Unknown backbone: {gnn_backbone}")

        # State tower maps query embeddings to the hidden space
        self.state_tower = nn.Sequential(
            nn.Linear(state_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )

    def forward(self, batch_state_embeddings, local_tables, his_tables, his_index):
        """
        Forward pass for action probability computation.

        Args:
            batch_state_embeddings: List[Tensor] of state embeddings.
            local_tables: Local episode tables (query, LLM, memory).
            his_tables: Historical memory tables.
            his_index: Integer role index (0=planner, 1=executor, 2=summarizer).

        Returns:
            action_probs: Tensor [batch_size, num_llms_per_role].
        """
        # Process historical tables to get enriched LLM embeddings
        his_embs = self.his_dbl(his_tables, reduce="mean")
        his_llm = his_embs["LLM_table"]

        # Reshape: [3*num_llms, hidden] -> flat list of [hidden] tensors
        # Structure: 3 roles x 9 LLMs = 27 embeddings
        transform_list = list(torch.unbind(his_llm.reshape(3, 9, -1).reshape(27, -1), dim=0))

        # Replace local LLM embeddings with enriched ones from history
        local_copy = dict(local_tables)
        llm_copy = dict(local_tables["LLM_table"])
        llm_copy["embedding"] = transform_list
        local_copy["LLM_table"] = llm_copy

        # Process local tables with enriched LLM embeddings
        local_embs = self.local_dbl(local_copy, reduce="mean")
        local_llm = local_embs["LLM_table"]

        # Select LLM embeddings for the current role
        if local_llm.size(0) == 0:
            role_embs = torch.zeros(9, local_llm.size(1), device=self.device)
        else:
            try:
                role_embs = (local_llm[his_index:his_index + 1]
                             if isinstance(his_index, int) and his_index < local_llm.size(0)
                             else local_llm[his_index])
            except Exception:
                role_embs = local_llm[:1] if local_llm.size(0) > 0 else local_llm

        # Compute state-action matching scores via dot product
        state = self.state_tower(batch_state_embeddings[0])
        scores = state @ role_embs.T
        return F.softmax(scores, dim=-1)


class ValueNetwork(nn.Module):
    """Simple state-only value network for the static router."""

    def __init__(self, state_embed_dim, hidden_dim=256):
        super().__init__()
        self.state_tower = nn.Sequential(
            nn.Linear(state_embed_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch_state_embeddings):
        """
        Estimate state value.

        Args:
            batch_state_embeddings: Tensor [batch_size, state_embed_dim] or [state_embed_dim].

        Returns:
            values: Tensor [batch_size, 1] or scalar.
        """
        if len(batch_state_embeddings.shape) == 1:
            batch_state_embeddings = batch_state_embeddings.unsqueeze(0)
            single_input = True
        else:
            single_input = False

        state = self.state_tower(batch_state_embeddings)
        values = self.value_head(state)

        if single_input:
            values = values.squeeze(0)
        return values


class RouteAgent:
    """
    PPO agent for the static router (no action masking).

    The buffer stores 8-tuples:
        (state, action, reward, log_prob, value, done, local_memory_table, his_index)
    """

    def __init__(self, embed_dim=384, candidate_embed_dim=768, hidden_dim=16,
                 lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4,
                 entropy_coef=0.0, buffer_size=7, device=None,
                 networks=None, gnn_backbone="dblayer", **kwargs):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef
        self.buffer_size = buffer_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize or accept pre-built networks
        if networks:
            self.policy = networks["policy"]
            self.value = networks["value"]
            self.policy_old = networks["policy_old"]
        else:
            self.policy = PolicyNetwork(
                embed_dim, candidate_embed_dim, hidden_dim,
                device=self.device, gnn_backbone=gnn_backbone
            ).to(self.device)
            self.value = ValueNetwork(embed_dim, hidden_dim).to(self.device)
            self.policy_old = PolicyNetwork(
                embed_dim, candidate_embed_dim, hidden_dim,
                device=self.device, gnn_backbone=gnn_backbone
            ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        # Optimizers (value network uses 2x learning rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=2 * lr)

        # Fixed-length experience buffer (oldest entries are dropped automatically)
        self.buffer = deque(maxlen=self.buffer_size)

    def select_action(self, state_embeddings, local_tables, his_tables, his_index, greedy=False):
        """
        Select an action using the old policy (no action masking).

        Args:
            state_embeddings: State embedding tensor(s).
            local_tables: Local episode tables.
            his_tables: Historical memory tables.
            his_index: Integer role index.
            greedy: If True, pick the highest-probability action.

        Returns:
            (action, log_prob, value, entropy) as Python scalars.
        """
        with torch.no_grad():
            action_probs = self.policy_old(state_embeddings, local_tables, his_tables, his_index)
            value = self.value(state_embeddings)

        dist = Categorical(action_probs)
        action = torch.argmax(action_probs, dim=-1) if greedy else dist.sample()
        return (
            action.cpu().item(),
            dist.log_prob(action).cpu().item(),
            value.cpu().item(),
            dist.entropy().cpu().item(),
        )

    def store_transition(self, state_embeddings, action, reward, log_prob, value, done,
                         local_memory_table, his_index):
        """Store a transition (no action mask)."""
        if isinstance(state_embeddings, torch.Tensor):
            state_embeddings = state_embeddings.to(self.device)
        elif isinstance(state_embeddings, list):
            state_embeddings = [s.to(self.device) for s in state_embeddings]
        self.buffer.append((
            state_embeddings, action, reward, log_prob, value, done,
            local_memory_table, his_index,
        ))

    def compute_gae(self, rewards, values, dones, next_value, lambda_gae=0.95):
        """Compute Generalized Advantage Estimation."""
        advantages, gae = [], 0
        values = values + [next_value]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * lambda_gae * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def is_buffer_full(self):
        """Check whether the experience buffer has reached capacity."""
        return len(self.buffer) >= self.buffer_size

    def update(self, global_query_id_embed, global_LLM_role_id_embed, his_memory_table, batch_size=32):
        """
        Perform PPO update using buffered transitions.

        Args:
            global_query_id_embed: Global query embedding table.
            global_LLM_role_id_embed: Global LLM-role embedding table.
            his_memory_table: MemoryTable object providing historical context.
            batch_size: Mini-batch size for gradient updates.

        Returns:
            (avg_policy_loss, avg_value_loss, avg_entropy)
        """
        if not self.is_buffer_full():
            return 0, 0, 0

        # Unpack buffer
        states, actions, rewards, log_probs, values, dones, local_mems, his_idxs = zip(*self.buffer)
        states = list(states)
        actions = torch.LongTensor(actions).to(self.device)
        old_lp = torch.FloatTensor(log_probs).to(self.device)

        # Compute GAE advantages and returns
        advantages = self.compute_gae(list(rewards), list(values), list(dones), 0)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Build historical tables for the update
        his_tables = {
            "memory_table": his_memory_table.get_table(),
            "query_table": global_query_id_embed,
            "LLM_table": global_LLM_role_id_embed,
        }

        ds = len(states)
        indices = list(range(ds))
        tp, tv, te, nb = 0, 0, 0, 0

        for _ in range(self.k_epochs):
            random.shuffle(indices)
            for s in range(0, ds, batch_size):
                bi = indices[s:min(s + batch_size, ds)]

                # Forward pass for each sample in the mini-batch
                bap, bsv = [], []
                for i in bi:
                    lt = {
                        "memory_table": local_mems[i],
                        "query_table": global_query_id_embed,
                        "LLM_table": global_LLM_role_id_embed,
                    }
                    bap.append(self.policy([states[i]], lt, his_tables, his_idxs[i]).squeeze(0))
                    bsv.append(self.value(states[i]).squeeze())

                bap = torch.stack(bap)
                bsv = torch.stack(bsv)

                # Compute new log-probs and entropy
                nlp, ent = [], []
                for j, idx in enumerate(bi):
                    d = Categorical(bap[j])
                    nlp.append(d.log_prob(actions[idx]))
                    ent.append(d.entropy())
                nlp = torch.stack(nlp)
                ent = torch.stack(ent)

                # PPO clipped surrogate loss
                ratios = torch.exp(nlp - old_lp[bi])
                adv = advantages[bi]
                pl = -torch.min(
                    ratios * adv,
                    torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv,
                ).mean()
                pl = pl - self.entropy_coef * ent.mean()

                # Value loss
                vl = F.mse_loss(bsv, returns[bi])

                # Gradient steps
                self.policy_optimizer.zero_grad()
                pl.backward()
                self.policy_optimizer.step()

                self.value_optimizer.zero_grad()
                vl.backward()
                self.value_optimizer.step()

                tp += pl.item()
                tv += vl.item()
                te += ent.mean().item()
                nb += 1

        # Sync old policy with updated policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        return tp / max(nb, 1), tv / max(nb, 1), te / max(nb, 1)

    def save_model(self, filepath):
        """Save policy, value, and optimizer state dicts."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
        }, filepath)

    def load_model(self, filepath):
        """Load policy, value, and optimizer state dicts."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.value.load_state_dict(ckpt["value_state_dict"])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(ckpt["value_optimizer_state_dict"])

    def eval(self):
        """Switch all networks to evaluation mode."""
        self.policy.eval()
        self.policy_old.eval()
        self.value.eval()

    def train(self):
        """Switch all networks to training mode."""
        self.policy.train()
        self.policy_old.train()
        self.value.train()
