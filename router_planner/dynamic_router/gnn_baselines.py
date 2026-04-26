"""
GNN-based alternatives to the DBLayer for table embedding.

Provides HomoGCNDBLayer (homogeneous GCN) and HeteroGCNDBLayer (heterogeneous GCN)
that can be used as drop-in replacements for DBLayer in the PolicyNetwork.
These process memory tables by constructing query-LLM interaction graphs.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv
from .graph_builder import build_agentic_graph_pyg, build_agentic_hetero_graph_pyg


class HomoGCNDBLayer(nn.Module):
    """
    Homogeneous GCN layer that builds a unified graph from memory tables
    and produces node embeddings via 2-layer GCN message passing.

    Falls back to simple linear projection when memory is empty.
    """

    def __init__(self, table_dims: dict, dim_out: int, skip_ratio: bool = True, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim_out = dim_out
        self.conv1 = None  # Initialized lazily based on actual input dimension
        self.conv2 = None
        self.hidden_channels = max(128, dim_out)

    def forward(self, tables: dict, reduce: str = "mean"):
        memory = tables.get("memory_table", {"query.id": []})

        # Fallback for empty memory: use linear projection
        if not memory or len(memory.get("query.id", [])) == 0:
            result = {}
            for t_name, td in tables.items():
                if not td or len(list(td.values())[0]) == 0:
                    result[t_name] = torch.empty(0, self.dim_out, device=self.device)
                elif t_name == "LLM_table":
                    embs = [e.to(self.device) if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32, device=self.device) for e in td["embedding"]]
                    feat = torch.stack(embs)
                    if not hasattr(self, "fallback_linear_llm"):
                        self.fallback_linear_llm = nn.Linear(feat.size(1), self.dim_out).to(self.device)
                    result[t_name] = self.fallback_linear_llm(feat)
                elif t_name == "query_table":
                    embs = [e.to(self.device) if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32, device=self.device) for e in td["query_embedding"]]
                    feat = torch.stack(embs)
                    if not hasattr(self, "fallback_linear_query"):
                        self.fallback_linear_query = nn.Linear(feat.size(1), self.dim_out).to(self.device)
                    result[t_name] = self.fallback_linear_query(feat)
                else:
                    result[t_name] = torch.empty(0, self.dim_out, device=self.device)
            return result

        # Build homogeneous graph and run GCN
        graph_data, _ = build_agentic_graph_pyg(memory, device=self.device)
        x, edge_index = graph_data.x.to(self.device), graph_data.edge_index.to(self.device)

        if self.conv1 is None:
            self.conv1 = GCNConv(x.size(1), self.hidden_channels).to(self.device)
            self.conv2 = GCNConv(self.hidden_channels, self.dim_out).to(self.device)

        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)

        # Map node embeddings back to table format
        meta = graph_data.node_meta["node_idx_to_info"]
        result = {}
        for t_name in tables:
            if t_name == "query_table":
                idxs = [i for i, m in enumerate(meta) if m["type"] == "query"]
                result[t_name] = x[idxs] if idxs else torch.empty(0, self.dim_out, device=self.device)
            elif t_name == "LLM_table":
                idxs = [i for i, m in enumerate(meta) if m["type"] == "llm"]
                result[t_name] = x[idxs] if idxs else torch.empty(0, self.dim_out, device=self.device)
            elif t_name == "memory_table":
                result[t_name] = x
            else:
                result[t_name] = torch.empty(0, self.dim_out, device=self.device)
        return result


class HeteroGCNDBLayer(nn.Module):
    """
    Heterogeneous GCN layer with separate node types for queries and LLMs.

    Uses SAGEConv for each edge type in a 2-layer architecture.
    Falls back to linear projection when memory is empty.
    """

    def __init__(self, table_dims: dict, dim_out: int, skip_ratio: bool = True, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dim_out = dim_out
        hidden = max(128, dim_out)
        edge_types = [("query", "to", "llm"), ("llm", "to", "query"), ("query", "rev_to", "llm"), ("llm", "rev_to", "query")]
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden) for et in edge_types}).to(self.device)
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), dim_out) for et in edge_types}).to(self.device)

    def forward(self, tables: dict, reduce: str = "mean"):
        memory = tables.get("memory_table", {"query.id": []})

        if not memory or len(memory.get("query.id", [])) == 0:
            result = {}
            for t_name, td in tables.items():
                if not td or len(list(td.values())[0]) == 0:
                    result[t_name] = torch.empty(0, self.dim_out, device=self.device)
                elif t_name == "LLM_table":
                    embs = [e.to(self.device) if isinstance(e, torch.Tensor) else torch.tensor(e, dtype=torch.float32, device=self.device) for e in td["embedding"]]
                    feat = torch.stack(embs)
                    if not hasattr(self, "fallback_linear"):
                        self.fallback_linear = nn.Linear(feat.size(1), self.dim_out).to(self.device)
                    result[t_name] = self.fallback_linear(feat)
                else:
                    result[t_name] = torch.empty(0, self.dim_out, device=self.device)
            return result

        hetero = build_agentic_hetero_graph_pyg(memory, device=self.device)
        x_dict = {k: v.to(self.device) for k, v in hetero.x_dict.items()}
        ei_dict = {k: v.to(self.device) for k, v in hetero.edge_index_dict.items()}

        x_dict = {k: v.relu() for k, v in self.conv1(x_dict, ei_dict).items()}
        x_dict = self.conv2(x_dict, ei_dict)

        result = {}
        for t_name in tables:
            if t_name == "query_table":
                result[t_name] = x_dict.get("query", torch.empty(0, self.dim_out, device=self.device))
            elif t_name == "LLM_table":
                result[t_name] = x_dict.get("llm", torch.empty(0, self.dim_out, device=self.device))
            elif t_name == "memory_table":
                result[t_name] = x_dict.get("query", torch.empty(0, self.dim_out, device=self.device))
            else:
                result[t_name] = torch.empty(0, self.dim_out, device=self.device)
        return result
