"""
Graph construction utilities for building PyG graphs from memory tables.

Provides builders for both homogeneous and heterogeneous graphs
representing query-LLM interaction histories, along with GCN model
classes for processing these graphs.

Graph Structure:
- Node types: 'query' (n_i / n_o embeddings) and 'llm' (role embeddings)
- Edge types: query -> llm (input), llm -> query (output), plus reverse edges
- Edge attributes: [cost, performance_score]
"""

import torch
from torch import nn
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import GCNConv, HeteroConv, SAGEConv
from typing import Dict, List, Any, Tuple


def build_agentic_graph_pyg(
    his_memory_table: Dict[str, List[Any]],
    llm_embedding_table: Dict[str, torch.Tensor] = None,
    device: torch.device = None,
) -> Tuple[Data, Dict[str, int]]:
    """
    Build an undirected homogeneous PyG graph from a memory table.

    Each memory entry creates: n_i_node -> llm_node -> n_o_node
    with reverse edges for undirected message passing.

    Args:
        his_memory_table: Memory table with keys: query.id, LLM_role.id,
            score, cost, n_i (List[Tensor]), n_o (List[Tensor]).
        llm_embedding_table: Optional pre-computed LLM embeddings.
        device: Target device.

    Returns:
        data: PyG Data object with node features, edges, and metadata.
        llm_role_to_node_idx: Mapping from LLM role ID to node index.
    """
    if device is None:
        device = torch.device("cpu")

    n_entries = len(his_memory_table["query.id"])
    emb_dim = his_memory_table["n_i"][0].shape[0]

    # Collect unique LLM roles (preserving order)
    llm_roles = []
    for role in his_memory_table["LLM_role.id"]:
        if role not in llm_roles:
            llm_roles.append(role)
    llm_role_to_idx = {r: i for i, r in enumerate(llm_roles)}

    # Prepare LLM node embeddings
    if llm_embedding_table is None:
        llm_embedding_table = {}
    llm_embs = []
    for r in llm_roles:
        if r in llm_embedding_table:
            e = llm_embedding_table[r].view(-1).to(device)
            if e.shape[0] != emb_dim:
                proj = nn.Linear(e.shape[0], emb_dim)
                with torch.no_grad():
                    e = proj(e.to(device))
            llm_embs.append(e)
        else:
            llm_embs.append(torch.randn(emb_dim, device=device))
    llm_embs = torch.stack(llm_embs, dim=0)

    node_feats, node_meta = [], []
    llm_role_to_node_idx = {}
    edge_src, edge_dst, edge_attrs = [], [], []

    for i in range(n_entries):
        # Create n_i (input query) node
        ni = his_memory_table["n_i"][i].to(device)
        ni_feat = torch.cat([ni, torch.tensor([0.0, -1.0], device=device)])
        node_feats.append(ni_feat)
        node_meta.append({"type": "query", "role": None, "entry_idx": i, "subtype": "n_i"})
        ni_idx = len(node_feats) - 1

        # Get or create LLM node
        role = his_memory_table["LLM_role.id"][i]
        if role not in llm_role_to_node_idx:
            role_idx = llm_role_to_idx[role]
            llm_feat = torch.cat([llm_embs[role_idx], torch.tensor([1.0, float(role_idx)], device=device)])
            node_feats.append(llm_feat)
            node_meta.append({"type": "llm", "role": role, "role_idx": role_idx})
            llm_role_to_node_idx[role] = len(node_feats) - 1
        llm_idx = llm_role_to_node_idx[role]

        # Create n_o (output query) node
        no = his_memory_table["n_o"][i].to(device)
        no_feat = torch.cat([no, torch.tensor([0.0, -1.0], device=device)])
        node_feats.append(no_feat)
        node_meta.append({"type": "query", "role": None, "entry_idx": i, "subtype": "n_o"})
        no_idx = len(node_feats) - 1

        # Build directed edges with cost/perf attributes
        attr = torch.tensor([float(his_memory_table["cost"][i]), float(his_memory_table["score"][i])], device=device)
        edge_src.extend([ni_idx, llm_idx]); edge_dst.extend([llm_idx, no_idx])
        edge_attrs.extend([attr, attr])

    # Add reverse edges for undirected graph
    full_src, full_dst, full_attr = [], [], []
    for s, d, a in zip(edge_src, edge_dst, edge_attrs):
        full_src.extend([s, d]); full_dst.extend([d, s]); full_attr.extend([a, a.clone()])

    x = torch.stack(node_feats, dim=0)
    edge_index = torch.tensor([full_src, full_dst], dtype=torch.long, device=device)
    edge_attr = torch.stack(full_attr, dim=0)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.node_meta = {"node_idx_to_info": node_meta, "llm_role_to_node_idx": llm_role_to_node_idx, "llm_roles": llm_roles, "emb_dim": emb_dim}
    return data, llm_role_to_node_idx


def build_agentic_hetero_graph_pyg(
    his_memory_table: Dict[str, List[Any]],
    llm_embedding_table: Dict[str, torch.Tensor] = None,
    device: torch.device = None,
) -> HeteroData:
    """
    Build a heterogeneous PyG graph with 'query' and 'llm' node types.

    Edge types: (query, to, llm), (llm, to, query), plus reverse edges.
    """
    if device is None:
        device = torch.device("cpu")

    n_entries = len(his_memory_table["query.id"])
    emb_dim = his_memory_table["n_i"][0].shape[0]

    llm_roles = []
    for role in his_memory_table["LLM_role.id"]:
        if role not in llm_roles:
            llm_roles.append(role)
    llm_role_to_idx = {r: i for i, r in enumerate(llm_roles)}

    if llm_embedding_table is None:
        llm_embedding_table = {}
    llm_embs = []
    for r in llm_roles:
        if r in llm_embedding_table and llm_embedding_table[r].shape[0] == emb_dim:
            llm_embs.append(llm_embedding_table[r].to(device))
        else:
            llm_embs.append(torch.randn(emb_dim, device=device))
    llm_embs = torch.stack(llm_embs, dim=0)

    hetero = HeteroData()
    query_feats = []
    q2l_edges, l2q_edges = [], []
    q2l_attr, l2q_attr = [], []
    qi = 0

    for i in range(n_entries):
        query_feats.append(his_memory_table["n_i"][i].to(device))
        ni_idx = qi; qi += 1
        llm_idx = llm_role_to_idx[his_memory_table["LLM_role.id"][i]]
        query_feats.append(his_memory_table["n_o"][i].to(device))
        no_idx = qi; qi += 1

        attr = torch.tensor([float(his_memory_table["cost"][i]), float(his_memory_table["score"][i])], device=device)
        q2l_edges.append([ni_idx, llm_idx]); q2l_attr.append(attr)
        l2q_edges.append([llm_idx, no_idx]); l2q_attr.append(attr)

    hetero["query"].x = torch.stack(query_feats, dim=0)
    hetero["llm"].x = llm_embs
    hetero[("query", "to", "llm")].edge_index = torch.tensor(q2l_edges, dtype=torch.long, device=device).t().contiguous()
    hetero[("query", "to", "llm")].edge_attr = torch.stack(q2l_attr)
    hetero[("llm", "to", "query")].edge_index = torch.tensor(l2q_edges, dtype=torch.long, device=device).t().contiguous()
    hetero[("llm", "to", "query")].edge_attr = torch.stack(l2q_attr)

    # Reverse edges for undirected message passing
    hetero[("llm", "rev_to", "query")].edge_index = hetero[("query", "to", "llm")].edge_index.flip(0)
    hetero[("llm", "rev_to", "query")].edge_attr = hetero[("query", "to", "llm")].edge_attr.clone()
    hetero[("query", "rev_to", "llm")].edge_index = hetero[("llm", "to", "query")].edge_index.flip(0)
    hetero[("query", "rev_to", "llm")].edge_attr = hetero[("llm", "to", "query")].edge_attr.clone()

    hetero.meta_info = {"llm_roles": llm_roles, "llm_role_to_idx": llm_role_to_idx}
    return hetero


# ============================================================================
# Standalone GCN Models (for direct graph processing)
# ============================================================================
class HomoGCN(torch.nn.Module):
    """2-layer GCN for homogeneous graphs."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        return self.conv2(x, data.edge_index)


class HeteroGCN(torch.nn.Module):
    """2-layer heterogeneous GCN with SAGEConv per edge type."""
    def __init__(self, metadata, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = HeteroConv({et: SAGEConv((-1, -1), hidden_channels) for et in metadata[1]})
        self.conv2 = HeteroConv({et: SAGEConv((-1, -1), out_channels) for et in metadata[1]})

    def forward(self, data):
        x_dict = {k: v.relu() for k, v in self.conv1(data.x_dict, data.edge_index_dict).items()}
        return self.conv2(x_dict, data.edge_index_dict)
