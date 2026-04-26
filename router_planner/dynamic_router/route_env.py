"""
Dynamic routing environment for the RouterPlanner project.

Implements a tree-structured query processing environment that supports:
- Direct execution (single LLM call)
- Single-level decomposition (planner -> sub-queries -> executor -> summarizer)
- Multi-level recursive decomposition (nested planning)
- Mixed decomposition (some sub-queries further decomposed)
- Force-planner-first capability
- Context propagation across sibling and parent queries
- Proper state transitions after execution

The environment exposes a step-based interface compatible with RL agents,
returning state embeddings, action masks, rewards, and memory tables.
"""

import json
import pandas as pd
import torch
import ast
import re
from transformers import GPT2Tokenizer
from ..shared.utils import model_prompting, EmbeddingModel
from ..shared.task_prompting import generate_task_query
from ..shared.response_eval import eval_perf
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


class Env_route:
    """
    Complete routing environment with improved support for:
    - direct execution
    - single-level decomposition
    - multi-level (recursive) decomposition
    - mixed decomposition (some sub-queries further decomposed)
    - force planner action capability
    - Context propagation for sub-queries
    - Proper state transitions after execution
    """

    def __init__(self, query_file: str, test_query_file: str,
                 llm_file: str, agent_file: str,
                 max_planner_calls: int = 5,
                 max_width: int = 3,
                 max_depth: int = 3,
                 device=None, emb_model=None):

        self.max_planner_calls = max_planner_calls
        self.max_width = max_width
        self.max_depth = max_depth

        self.device = device if device is not None else torch.device("cpu")
        self.embedding_model = emb_model if emb_model is not None else EmbeddingModel(device=self.device)

        # === Load datasets ===
        self.query_data = pd.read_csv(query_file)
        self.query_data["query.id"] = range(len(self.query_data))

        self.test_query_data = pd.read_csv(test_query_file)
        self.test_query_data["query.id"] = range(len(self.test_query_data))

        # Merge and encode task names uniformly across train and test
        combined = pd.concat([self.query_data, self.test_query_data], axis=0)
        combined["task_idx"] = pd.factorize(combined["task_name"])[0]

        # Split back into train and test
        self.query_data = combined.iloc[:len(self.query_data)].copy()
        self.test_query_data = combined.iloc[len(self.query_data):].copy()

        # === LLM meta ===
        with open(llm_file, "r") as f:
            self.llm_data = json.load(f)
        items = list(self.llm_data.items())
        # Preserve a sane slice if original intended to exclude some items
        self.llm_data = dict(items[:3] + items[4:-8]) if len(items) > 10 else dict(items)

        # === Agent roles meta ===
        with open(agent_file, "r") as f:
            self.agent_data = json.load(f)

        # === Build action space ===
        self.action_map = {}
        self.action_to_role = {}
        action_idx = 0

        for role in ["planner", "executor", "summarizer"]:
            for llm_name in self.llm_data.keys():
                self.action_map[action_idx] = {
                    "role": role,
                    "llm": llm_name,
                    "embedding": self._create_role_llm_embedding(role, llm_name)
                }
                self.action_to_role[action_idx] = role
                action_idx += 1

        self.num_actions = len(self.action_map)

        # === Create role-specific embeddings ===
        self._create_llm_role_embeddings()

        # === Run mode ===
        self.test_mode = False
        self.current_test_index = 0

        # === Runtime variables ===
        self.planner_call_count = 0
        self.total_cost = 0.0

        # Tree structure for query processing
        self.root_node = None
        self.current_node = None
        self.execution_path = []  # Track execution path for debugging

        # Current state
        self.state = None
        self.current_agent_llm_embedding = None
        self.action_mask = None

        # Memory tables
        self.local_memory_table = None
        self.table_query_id = -1
        self.current_query = None
        self.query_id_embed = None

        # Force planner capability
        self.force_planner_first = False

    def _create_role_llm_embedding(self, role: str, llm_name: str) -> torch.Tensor:
        """Create embedding for role-LLM combination."""
        role_emb = torch.tensor(self.agent_data[role]["embedding"], dtype=torch.float32).to(self.device)
        llm_emb = torch.tensor(self.llm_data[llm_name]["embedding"], dtype=torch.float32).to(self.device)
        return torch.cat([role_emb.reshape(-1), llm_emb.reshape(-1)], dim=0)

    def _create_llm_role_embeddings(self):
        """Create compatibility embeddings for upstream network."""
        llm_role_ids, llm_role_embs = [], []

        for action_idx, action_info in self.action_map.items():
            role = action_info["role"]
            llm_role_ids.append(f"{role}_{action_idx}")
            llm_role_embs.append(action_info["embedding"].detach().clone().to(self.device))

        self.LLM_role_id_embed = {"LLM_role.id": llm_role_ids, "embedding": llm_role_embs}

        # Also create agent_llm_embedding for compatibility
        self.agent_llm_embedding = {}
        for role in ["planner", "executor", "summarizer"]:
            role_embeddings = []
            for action_idx, action_info in self.action_map.items():
                if action_info["role"] == role:
                    role_embeddings.append(action_info["embedding"].detach().clone())
            if role_embeddings:
                self.agent_llm_embedding[role] = {
                    "embedding": torch.stack(role_embeddings, dim=0).to(self.device)
                }

    def _create_node(self, query: str, parent=None, depth: int = 0):
        """Create a new node in the query tree."""
        return {
            "query": query,
            "parent": parent,
            "depth": depth,
            "children": [],
            "status": "pending",  # pending, planning, executing, summarizing, completed
            "planned": False,
            "executed": False,
            "response": None,
            "summary": None,
            "current_child_idx": 0,
            "sub_queries": [],
            "child_responses": [],
            "sibling_context": []  # Store responses from previous siblings
        }

    def _get_full_context(self, node):
        """Get full context including parent queries and sibling responses."""
        context = {
            "original_query": self._get_root_query(node),
            "parent_queries": self._get_parent_queries(node),
            "sibling_responses": self._get_current_sibling_responses(node)
        }
        return context

    def _get_root_query(self, node):
        """Get the original root query."""
        current = node
        while current.get("parent") is not None:
            current = current["parent"]
        return current["query"]

    def _get_parent_queries(self, node):
        """Get all parent queries up to root."""
        queries = []
        current = node.get("parent")
        while current is not None:
            queries.append(current["query"])
            current = current.get("parent")
        return list(reversed(queries))  # Root to immediate parent order

    def _get_current_sibling_responses(self, node):
        """Get the latest sibling node responses."""
        if node.get("parent") is None:
            return []

        parent = node["parent"]
        responses = []

        for child in parent["children"]:
            if child == node:
                break  # Only get responses from preceding siblings
            if child.get("response"):
                responses.append({
                    "query": child["query"],
                    "response": child["response"]
                })

        return responses

    def _update_sibling_contexts_after_completion(self, completed_node):
        """Update context for all subsequent sibling nodes after a node completes."""
        if not completed_node.get("parent"):
            return

        parent = completed_node["parent"]
        try:
            node_idx = parent["children"].index(completed_node)
        except ValueError:
            return

        # Update context for all subsequent siblings
        for i in range(node_idx + 1, len(parent["children"])):
            sibling = parent["children"][i]
            # Add the completed node's response to the sibling's context
            if completed_node.get("response"):
                sibling["sibling_context"].append({
                    "query": completed_node["query"],
                    "response": completed_node["response"]
                })

    def _find_next_node_to_process(self):
        """Find the next node that needs processing using depth-first search."""
        if self.root_node is None:
            return None

        def dfs(node):
            # If the current node is pending, return it directly
            if node["status"] == "pending":
                return node

            # If node has children, check them
            for child in node["children"]:
                result = dfs(child)
                if result:
                    return result

            # If node has children and all are completed, and node itself
            # is in planning state and has no summary yet, it needs summarization
            if (node["children"] and
                    all(child["status"] == "completed" for child in node["children"]) and
                    node["status"] == "planning" and
                    not node.get("summary")):
                return node

            return None

        return dfs(self.root_node)

    def _get_node_state(self, node):
        """Get the state vector corresponding to the given node."""
        if node["status"] == "pending":
            # Pending node: use query embedding
            return torch.tensor(
                self.embedding_model.get_embedding(node["query"]),
                dtype=torch.float32
            ).to(self.device)
        elif node["status"] == "planning" and node["children"]:
            # Node needing summarizer: use child responses embedding
            child_responses = [child["response"] for child in node["children"]
                               if child.get("response")]
            if child_responses:
                context = "\n".join(child_responses)
                return torch.tensor(
                    self.embedding_model.get_embedding(context),
                    dtype=torch.float32
                ).to(self.device)

        # Default: return current state
        return self.state

    def _compute_action_mask(self, is_first_step: bool = False) -> np.ndarray:
        """
        Compute mask for valid actions based on current state.

        Args:
            is_first_step: Whether this is the first step of the episode.
        """
        mask = np.zeros(self.num_actions, dtype=np.float32)

        if self.current_node is None:
            # Fallback: allow some executor actions
            for action_idx, action_info in self.action_map.items():
                if action_info["role"] == "executor":
                    mask[action_idx] = 1.0
            return mask

        # Check if we need to force planner action on first step
        if is_first_step and self.force_planner_first:
            # Only allow planner actions if we haven't exceeded limits
            can_use_planner = (self.planner_call_count < self.max_planner_calls and
                               self.current_node["depth"] < self.max_depth)

            if can_use_planner:
                for action_idx, action_info in self.action_map.items():
                    if action_info["role"] == "planner":
                        mask[action_idx] = 1.0

                # If we have valid planner actions, return early
                if mask.sum() > 0:
                    return mask

        can_use_planner = (self.planner_call_count < self.max_planner_calls and
                           self.current_node["depth"] < self.max_depth)

        node = self.current_node
        # If node is pending, it can either be planned (if allowed) or executed directly
        if node["status"] == "pending":
            for action_idx, action_info in self.action_map.items():
                if action_info["role"] == "executor":
                    mask[action_idx] = 1.0
                elif action_info["role"] == "planner" and can_use_planner:
                    mask[action_idx] = 1.0

        # Handle planning state precisely
        elif node["status"] == "planning":
            # If all children are completed, allow summarizer
            if node["children"] and all(child["status"] == "completed" for child in node["children"]):
                for action_idx, action_info in self.action_map.items():
                    if action_info["role"] == "summarizer":
                        mask[action_idx] = 1.0
            # Otherwise, do not allow any action on a planning-state node;
            # _find_next_node_to_process should locate the correct pending child

        # If node is executing and has children: allow summarizer if children all done
        elif node["status"] == "executing":
            if node["children"] and all(child["status"] == "completed" for child in node["children"]):
                for action_idx, action_info in self.action_map.items():
                    if action_info["role"] == "summarizer":
                        mask[action_idx] = 1.0
            # If no children (leaf) and not executed, allow executor
            elif not node["children"] and not node["executed"]:
                for action_idx, action_info in self.action_map.items():
                    if action_info["role"] == "executor":
                        mask[action_idx] = 1.0

        # If node is summarizing (root summarized and needs final executor)
        elif node["status"] == "summarizing":
            if node["depth"] == 0:
                for action_idx, action_info in self.action_map.items():
                    if action_info["role"] == "executor":
                        mask[action_idx] = 1.0

        # Fallback: ensure at least one executor action is enabled
        if mask.sum() == 0:
            for action_idx, action_info in self.action_map.items():
                if action_info["role"] == "executor":
                    mask[action_idx] = 1.0

        return mask

    def set_force_planner_first(self, force: bool):
        """Set whether to force planner action on first step."""
        self.force_planner_first = force

    def reset(self):
        """Reset environment for new episode."""
        # Sample query
        if self.test_mode:
            if self.current_test_index >= len(self.test_query_data):
                self.current_test_index = 0
            row = self.test_query_data.iloc[self.current_test_index].to_dict()
            self.table_query_id = int(row["query.id"])
            self.current_test_index += 1
            self.his_index = [self.table_query_id, row["task_idx"]]
        else:
            row = self.query_data.sample(1).iloc[0].to_dict()
            self.table_query_id = int(row["query.id"])
            self.his_index = [self.table_query_id, row["task_idx"]]

        self.current_query = row

        # Full query embeddings
        src = self.test_query_data if self.test_mode else self.query_data
        self.query_id_embed = {
            "query.id": src["query.id"].tolist(),
            "query_embedding": [self.to_tensor(x).to(self.device) for x in src["query_embedding"]]
        }

        # Initialize tree with root node
        self.root_node = self._create_node(self.current_query["query"], parent=None, depth=0)
        self.current_node = self.root_node
        self.execution_path = []

        self.planner_call_count = 0
        self.total_cost = 0.0

        # Initial state is the query embedding
        self.state = self.to_tensor(self.current_query["query_embedding"]).to(self.device)

        # Compute initial action mask (this is the first step)
        self.action_mask = self._compute_action_mask(is_first_step=True)

        # Set current embeddings to all possible actions
        all_embeddings = []
        for action_info in self.action_map.values():
            all_embeddings.append(action_info["embedding"])
        self.current_agent_llm_embedding = torch.stack(all_embeddings, dim=0)

        # Initialize memory table
        self.local_memory_table = {
            "query.id": [],
            "LLM_role.id": [],
            "score": [],
            "cost": [],
            "n_i": [],
            "n_o": []
        }

        return (self.state, self.current_agent_llm_embedding,
                self.query_id_embed, self.LLM_role_id_embed,
                self.local_memory_table, self.his_index, self.action_mask)

    def next_step(self, action: int):
        """Execute one step with the selected action."""

        # Validate action against mask
        if self.action_mask[action] == 0:
            state = None if self.current_node is None else self.current_node.get("status")
            raise ValueError(f"Invalid action {action} for node_state={state}. Mask={self.action_mask}")

        action_info = self.action_map[action]
        role = action_info["role"]
        llm_name = action_info["llm"]

        self.execution_path.append(f"{role}_{llm_name}")

        # Route to appropriate handler based on role
        if role == "planner":
            return self._step_planner(action, llm_name)
        elif role == "executor":
            return self._step_executor(action, llm_name)
        elif role == "summarizer":
            return self._step_summarizer(action, llm_name)
        else:
            raise ValueError(f"Unknown role: {role}")

    def _step_planner(self, action: int, llm_name: str):
        """Handle planner action - decompose query into sub-queries with context."""

        if self.current_node is None or self.current_node["status"] not in ("pending",):
            raise ValueError("Invalid state for planning; node must be pending to plan")

        self.planner_call_count += 1

        # Get full context for planning
        context = self._get_full_context(self.current_node)

        # Call planner with context
        query = self.current_node["query"]
        sub_queries_str, prompt_used = self.planner(query, llm_name, context)
        step_cost = self.compute_cost(llm_name, prompt_used, sub_queries_str)
        self.total_cost += step_cost

        # Parse sub-queries (respect max_width)
        sub_queries = self._parse_dynamic_queries(sub_queries_str, fallback=query)

        # Update node status to planning
        self.current_node["status"] = "planning"
        self.current_node["planned"] = True
        self.current_node["sub_queries"] = sub_queries

        # Create child nodes with sibling context
        for i, sub_query in enumerate(sub_queries):
            child_node = self._create_node(
                query=sub_query,
                parent=self.current_node,
                depth=self.current_node["depth"] + 1
            )
            # Set sibling context for this child (responses from previous siblings)
            child_node["sibling_context"] = []
            for j in range(i):
                if self.current_node["children"][j].get("response"):
                    child_node["sibling_context"].append({
                        "query": self.current_node["children"][j]["query"],
                        "response": self.current_node["children"][j]["response"]
                    })
            self.current_node["children"].append(child_node)

        # Log to memory
        out_emb = torch.tensor(self.embedding_model.get_embedding(sub_queries_str), dtype=torch.float32).to(self.device)
        self.append_local_memory({
            "query.id": self.table_query_id,
            "LLM_role.id": f"planner_{action}",
            "score": 0.0,
            "cost": step_cost / 1000.0,
            "n_i": self.state,
            "n_o": out_emb
        })

        # Move to first child
        if self.current_node["children"]:
            self.current_node = self.current_node["children"][0]
            self.state = torch.tensor(
                self.embedding_model.get_embedding(self.current_node["query"]),
                dtype=torch.float32
            ).to(self.device)

        # Update action mask and embeddings (not first step anymore)
        self.action_mask = self._compute_action_mask(is_first_step=False)
        all_embeddings = [a["embedding"] for a in self.action_map.values()]
        self.current_agent_llm_embedding = torch.stack(all_embeddings, dim=0)

        done = False
        reward = 0.0

        return (self.state, done, step_cost, reward, self.current_agent_llm_embedding,
                self.local_memory_table, self.his_index, self.action_mask)

    def _step_executor(self, action: int, llm_name: str):
        """Handle executor action - generate response for query with context."""

        if self.current_node is None:
            raise ValueError("No current node for execution")

        node = self.current_node

        # Get full context for execution
        full_context = self._get_full_context(node)

        # Determine execution context
        is_final = False
        context = ""

        # Check if this is final execution at root with context
        if node["depth"] == 0 and node.get("summary"):
            # Final execution with summary context
            is_final = True
            context = node["summary"]
        elif node["depth"] == 0 and not node["planned"]:
            # Direct execution at root without planning
            is_final = True
            context = ""

        # Execute query with full context
        query = node["query"]
        if is_final:
            response, prompt_used = self.executor(query, llm_name, if_final=True,
                                                  context=context, full_context=full_context)
            # Calculate reward for final answer
            reward = eval_perf(
                metric=self.current_query['metric'],
                prediction=response,
                ground_truth=self.current_query['gt'],
                task_name=self.current_query['task_name'],
                task_id=self.current_query['task_id']
            )
            done = True
        else:
            response, prompt_used = self.executor(query, llm_name, if_final=False,
                                                  full_context=full_context)
            reward = 0.0
            done = False

        step_cost = self.compute_cost(llm_name, prompt_used, response)
        self.total_cost += step_cost

        # Update node - mark as completed after execution
        node["response"] = response
        node["executed"] = True
        node["status"] = "completed"

        # Log to memory
        out_emb = torch.tensor(self.embedding_model.get_embedding(response),
                               dtype=torch.float32).to(self.device)
        self.append_local_memory({
            "query.id": self.table_query_id,
            "LLM_role.id": f"executor_{action}",
            "score": reward,
            "cost": step_cost / 1000.0,
            "n_i": self.state,
            "n_o": out_emb
        })

        self.state = out_emb

        # Immediately update context for all subsequent sibling nodes
        self._update_sibling_contexts_after_completion(node)

        if not done:
            # Move to the next pending node
            self.current_node = self._find_next_node_to_process()
            if self.current_node is None:
                done = True
            else:
                # Update state to the new current node
                self.state = self._get_node_state(self.current_node)

        # Update action mask and embeddings
        self.action_mask = self._compute_action_mask(is_first_step=False)
        all_embeddings = [a["embedding"] for a in self.action_map.values()]
        self.current_agent_llm_embedding = torch.stack(all_embeddings, dim=0)

        return (self.state, done, step_cost, reward, self.current_agent_llm_embedding,
                self.local_memory_table, self.his_index, self.action_mask)

    def _step_summarizer(self, action: int, llm_name: str):
        """Handle summarizer action - summarize collected responses."""

        if self.current_node is None:
            raise ValueError("No current node for summarization")

        node = self.current_node

        # Ensure there are child responses to summarize
        child_responses = [child["response"] for child in node["children"]
                           if child.get("response")]
        if not child_responses:
            raise ValueError("No child responses to summarize")

        context = "\n".join(child_responses)
        summary, prompt_used = self.summarizer(context, llm_name, parent_query=node["query"])

        step_cost = self.compute_cost(llm_name, prompt_used, summary)
        self.total_cost += step_cost

        # Update node
        node["summary"] = summary

        # For non-root nodes, summary becomes the node's response
        if node["depth"] == 0:
            # Root node: after summarization, mark as waiting for final execution
            node["status"] = "summarizing"
        else:
            # Non-root node: summary becomes the response
            node["status"] = "completed"
            node["response"] = summary
            node["executed"] = True

        # Log to memory
        out_emb = torch.tensor(self.embedding_model.get_embedding(summary),
                               dtype=torch.float32).to(self.device)
        self.append_local_memory({
            "query.id": self.table_query_id,
            "LLM_role.id": f"summarizer_{action}",
            "score": 0.0,
            "cost": step_cost / 1000.0,
            "n_i": self.state,
            "n_o": out_emb
        })

        self.state = out_emb

        # Update sibling context immediately for non-root nodes
        if node["depth"] != 0:
            self._update_sibling_contexts_after_completion(node)

        # Find the next node to process
        done = False
        if node["depth"] == 0:
            # Root node summarization complete; stay at root waiting for final executor
            pass
        else:
            # Non-root node: find next pending node
            self.current_node = self._find_next_node_to_process()
            if self.current_node is None:
                done = True
            else:
                self.state = self._get_node_state(self.current_node)

        # Update action mask and embeddings
        self.action_mask = self._compute_action_mask(is_first_step=False)
        all_embeddings = [a["embedding"] for a in self.action_map.values()]
        self.current_agent_llm_embedding = torch.stack(all_embeddings, dim=0)

        return (self.state, done, step_cost, 0.0, self.current_agent_llm_embedding,
                self.local_memory_table, self.his_index, self.action_mask)

    def _parse_dynamic_queries(self, text: str, fallback: str) -> list:
        """
        Parse LLM output to get sub-queries with simple even grouping when exceeding max_width.
        If LLM generates more than max_width queries, group them evenly.
        """
        lines = [x.strip() for x in text.split("\n") if x.strip()]
        if not lines:
            return [fallback]

        # If within limit, return as-is
        if len(lines) <= self.max_width:
            return lines

        # Calculate even distribution
        num_queries = len(lines)
        base_size = num_queries // self.max_width
        extra = num_queries % self.max_width

        # Create groups
        result = []
        start = 0

        for i in range(self.max_width):
            # First 'extra' groups get one more query
            group_size = base_size + (1 if i < extra else 0)
            end = start + group_size

            group_queries = lines[start:end]
            if len(group_queries) == 1:
                result.append(group_queries[0])
            else:
                # Simple combination with numbering
                combined = "\n".join(f"{j + 1}. {q}" for j, q in enumerate(group_queries))
                result.append(combined)

            start = end

        return result

    def get_tree_depth(self):
        """Get the current tree depth for debugging."""
        if self.root_node is None:
            return 0

        def get_max_depth(node):
            if not node["children"]:
                return node["depth"]
            return max(get_max_depth(child) for child in node["children"])

        return get_max_depth(self.root_node)

    def get_tree_size(self):
        """Get the total number of nodes in the tree."""
        if self.root_node is None:
            return 0

        def count_nodes(node):
            count = 1
            for child in node["children"]:
                count += count_nodes(child)
            return count

        return count_nodes(self.root_node)

    # -------------------- Helper methods --------------------
    def to_tensor(self, s: str) -> torch.Tensor:
        """Parse a string representation of a tensor into a torch.Tensor."""
        s = s.strip()
        if s.startswith("tensor("):
            s = s[len("tensor("):].rstrip(")")
        s = re.sub(r"device='[^']*'", "", s)
        s = re.sub(r"dtype=\w+\.\w+", "", s)
        s = s.replace(", ,", ",").strip().rstrip(",")
        data = ast.literal_eval(s)
        return torch.tensor(data, dtype=torch.float32)

    def append_local_memory(self, row: dict):
        """Append a record to the local memory table."""
        for k in ["query.id", "LLM_role.id", "score", "cost", "n_i", "n_o"]:
            self.local_memory_table[k].append(row[k])

    def enable_test_mode(self):
        """Enable test mode for sequential evaluation."""
        self.test_mode = True
        self.current_test_index = 0

    def disable_test_mode(self):
        """Disable test mode and return to training mode."""
        self.test_mode = False
        self.current_test_index = 0

    # -------------------- LLM calls --------------------
    def compute_cost(self, llm_name: str, input_text: str, output_text: str) -> float:
        """Compute the monetary cost of an LLM call based on token counts."""
        in_tokens = len(tokenizer(input_text)["input_ids"])
        out_tokens = len(tokenizer(output_text)["input_ids"])
        in_price = self.llm_data[llm_name]["input_price"]
        out_price = self.llm_data[llm_name]["output_price"]
        return in_tokens * in_price + out_tokens * out_price

    def planner(self, query: str, llm_name: str = "mistral-nemo-12b-instruct", context: dict = None):
        """Planner with dynamic number of sub-queries and full context."""
        # Build context string
        context_str = ""
        if context:
            if context.get("original_query"):
                context_str += f"Original query: {context['original_query']}\n"
            if context.get("parent_queries"):
                for i, pq in enumerate(context["parent_queries"]):
                    context_str += f"Parent query level {i + 1}: {pq}\n"
            if context.get("sibling_responses"):
                context_str += "\nPrevious sub-query responses:\n"
                for sr in context["sibling_responses"]:
                    context_str += f"Q: {sr['query']}\nA: {sr['response']}\n\n"

        prompt = f"""You are a query decomposition assistant. Your task is to decompose the user's query into atomic and independent sub-queries.

{context_str if context_str else ""}

Current query to decompose: {query}

Instructions:
- Determine the optimal number of sub-queries needed (minimum 1, maximum 3)
- Keep each sub-query self-contained and non-overlapping
- Use fewer sub-queries for simple questions, more for complex ones
- Consider the context and previous responses when decomposing
- If previous siblings have answered part of the question, focus on the remaining parts
- Output one sub-query per line, no numbering or extra words
- You MUST output at least 1 and at most 3 sub-queries"""

        resp = model_prompting(self.llm_data[llm_name]["model"], prompt)
        return resp, prompt

    def executor(self, query: str, llm_name: str = "mistral-nemo-12b-instruct",
                 if_final: bool = False, context: str = "", full_context: dict = None) -> tuple:
        """Execute query with full context including parent queries and sibling responses."""

        # Build full context string
        context_str = ""
        if full_context:
            if full_context.get("original_query"):
                context_str += f"Original query: {full_context['original_query']}\n"
            if full_context.get("parent_queries"):
                context_str += "Query hierarchy:\n"
                for i, pq in enumerate(full_context["parent_queries"]):
                    context_str += f"  Level {i + 1}: {pq}\n"
            if full_context.get("sibling_responses"):
                context_str += "\nPrevious related responses:\n"
                for sr in full_context["sibling_responses"]:
                    context_str += f"Q: {sr['query']}\nA: {sr['response']}\n\n"

        if if_final:
            query = generate_task_query(task_name=self.current_query['task_name'], sample_data=self.current_query)
            if context:
                prompt = f"""You are a helpful assistant. Given the following context and the user's query, provide a direct, complete, and accurate answer.

Full context:
{context_str if context_str else ""}

Summary of sub-query responses:
{context}

User query:
{query}

Answer:"""
            else:
                prompt = f"""{context_str if context_str else ""}

{query}"""
        else:
            # Non-final execution with full context
            if context_str:
                prompt = f"""You are a helpful assistant. Given the following context, answer the specific sub-query. You must output according to the output format required in the query and do not output any additional information. Current sub-query to answer:
{query}

Context:
{context_str}

Answer:"""
            else:
                prompt = query

        response = model_prompting(llm_model=self.llm_data[llm_name]["model"], prompt=prompt)
        return response, prompt

    def summarizer(self, context: str, llm_name: str = "mistral-nemo-12b-instruct", parent_query: str = ""):
        """Summarizer that creates a comprehensive response for the parent query."""
        prompt = f"""You are a professional summarizer. Synthesize the following child answers into a comprehensive, coherent response that directly addresses the PARENT QUERY.

PARENT QUERY:
{parent_query}

CHILD ANSWERS:
{context}

Instructions:
- Create a complete answer to the parent query based on the child answers
- Maintain all important information from the child answers
- Ensure the response is coherent and well-structured
- This summary will be used as the complete response to the parent query

Summary:"""
        resp = model_prompting(self.llm_data[llm_name]["model"], prompt)
        return resp, prompt
