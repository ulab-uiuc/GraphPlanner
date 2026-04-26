"""
Routing environment for the Static Router.

Implements a DFS-based multi-agent workflow with fixed width and depth:
  - The root node (original query) is pushed onto the stack.
  - If the current node depth < max_depth-1: planner decomposes into child
    queries, then DFS recurses into each child.
  - If the current node depth == max_depth-1: planner generates leaf queries,
    executor processes each leaf, then summarizer aggregates results.
  - Child summaries propagate upward; when a parent has all child summaries,
    its summarizer runs.
  - The root summarizer produces a final summary, which is used by a final
    executor call to produce the answer and compute the reward.

Each call to ``next_step`` triggers exactly one LLM invocation.
"""

import json
import ast
import re

import pandas as pd
import torch
from transformers import GPT2Tokenizer

from ..shared.utils import model_prompting, EmbeddingModel
from ..shared.response_eval import eval_perf
from ..shared.task_prompting import generate_task_query

# Tokenizer used for cost estimation (token counting)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


class Env_route:
    """
    Static routing environment with DFS execution over arbitrary width/depth.

    Phases cycle through: planner -> executor -> summarizer -> final -> done.
    ``his_index`` is an integer selecting the current role
    (0=planner, 1=executor, 2=summarizer).
    """

    def __init__(self, query_file: str, test_query_file: str,
                 llm_file: str, agent_file: str,
                 width: int, depth: int,
                 device=None, emb_model=None):

        assert width >= 1 and depth >= 1
        self.width = int(width)
        self.depth = int(depth)

        self.device = device if device is not None else torch.device("cpu")
        self.embedding_model = emb_model if emb_model is not None else EmbeddingModel(device=self.device)

        # --- Load datasets ---
        self.query_data = pd.read_csv(query_file)
        self.query_data["query.id"] = range(len(self.query_data))

        self.test_query_data = pd.read_csv(test_query_file)
        self.test_query_data["query.id"] = range(len(self.test_query_data))

        # --- LLM metadata ---
        with open(llm_file, "r") as f:
            self.llm_data = json.load(f)
        # Exclude the last 5 entries (reserved/internal models)
        self.llm_data = dict(list(self.llm_data.items())[:-5])

        # --- Agent role metadata ---
        with open(agent_file, "r") as f:
            self.agent_data = json.load(f)

        # --- Action space: index -> LLM name + embedding ---
        self.action_map = {}
        for idx, name in enumerate(self.llm_data.keys()):
            self.action_map[idx] = {
                "llm": name,
                "embedding": torch.tensor(
                    self.llm_data[name]["embedding"], dtype=torch.float32
                ).to(self.device),
            }

        # --- Role x LLM concatenated embeddings (planner / executor / summarizer) ---
        self.agent_llm_embedding = {}
        for role, role_item in self.agent_data.items():
            role_emb = torch.tensor(role_item["embedding"], dtype=torch.float32).to(self.device)
            pool = []
            for a in self.action_map.values():
                pool.append(torch.cat([role_emb.reshape(-1), a["embedding"].reshape(-1)], dim=0))
            self.agent_llm_embedding[role] = {
                "embedding": torch.stack(pool, dim=0).to(self.device),
            }

        # --- Flatten role-LLM embeddings for upstream compatibility ---
        llm_role_ids, llm_role_embs = [], []
        for role, pack in self.agent_llm_embedding.items():
            emb = pack["embedding"]
            for i in range(emb.shape[0]):
                llm_role_ids.append(f"{role}_{i}")
                llm_role_embs.append(emb[i].detach().clone())
        self.LLM_role_id_embed = {"LLM_role.id": llm_role_ids, "embedding": llm_role_embs}

        # --- Run mode ---
        self.test_mode = False
        self.current_test_index = 0

        # --- Runtime state (reset each episode) ---
        self.phase = "planner"  # 'planner' | 'executor' | 'summarizer' | 'final' | 'done'
        self.his_index = 0
        self.total_cost = 0.0

        # DFS stack; each node is a dict with fields:
        #   query, depth, planned, child_queries, child_idx, child_summaries,
        #   leaf_queries, leaf_idx, leaf_responses, summary
        self.stack = []
        self.final_summary = ""

        # Current observation and action pool
        self.state = None
        self.current_agent_llm_embedding = None

        # Local memory table for the current episode
        self.local_memory_table = None
        self.table_query_id = -1
        self.current_query = None

        # Full query id/embedding table (for upstream network compatibility)
        self.query_id_embed = None

    # ------------------------------------------------------------------ #
    #                          Utility methods                            #
    # ------------------------------------------------------------------ #

    def enable_test_mode(self):
        self.test_mode = True
        self.current_test_index = 0

    def disable_test_mode(self):
        self.test_mode = False
        self.current_test_index = 0

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
        """Append a single record to the local memory table."""
        for k in ["query.id", "LLM_role.id", "score", "cost", "n_i", "n_o"]:
            self.local_memory_table[k].append(row[k])

    # ------------------------------------------------------------------ #
    #                       Environment interface                         #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset environment for a new episode and return initial observation."""
        # Sample a query
        if self.test_mode:
            if self.current_test_index >= len(self.test_query_data):
                self.current_test_index = 0
            row = self.test_query_data.iloc[self.current_test_index].to_dict()
            self.table_query_id = int(row["query.id"])
            self.current_test_index += 1
        else:
            row = self.query_data.sample(1).iloc[0].to_dict()
            self.table_query_id = int(row["query.id"])
        self.current_query = row

        # Build full query id + embedding table
        src = self.test_query_data if self.test_mode else self.query_data
        self.query_id_embed = {
            "query.id": src["query.id"].tolist(),
            "query_embedding": [self.to_tensor(x).to(self.device) for x in src["query_embedding"]],
        }

        # Initialize DFS stack with the root node
        self.stack = [{
            "query": self.current_query["query"],
            "depth": 0,
            "planned": False,
            "child_queries": [],
            "child_idx": 0,
            "child_summaries": [],
            "leaf_queries": [],
            "leaf_idx": 0,
            "leaf_responses": [],
            "summary": "",
        }]

        self.phase = "planner"
        self.his_index = 0
        self.total_cost = 0.0
        self.final_summary = ""

        # Initial observation: root query embedding; action pool: planner LLMs
        self.state = self.to_tensor(self.current_query["query_embedding"]).to(self.device)
        self.current_agent_llm_embedding = self.agent_llm_embedding["planner"]["embedding"].clone()

        # Initialize empty local memory table
        self.local_memory_table = {
            "query.id": [],
            "LLM_role.id": [],
            "score": [],
            "cost": [],
            "n_i": [],
            "n_o": [],
        }

        return (
            self.state,
            self.current_agent_llm_embedding,
            self.query_id_embed,
            self.LLM_role_id_embed,
            self.local_memory_table,
            self.his_index,
        )

    def next_step(self, action: int):
        """
        Advance the environment by one LLM call based on the current phase.

        Returns:
            (state, done, step_cost, reward, agent_llm_embedding,
             local_memory_table, his_index)
        """
        llm_name = self.action_map[action]["llm"]

        if self.phase == "planner":
            return self._step_planner(action, llm_name)
        elif self.phase == "executor":
            return self._step_executor(action, llm_name)
        elif self.phase == "summarizer":
            return self._step_summarizer(action, llm_name)
        elif self.phase == "final":
            return self._step_final(action, llm_name)
        elif self.phase == "done":
            return (self.state, True, 0.0, 0.0,
                    self.current_agent_llm_embedding,
                    self.local_memory_table, self.his_index)
        else:
            raise ValueError(f"Unknown phase: {self.phase}")

    def obtain_question(self):
        """Return the question string for the current phase (used by baselines)."""
        if self.phase == "planner":
            node = self.stack[-1]
            return (f"Decompose the following question into {self.width} "
                    f"atomic sub-questions:\n{node['query']}")
        elif self.phase == "executor":
            node = self.stack[-1]
            li = node["leaf_idx"]
            return node["leaf_queries"][li] if li < len(node["leaf_queries"]) else ""
        elif self.phase == "summarizer":
            node = self.stack[-1]
            if self._is_leaf_level(node["depth"]):
                context = "\n".join(node["leaf_responses"])
            else:
                context = "\n".join(node["child_summaries"])
            return ("Summarize the following content into a concise paragraph "
                    f"without bullet points. Content:\n{context}")
        elif self.phase == "final":
            return self.current_query["query"]
        else:
            return ""

    # ------------------------------------------------------------------ #
    #                  Phase implementations (DFS logic)                   #
    # ------------------------------------------------------------------ #

    def _is_leaf_level(self, node_depth: int) -> bool:
        """Check if this node is at the leaf level (one planner away from leaves)."""
        return node_depth == self.depth - 1

    def _parse_lines(self, text: str, fallback: str) -> list:
        """Parse planner output into a list of sub-queries, capped at self.width."""
        lines = [x.strip() for x in text.split("\n") if x.strip()]
        if not lines:
            lines = [fallback]
        if len(lines) > self.width:
            lines = lines[:self.width]
        return lines

    def _step_planner(self, action: int, llm_name: str):
        """Planner phase: decompose current node's query into sub-queries."""
        node = self.stack[-1]
        query = node["query"]

        sub_queries_str, prompt_used = self.planner(query, self.width, llm_name)
        step_cost = self.compute_cost(llm_name, prompt_used, sub_queries_str)
        self.total_cost += step_cost

        out_emb = torch.tensor(
            self.embedding_model.get_embedding(sub_queries_str),
            dtype=torch.float32,
        ).to(self.device)
        self.append_local_memory({
            "query.id": self.table_query_id,
            "LLM_role.id": f"planner_d{node['depth']}_{action}",
            "score": 0.0,
            "cost": step_cost / 1000.0,
            "n_i": self.state,
            "n_o": out_emb,
        })

        node["planned"] = True

        if self._is_leaf_level(node["depth"]):
            # Leaf level: planner produces leaf queries, switch to executor
            node["leaf_queries"] = self._parse_lines(sub_queries_str, fallback=query)
            node["leaf_idx"] = 0
            node["leaf_responses"] = []

            self.phase = "executor"
            self.current_agent_llm_embedding = self.agent_llm_embedding["executor"]["embedding"].clone()
            self.his_index = 1
            nxt = node["leaf_queries"][0]
            self.state = torch.tensor(
                self.embedding_model.get_embedding(nxt), dtype=torch.float32,
            ).to(self.device)
        else:
            # Internal node: planner produces child queries, DFS into first child
            node["child_queries"] = self._parse_lines(sub_queries_str, fallback=query)
            node["child_idx"] = 0
            node["child_summaries"] = []

            child_q = node["child_queries"][0]
            self.stack.append({
                "query": child_q,
                "depth": node["depth"] + 1,
                "planned": False,
                "child_queries": [],
                "child_idx": 0,
                "child_summaries": [],
                "leaf_queries": [],
                "leaf_idx": 0,
                "leaf_responses": [],
                "summary": "",
            })

            self.phase = "planner"
            self.current_agent_llm_embedding = self.agent_llm_embedding["planner"]["embedding"].clone()
            self.his_index = 0
            self.state = torch.tensor(
                self.embedding_model.get_embedding(child_q), dtype=torch.float32,
            ).to(self.device)

        done, reward = False, 0.0
        return (self.state, done, step_cost, reward,
                self.current_agent_llm_embedding,
                self.local_memory_table, self.his_index)

    def _step_executor(self, action: int, llm_name: str):
        """Executor phase: execute a leaf query."""
        node = self.stack[-1]
        assert self._is_leaf_level(node["depth"]), "Executor should only run at the leaf level"

        li = node["leaf_idx"]
        if li < len(node["leaf_queries"]):
            cur_leaf = node["leaf_queries"][li]
            resp, prompt_used = self.executor(cur_leaf, llm_name)
            step_cost = self.compute_cost(llm_name, prompt_used, resp)
            self.total_cost += step_cost

            node["leaf_responses"].append(resp)

            out_emb = torch.tensor(
                self.embedding_model.get_embedding(resp), dtype=torch.float32,
            ).to(self.device)
            self.append_local_memory({
                "query.id": self.table_query_id,
                "LLM_role.id": f"executor_leaf_d{node['depth']}_{li}_{action}",
                "score": 0.0,
                "cost": step_cost / 1000.0,
                "n_i": self.state,
                "n_o": out_emb,
            })

            node["leaf_idx"] += 1

            if node["leaf_idx"] < len(node["leaf_queries"]):
                # More leaves to execute
                nxt = node["leaf_queries"][node["leaf_idx"]]
                self.state = torch.tensor(
                    self.embedding_model.get_embedding(nxt), dtype=torch.float32,
                ).to(self.device)
                self.phase = "executor"
                self.current_agent_llm_embedding = self.agent_llm_embedding["executor"]["embedding"].clone()
                self.his_index = 1
            else:
                # All leaves done, switch to summarizer
                ctx = "\n".join(node["leaf_responses"])
                self.state = torch.tensor(
                    self.embedding_model.get_embedding(ctx), dtype=torch.float32,
                ).to(self.device)
                self.phase = "summarizer"
                self.current_agent_llm_embedding = self.agent_llm_embedding["summarizer"]["embedding"].clone()
                self.his_index = 1

            done, reward = False, 0.0
            return (self.state, done, step_cost, reward,
                    self.current_agent_llm_embedding,
                    self.local_memory_table, self.his_index)

        # Should not reach here under normal operation
        return (self.state, False, 0.0, 0.0,
                self.current_agent_llm_embedding,
                self.local_memory_table, self.his_index)

    def _step_summarizer(self, action: int, llm_name: str):
        """Summarizer phase: aggregate child/leaf results and pop the stack."""
        node = self.stack[-1]

        # Build context from leaf responses or child summaries
        if self._is_leaf_level(node["depth"]):
            ctx = "\n".join(node["leaf_responses"])
        else:
            ctx = "\n".join(node["child_summaries"])

        summary, prompt_used = self.summarizer(ctx, llm_name)
        step_cost = self.compute_cost(llm_name, prompt_used, summary)
        self.total_cost += step_cost
        node["summary"] = summary

        out_emb = torch.tensor(
            self.embedding_model.get_embedding(summary), dtype=torch.float32,
        ).to(self.device)
        self.append_local_memory({
            "query.id": self.table_query_id,
            "LLM_role.id": f"summarizer_d{node['depth']}_{action}",
            "score": 0.0,
            "cost": step_cost / 1000.0,
            "n_i": self.state,
            "n_o": out_emb,
        })

        # Node complete: pop from stack and propagate summary upward
        finished = self.stack.pop()

        if len(self.stack) == 0:
            # Root node finished: enter final phase
            self.final_summary = finished["summary"]
            self.phase = "final"
            self.current_agent_llm_embedding = self.agent_llm_embedding["executor"]["embedding"].clone()
            self.his_index = 1
            self.state = out_emb
            done, reward = False, 0.0
            return (self.state, done, step_cost, reward,
                    self.current_agent_llm_embedding,
                    self.local_memory_table, self.his_index)

        # Parent still has work to do
        parent = self.stack[-1]
        parent["child_summaries"].append(finished["summary"])

        if parent["planned"] and parent["child_idx"] + 1 < len(parent["child_queries"]):
            # More siblings: DFS into the next child
            parent["child_idx"] += 1
            next_child_q = parent["child_queries"][parent["child_idx"]]
            self.stack.append({
                "query": next_child_q,
                "depth": parent["depth"] + 1,
                "planned": False,
                "child_queries": [],
                "child_idx": 0,
                "child_summaries": [],
                "leaf_queries": [],
                "leaf_idx": 0,
                "leaf_responses": [],
                "summary": "",
            })
            self.phase = "planner"
            self.current_agent_llm_embedding = self.agent_llm_embedding["planner"]["embedding"].clone()
            self.his_index = 0
            self.state = torch.tensor(
                self.embedding_model.get_embedding(next_child_q), dtype=torch.float32,
            ).to(self.device)
        else:
            # All children done: summarize the parent node next
            ctx_p = "\n".join(parent["child_summaries"])
            self.phase = "summarizer"
            self.current_agent_llm_embedding = self.agent_llm_embedding["summarizer"]["embedding"].clone()
            self.his_index = 1
            self.state = torch.tensor(
                self.embedding_model.get_embedding(ctx_p), dtype=torch.float32,
            ).to(self.device)

        done, reward = False, 0.0
        return (self.state, done, step_cost, reward,
                self.current_agent_llm_embedding,
                self.local_memory_table, self.his_index)

    def _step_final(self, action: int, llm_name: str):
        """Final phase: produce the answer using root summary + original query."""
        final_answer, prompt_used = self.executor(
            query=self.current_query["query"],
            llm_name=llm_name,
            if_final=True,
            context=self.final_summary,
        )
        if final_answer is None or final_answer.strip() == "":
            final_answer = "No Answer"
        step_cost = self.compute_cost(llm_name, prompt_used, final_answer)
        self.total_cost += step_cost

        reward = eval_perf(
            metric=self.current_query["metric"],
            prediction=final_answer,
            ground_truth=self.current_query["gt"],
            task_name=self.current_query["task_name"],
            task_id=self.current_query["task_id"],
        )

        out_emb = torch.tensor(
            self.embedding_model.get_embedding(final_answer), dtype=torch.float32,
        ).to(self.device)
        self.append_local_memory({
            "query.id": self.table_query_id,
            "LLM_role.id": f"executor_final_{action}",
            "score": reward,
            "cost": step_cost / 1000.0,
            "n_i": self.state,
            "n_o": out_emb,
        })

        self.state = out_emb
        self.phase = "done"
        self.his_index = 2
        done = True
        return (self.state, done, step_cost, reward,
                self.current_agent_llm_embedding,
                self.local_memory_table, self.his_index)

    # ------------------------------------------------------------------ #
    #                         LLM call templates                          #
    # ------------------------------------------------------------------ #

    def compute_cost(self, llm_name: str, input_text: str, output_text: str) -> float:
        """Estimate the dollar cost of an LLM call based on token counts."""
        in_tokens = len(tokenizer(input_text)["input_ids"])
        out_tokens = len(tokenizer(output_text)["input_ids"])
        in_price = self.llm_data[llm_name]["input_price"]
        out_price = self.llm_data[llm_name]["output_price"]
        return in_tokens * in_price + out_tokens * out_price

    def planner(self, query: str, num: int, llm_name: str = "mistral-nemo-12b-instruct"):
        """Decompose a query into exactly ``num`` atomic sub-queries."""
        prompt = f"""You are a query decomposition assistant.

Your task is to decompose the user's query into exactly {num} atomic and independent sub-queries.
- Keep each sub-query self-contained and non-overlapping.
- Prefer factual, directly answerable units.
- Output EXACTLY {num} lines, one sub-query per line, no numbering or extra words.

User query:
{query}
"""
        resp = model_prompting(self.llm_data[llm_name]["model"], prompt)
        return resp, prompt

    def executor(self, query: str, llm_name: str = "mistral-nemo-12b-instruct",
                 if_final: bool = False, context: str = "") -> tuple:
        """
        Execute a query, optionally using additional context for the final answer.

        Args:
            query: The user's original question or task.
            llm_name: Name of the LLM to use.
            if_final: Whether to include context in the prompt (final phase).
            context: Background information for answering.

        Returns:
            (response, prompt_used)
        """
        if if_final:
            query = generate_task_query(
                task_name=self.current_query["task_name"],
                sample_data=self.current_query,
            )
            prompt = f"""You are a helpful assistant.
Given the following context and the user's query, provide a direct, complete, and accurate answer. You must output according to the output format required in the query and do not output any additional information.

User query:
{query}

Context:
{context}

Answer:
"""
        else:
            prompt = query

        response = model_prompting(llm_model=self.llm_data[llm_name]["model"], prompt=prompt)
        return response, prompt

    def summarizer(self, context: str, llm_name: str = "mistral-nemo-12b-instruct"):
        """Summarize content into a concise paragraph."""
        prompt = f"""You are a professional summarizer.
Summarize the following content into a concise, coherent paragraph without bullet points.
Make it fluent and logically connected.

Content:
{context}

Summary:"""
        resp = model_prompting(self.llm_data[llm_name]["model"], prompt)
        return resp, prompt
