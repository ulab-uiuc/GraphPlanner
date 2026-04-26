# GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs

<p align="center">
    <a href="https://github.com/ulab-uiuc/GraphPlanner">
        <img alt="Stars" src="https://img.shields.io/github/stars/ulab-uiuc/GraphPlanner">
    </a>
    <a href="https://github.com/ulab-uiuc/GraphPlanner">
        <img alt="Forks" src="https://img.shields.io/github/forks/ulab-uiuc/GraphPlanner">
    </a>
    <a href="https://github.com/ulab-uiuc/GraphPlanner/issues">
        <img alt="Issues" src="https://img.shields.io/github/issues/ulab-uiuc/GraphPlanner">
    </a>
    <a href="https://github.com/ulab-uiuc/GraphPlanner/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
    </a>
    <!-- <a href="[ARXIV_URL]">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-red?logo=arxiv">
    </a> -->
</p>

<!-- <p align="center">
    <a href="[PROJECT_PAGE_URL]">🌐 Project Page</a> |
    <a href="[ARXIV_URL]">📜 arXiv</a> |
    <a href="[DATASET_URL]">📂 Dataset</a>
</p> -->


<div align="center">
  <!-- <img src="./figures/overview.png" width="700" alt="GraphPlanner Overview"> -->
  <p><b>GraphPlanner learns to route queries to optimal LLM agents using graph memory-augmented reinforcement learning with PPO and relational database-inspired aggregation layers.</b></p>
</div>


## 🔥 News

**[2026.02]** 🚀 Initial release of **GraphPlanner** — Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs.


## 📖 Overview

GraphPlanner is a reinforcement learning framework for intelligently routing user queries to optimal LLM agents. It uses a PPO-based policy network with **database-inspired relational aggregation layers (DBLayer)** and optional **Graph Neural Network (GNN) backbones** to learn routing decisions from graph-structured interaction memory.

### Key Features

- **Graph Memory**: Builds query-LLM interaction graphs from historical execution traces, enabling experience transfer across episodes
- **Dual DBLayers**: Novel relational aggregation layers that process both local (current episode) and historical (cross-episode) memory tables via scatter-based foreign key propagation
- **Dynamic & Static Routing**: Two routing modes — adaptive tree-based decomposition with action masking, and fixed-structure DFS decomposition
- **GNN Backbones**: Pluggable graph encoders including HomoGCN and HeteroGCN as alternatives to the default DBLayer
- **Multi-Agent Orchestration**: Learned routing across planner, executor, and summarizer roles with multiple LLM backends
- **12+ Benchmark Support**: Comprehensive evaluation across math, code, knowledge, reasoning, commonsense, and generation tasks


## 🏗️ Architecture

```
router_planner/
├── shared/                    # Shared utilities across both routers
│   ├── utils.py               # Embedding model, text metrics, LLM API client
│   ├── task_prompting.py      # Task-specific prompt formatting (12+ benchmarks)
│   ├── response_eval.py       # Unified evaluation dispatcher
│   ├── math_eval.py           # MATH benchmark expression normalization
│   ├── agent_call.py          # Standalone planner/executor/summarizer functions
│   └── evaluation/            # Code execution evaluation
│       ├── human_eval/        # HumanEval benchmark evaluation harness
│       └── mbpp/              # MBPP benchmark evaluation harness
│
├── dynamic_router/            # Dynamic routing with adaptive decomposition
│   ├── network.py             # PolicyNetwork + ValueNetwork with dual DBLayers
│   ├── gnn_baselines.py       # HomoGCN / HeteroGCN backbone alternatives
│   ├── graph_builder.py       # PyG graph construction from memory tables
│   ├── route_env.py           # Tree-based query decomposition environment
│   └── train.py               # PPO training loop with action masking
│
└── static_router/             # Static routing with fixed decomposition
    ├── network.py             # Simplified PolicyNetwork + ValueNetwork
    ├── route_env.py           # DFS-based fixed-width/depth environment
    └── train.py               # PPO training loop
```

### Dynamic Router

- **Adaptive decomposition**: The agent decides whether to decompose queries, execute directly, or summarize, using a tree-based structure with configurable max width and depth
- **Action masking**: The environment computes valid actions per state (e.g., only allow planner when decomposition budget remains)
- **Dual DBLayers**: Separate processing of local and historical memory tables with task/query embeddings

### Static Router

- **Fixed decomposition**: Predetermined width/depth DFS structure where the planner always decomposes into exactly `width` sub-queries at each level
- **Role-based selection**: The policy selects which LLM to use for each predetermined role (planner/executor/summarizer)

### GNN Backbones

Both routers support three backbone options for table embedding:

| Backbone | Description |
|----------|-------------|
| `dblayer` (default) | Custom DBLayer with scatter-based relational aggregation |
| `homo` | Homogeneous GCN on a unified query-LLM interaction graph |
| `hetero` | Heterogeneous GCN with separate node/edge types for queries and LLMs |


## 🛠️ Environment Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda (for environment management)

### Installation

```bash
# Create and activate conda environment
conda create -n graphplanner python=3.10
conda activate graphplanner

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and scatter extensions
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install core dependencies
pip install transformers             # Longformer embedding model
pip install openai                   # NVIDIA NIM API client (OpenAI-compatible)
pip install sentence-transformers    # Sentence embeddings
pip install bert-score               # BERT-score evaluation metric
pip install scikit-learn             # PCA dimensionality reduction
pip install pandas numpy tqdm        # Data processing and progress bars
pip install wandb                    # Experiment tracking
pip install tiktoken                 # Token counting for cost estimation
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0 | Core deep learning framework |
| `torch-geometric` | >= 2.3 | Graph neural network operations |
| `torch-scatter` | >= 2.1 | Scatter operations for DBLayer |
| `transformers` | >= 4.30 | Longformer embedding model |
| `openai` | >= 1.0 | NVIDIA NIM API client |
| `bert-score` | >= 0.3 | BERT-score evaluation |
| `wandb` | >= 0.15 | Experiment tracking |

### API Configuration

Set your API keys as environment variables:

```bash
# NVIDIA NIM API keys (multiple keys for round-robin load balancing)
export NVIDIA_API_KEYS="[YOUR_API_KEY_1],[YOUR_API_KEY_2],[YOUR_API_KEY_3]"

# W&B experiment tracking (optional)
export WANDB_API_KEY="[YOUR_WANDB_API_KEY]"
```


## 📂 Data Preparation

Place the following data files in `data/` and `config/` directories at the project root:

```
data/
├── router_data_train.csv        # Training queries with embeddings
├── router_data_test.csv         # Test queries with embeddings
├── HumanEval.jsonl              # HumanEval benchmark problems
└── mbpp.jsonl                   # MBPP benchmark problems

config/
├── llm_descriptions_with_embeddings.json   # LLM metadata and embeddings
└── agent_roles_with_embeddings.json        # Agent role metadata and embeddings
```

The CSV files should contain these columns: `query`, `query_embedding`, `gt` (ground truth), `metric`, `task_name`, `task_id`, `choices` (for MC tasks).


## 🚀 Usage

### Training the Dynamic Router

```bash
python -m router_planner.dynamic_router.train \
    --query_file data/router_data_train.csv \
    --test_query_file data/router_data_test.csv \
    --llm_file config/llm_descriptions_with_embeddings.json \
    --agent_file config/agent_roles_with_embeddings.json \
    --max_planner_calls 2 \
    --max_episodes 1000 \
    --batch_size 1 \
    --test_batch_size 20 \
    --gnn_backbone dblayer \
    --wandb_project graphplanner_dynamic \
    --experiment_name dynamic_v1
```

### Training the Static Router

```bash
python -m router_planner.static_router.train \
    --query_file data/router_data_train.csv \
    --test_query_file data/router_data_test.csv \
    --llm_file config/llm_descriptions_with_embeddings.json \
    --agent_file config/agent_roles_with_embeddings.json \
    --width 3 \
    --depth 2 \
    --max_episodes 1000 \
    --gnn_backbone dblayer \
    --wandb_project graphplanner_static \
    --experiment_name static_v1
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--gnn_backbone` | GNN backbone: `dblayer`, `homo`, or `hetero` | `dblayer` |
| `--max_planner_calls` | Max decomposition calls per episode (dynamic only) | `2` |
| `--width` | Fixed decomposition width (static only) | `3` |
| `--depth` | Fixed decomposition depth (static only) | `2` |
| `--max_episodes` | Total training episodes | `1000` |
| `--batch_size` | Training batch size (parallel environments) | `1` |
| `--test_batch_size` | Testing batch size | `20` |
| `--test_every_n_episodes` | Evaluation frequency during training | `20` |
| `--force_planner_episodes` | Initial episodes forcing planner use (dynamic) | `0` |
| `--save_dir` | Directory for model checkpoints | `./checkpoints` |
| `--load_best_model_path` | Resume training from a saved checkpoint | `None` |


## 📊 Supported Benchmarks

The framework evaluates across 12+ benchmarks in multiple categories:

| Category | Tasks | Metric |
|----------|-------|--------|
| **Math** | GSM8K, MATH | Numeric answer extraction |
| **Code** | HumanEval, MBPP | Functional correctness (pass@1) |
| **Knowledge** | NaturalQA, TriviaQA | Contained Exact Match (CEM) |
| **Reasoning** | MMLU, GPQA | Multiple-choice accuracy |
| **Commonsense** | CommonsenseQA, OpenBookQA, ARC-Challenge | Multiple-choice accuracy |
| **Generation** | CommonGen | Concept coverage |


## 📜 Citation

```bibtex
@inproceedings{fenggraphplanner,
  title={GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs},
  author={Feng, Tao and Zhang, Haozhen and Lei, Zijie and Han, Peixuan and You, Jiaxuan},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```


## 📄 License

This project is released under the [MIT License](LICENSE).
