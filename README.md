# GraphPlanner: Graph Memory-Augmented Agentic Routing for Multi-Agent LLMs

GraphPlanner is a reinforcement learning framework for intelligently routing user queries to optimal LLM agents. It uses a PPO-based policy network with database-inspired relational aggregation layers (DBLayer) and optional Graph Neural Network (GNN) backbones to learn routing decisions from graph-structured interaction memory.

## Architecture Overview

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
    ├── graph_builder.py       # (Re-exported from dynamic_router)
    ├── gnn_baselines.py       # (Re-exported from dynamic_router)
    ├── route_env.py           # DFS-based fixed-width/depth environment
    └── train.py               # PPO training loop (no action masking)
```

### Key Components

#### DBLayer (Database Layer)

A novel relational aggregation layer inspired by database operations. It processes multiple "tables" (query, LLM, memory) through per-table linear transformations, then uses scatter-based aggregation along foreign key relationships to propagate information across entities.

#### Dynamic Router

- **Adaptive decomposition**: The agent decides whether to decompose queries, execute directly, or summarize, using a tree-based structure with configurable max width and depth.
- **Action masking**: The environment computes valid actions per state (e.g., only allow planner when decomposition budget remains).
- **Dual DBLayers**: Separate processing of local (current episode) and historical (cross-episode) memory tables.
- **Task/Query embeddings**: Additional context from task type and query content for state representation.

#### Static Router

- **Fixed decomposition**: Uses a predetermined width/depth DFS structure. The planner always decomposes into exactly `width` sub-queries at each level.
- **Role-based selection**: The policy selects which LLM to use for each predetermined role (planner/executor/summarizer).
- **Simpler architecture**: No action masking needed; roles are determined by the DFS traversal phase.

#### GNN Backbones

Both routers support three backbone options for table embedding:

| Backbone | Description |
|----------|-------------|
| `dblayer` (default) | Custom DBLayer with scatter-based relational aggregation |
| `homo` | Homogeneous GCN on a unified query-LLM interaction graph |
| `hetero` | Heterogeneous GCN with separate node/edge types for queries and LLMs |

## Environment Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda (for environment management)

### Installation

```bash
# Create and activate conda environment
conda create -n router_planner python=3.10
conda activate router_planner

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 11.8:
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

# Install additional utilities
pip install tiktoken                 # Token counting for cost estimation
pip install datasets                 # HuggingFace datasets
pip install accelerate               # Model acceleration
```

### Key Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >= 2.0 | Core deep learning framework |
| `torch-geometric` | >= 2.3 | Graph neural network operations |
| `torch-scatter` | >= 2.1 | Scatter operations for DBLayer |
| `transformers` | >= 4.30 | Longformer embedding model |
| `openai` | >= 1.0 | NVIDIA NIM API client |
| `bert-score` | >= 0.3 | BERT-score evaluation |
| `scikit-learn` | >= 1.0 | PCA and ML utilities |
| `wandb` | >= 0.15 | Experiment tracking |
| `pandas` | >= 2.0 | Data management |

### API Configuration

The framework uses the NVIDIA NIM API (OpenAI-compatible) for LLM inference. Set your API keys as an environment variable:

```bash
export NVIDIA_API_KEYS="[YOUR_API_KEY_1],[YOUR_API_KEY_2],[YOUR_API_KEY_3]"
```

Multiple keys enable thread-safe round-robin load balancing across API endpoints.

For W&B experiment tracking:

```bash
export WANDB_API_KEY="[YOUR_WANDB_API_KEY]"
```

### Data Preparation

Place the following data files in a `data/` directory at the project root:

```
data/
├── router_data_train.csv        # Training queries with embeddings
├── router_data_test.csv         # Test queries with embeddings
├── HumanEval.jsonl              # HumanEval benchmark problems
└── mbpp.jsonl                   # MBPP benchmark problems
```

Configuration files in a `config/` directory:

```
config/
├── llm_descriptions_with_embeddings.json   # LLM metadata and embeddings
└── agent_roles_with_embeddings.json        # Agent role metadata and embeddings
```

The CSV files should contain these columns: `query`, `query_embedding`, `gt` (ground truth), `metric`, `task_name`, `task_id`, `choices` (for MC tasks).

## Usage

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
    --wandb_project router_planner_dynamic \
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
    --wandb_project router_planner_static \
    --experiment_name static_v1
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--gnn_backbone` | GNN backbone: `dblayer`, `homo`, or `hetero` | `dblayer` |
| `--max_planner_calls` | Max decomposition calls per episode (dynamic only) | 2 |
| `--width` | Fixed decomposition width (static only) | 3 |
| `--depth` | Fixed decomposition depth (static only) | 2 |
| `--max_episodes` | Total training episodes | 1000 |
| `--batch_size` | Training batch size (parallel environments) | 1 |
| `--test_batch_size` | Testing batch size | 20 |
| `--test_every_n_episodes` | Evaluation frequency during training | 20 |
| `--force_planner_episodes` | Initial episodes forcing planner use (dynamic) | 0 |
| `--save_dir` | Directory for model checkpoints | `./checkpoints` |
| `--load_best_model_path` | Resume training from a saved checkpoint | None |

## Supported Benchmarks

The framework evaluates across 12+ benchmarks in multiple categories:

| Category | Tasks | Metric |
|----------|-------|--------|
| **Math** | GSM8K, MATH | Numeric answer extraction |
| **Code** | HumanEval, MBPP | Functional correctness (pass@1) |
| **Knowledge** | NaturalQA, TriviaQA | Contained Exact Match (CEM) |
| **Reasoning** | MMLU, GPQA | Multiple-choice accuracy |
| **Commonsense** | CommonsenseQA, OpenBookQA, ARC-Challenge | Multiple-choice accuracy |
| **Generation** | CommonGen | Concept coverage |

## License

This project is for research purposes.
