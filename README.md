---
title: ZeroTrace
emoji: 👾
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
tags:
  - openenv
  - code-repair
  - reinforcement-learning
  - agent-evaluation
  - pytorch
---

# ZeroTrace

**Leave zero bugs. Leave zero trace.**

An autonomous benchmark for self-healing code and logic repair.
A language model agent diagnoses and patches real Python and PyTorch bugs,
verified by hidden unit tests across six difficulty levels.

Built for the [Meta PyTorch Hackathon](https://pytorch.devpost.com/).

---

## What makes this different

Most coding benchmarks ask models to generate code from scratch.
ZeroTrace tests something harder: **finding and fixing bugs in existing code** --
the task engineers actually spend their time on.

The agent operates in a step-based loop. It reads the buggy code, inspects
errors, edits the file, runs tests, and submits a fix. All within a sandboxed
environment with a structured reward signal, security scanning, and
a 15-step budget.

---

## Environment spec

| Field          | Value |
|----------------|-------|
| Observation    | Buggy code, terminal output, test results, conversation history |
| Actions        | `INSPECT_ERROR`, `EDIT_CODE`, `RUN_COMPILER`, `EXECUTE_UNIT_TEST`, `QUERY_CONTEXT`, `SUBMIT_FIX`, `SEARCH_DOCS`, `RUN_SNIPPET` |
| Reward         | Partial credit per test passed, linear speed decay per step, penalties for bad actions |
| Episode length | Max 15 steps |
| Security       | Static code scanner blocks dangerous patterns before sandbox execution |

---

## Tasks

### Classic Python (Levels 1-3)

| Level | Task | Difficulty | Bug |
|-------|------|------------|-----|
| 1 | Fix KeyError | Easy | Missing dictionary key guard |
| 2 | Fix resource leak | Medium | Unclosed file handle |
| 3 | Fix race condition | Hard | Thread-unsafe counter |

### PyTorch (Levels 4-6)

| Level | Task | Difficulty | Bug |
|-------|------|------------|-----|
| 4 | Fix dtype mismatch | Easy | Float targets passed to CrossEntropyLoss (needs Long) |
| 5 | Fix NaN gradient | Medium | `log(0)` in custom cross-entropy causes NaN loss |
| 6 | Fix softmax dimension | Hard | `softmax(dim=0)` instead of `dim=-1` in attention |
| 7 | Fix DDP batch shape | Expert | Linear layer and DistributedDataParallel (DDP) missing batch dimension |

---

## Baseline scores

Tested with `Qwen/Qwen2.5-72B-Instruct` via `inference.py`:

| Task | Reward | Steps |
|------|--------|-------|
| Level 1 -- KeyError | 1.00 | 3 |
| Level 2 -- Resource leak | 1.00 | 3 |
| Level 3 -- Race condition | 1.00 | 3 |
| **Mean** | **1.00** | **3** |

Scores may vary between runs depending on model non-determinism.

---

## Features

- **Multi-turn memory** -- agent sees its last 5 conversation turns for better reasoning
- **Leaderboard** -- thread-safe JSON-backed ranking, auto-recorded after each episode
- **Replay** -- every episode is logged step-by-step; scrub through past runs in the UI
- **Model comparison** -- run two models concurrently on the same task, side-by-side results
- **Report export** -- generate and download a Markdown summary of any run
- **Offline tools** -- `SEARCH_DOCS` queries a built-in Python/PyTorch doc index; `RUN_SNIPPET` runs short verification code in a 5-second sandbox
- **Security scanning** -- static analysis blocks `os.system`, `eval`, `exec`, `subprocess`, network calls, and other dangerous patterns before code reaches the sandbox
- **API versioning** -- all routes under `/api/v1/`; legacy unversioned routes kept as deprecated aliases

---

## API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/reset` | POST | Reset environment to a task |
| `/api/v1/step` | POST | Take an action |
| `/api/v1/state` | GET | Get current observation |
| `/api/v1/tasks` | GET | List all available tasks |
| `/api/v1/leaderboard` | GET | Aggregated model rankings |
| `/api/v1/leaderboard/record` | POST | Manually record a result |
| `/api/v1/replay/{task_id}` | GET | Fetch replay log for a task |

Legacy routes (`/reset`, `/step`, `/state`, `/tasks`) are preserved for backward compatibility.

---

## Setup

### Local development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/ZeroTrace.git
cd ZeroTrace

# Install dependencies
pip install -r requirements.txt

# Set environment variables
# On Linux/Mac:
export HF_TOKEN=your_huggingface_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# On Windows (PowerShell):
$env:HF_TOKEN="your_huggingface_token"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"

# Start the server
python app.py
# Open http://localhost:7860
```

### Docker

```bash
docker build -t zerotrace .
docker run -p 7860:7860 \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e HF_TOKEN=your_huggingface_token \
  zerotrace
```

---

## Running inference

```bash
# Classic tasks only (default)
python inference.py

# Include PyTorch tasks
# On Linux/Mac:
export RUN_TORCH_TASKS=1
# On Windows:
$env:RUN_TORCH_TASKS="1"

python inference.py
```

The script hits the live server at `http://localhost:7860` (configurable via `ZEROTRACE_BASE_URL`).

---

## Reward function

| Signal | Value | Trigger |
|--------|-------|---------|
| Full pass | `+1.0` | All tests pass |
| Partial credit | `passed / total` | Some tests pass |
| Bad submit | `-0.3` | Submitting with failing tests |
| Unnecessary compiler check | `-0.1` | Running compiler on valid syntax |
| Efficiency penalty | `-0.05 * (step - 10)` | After step 10 |

Final reward is clamped to `[-1.0, 1.0]`.

---

## Project structure

```
ZeroTrace/
 app.py                   Main application (Gradio UI + FastAPI)
 inference.py             CLI inference script
 openenv.yaml             OpenEnv specification
 Dockerfile               Container config
 requirements.txt         Python dependencies
 environment/
   models.py              Pydantic request/response models
   env.py                 FastAPI routes (v1 + legacy)
   sandbox.py             Subprocess-based code execution
   test_runner.py          Graders for all 6 levels
   state_machine.py       Episode lifecycle + replay + security
 tasks/
   level1.py              KeyError bug
   level2.py              Resource leak bug
   level3.py              Race condition bug
   torch_dtype.py         PyTorch dtype mismatch
   torch_nan_grad.py      PyTorch NaN gradient
   torch_wrong_dim.py     PyTorch softmax dimension
 agent/
   prompt_builder.py      System/user prompt construction
   zerotrace_agent.py     Action parser + agent runner
   tools.py               Offline doc search + snippet runner
 leaderboard/
   store.py               JSON-backed leaderboard
   replay.py              Episode replay storage
 reports/
   exporter.py            Markdown report generator
 security/
   scanner.py             Static code safety scanner
```

---

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | -- | HuggingFace API token |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model to use for inference |
| `ZEROTRACE_BASE_URL` | No | `http://localhost:7860` | Server URL for inference.py |
| `RUN_TORCH_TASKS` | No | `0` | Set to `1` to include PyTorch tasks in inference |

---

## License

All Rights Reserved. See [LICENSE](LICENSE) for details.
