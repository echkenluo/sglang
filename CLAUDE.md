# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SGLang is a high-performance serving framework for large language models (LLMs) and vision-language models (VLMs). It provides fast inference with features like RadixAttention for prefix caching, continuous batching, tensor/pipeline/expert/data parallelism, and structured outputs.

## Repository Structure

- **python/sglang/** - Main Python package
  - `srt/` - SGLang Runtime (backend serving engine)
    - `entrypoints/` - Server entry points (`engine.py`, `http_server_engine.py`, `grpc_server.py`)
    - `managers/` - Core runtime managers (`scheduler.py`, `tokenizer_manager.py`, `detokenizer_manager.py`)
    - `models/` - Model implementations (130+ supported models)
    - `layers/` - Neural network layers and attention backends
    - `mem_cache/` - Memory and KV cache management
    - `lora/` - LoRA adapter support
    - `disaggregation/` - Prefill-decode disaggregation
  - `lang/` - Frontend language for programming LLM applications
    - `api.py` - Public API (`gen`, `select`, `image`, `video`, etc.)
    - `interpreter.py` - SGLang program interpreter
    - `backend/` - Backend connectors (OpenAI, Anthropic, etc.)
- **sgl-kernel/** - CUDA kernel library (separate Python package)
- **test/** - Test suites
  - `srt/` - Backend runtime tests
  - `lang/` - Frontend language tests
- **benchmark/** - Performance benchmarking scripts
- **docs/** - Documentation (Sphinx-based)

## Common Commands

### Installation (Development)
```bash
pip install -e "python"
```

### Running a Server
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000
```

### Running Tests
```bash
# Run a single test file
cd test/srt
python test_srt_endpoint.py

# Run a single test method
python test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Run test suite (from test/ directory)
cd test
python run_suite.py --hw cuda --suite stage-a-test-1
```

### Code Formatting
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

### Benchmarking
```bash
# Single batch latency (no server needed)
python -m sglang.bench_one_batch --model-path <model> --batch 32 --input-len 256 --output-len 32

# Offline throughput
python -m sglang.bench_offline_throughput --model-path <model> --num-prompts 10

# Online serving (requires running server)
python -m sglang.bench_serving --backend sglang --num-prompt 10
```

## Architecture Overview

### Engine Components
The `Engine` class (in `srt/entrypoints/engine.py`) orchestrates three components:
1. **TokenizerManager** - Tokenizes requests and sends to scheduler
2. **Scheduler** (subprocess) - Manages batching, forwards requests, runs model inference via `TpWorker`
3. **DetokenizerManager** (subprocess) - Detokenizes output tokens

### Request Flow
1. Request enters via HTTP/gRPC endpoint or Engine API
2. TokenizerManager tokenizes and forwards to Scheduler
3. Scheduler batches requests and runs inference through TpWorker/ModelRunner
4. Output tokens sent to DetokenizerManager
5. Detokenized response returned to client

### Key Abstractions
- **ServerArgs** (`srt/server_args.py`) - All server configuration options
- **ModelRunner** (`srt/model_executor/model_runner.py`) - Manages model execution
- **RadixCache** (`srt/mem_cache/`) - Prefix caching with radix tree

## Code Style Guidelines

- Avoid code duplication (extract shared functions for 5+ repeated lines)
- Minimize CPU-GPU synchronization (`tensor.item()`, `tensor.cpu()`)
- Cache runtime checks in model forward passes as boolean values
- Keep files under 2,000 lines; split if needed
- Tests should run under 500 seconds per file
- New hardware support: prefer new files over modifying existing code
- Common code path (NVIDIA) should be first branch in if/else blocks

## sgl-kernel Updates

sgl-kernel is a separate package. To update kernels:
1. Submit PR to update kernel source (don't use it yet in sglang)
2. Bump sgl-kernel version (triggers PyPI release)
3. Update version in `python/pyproject.toml` and add caller code

## CI Information

- CI permission required to trigger tests (listed in `.github/CI_PERMISSIONS.json`)
- Add "run-ci" label or use `/tag-run-ci-label` comment
- `/rerun-failed-ci` reruns failed tests
- `/rerun-stage <stage-name>` reruns specific stage

## Testing Notes

- Tests use Python's `unittest` framework
- Add `unittest.main()` for unittest or `pytest.main([__file__])` for pytest
- Reuse server launches across test cases for speed
- Register CI tests with `register_cuda_ci(est_time=X, suite="suite-name")`
