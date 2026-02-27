# CLAUDE.md

NVIDIA Model Optimizer (ModelOpt): open-source library for model optimization techniques including
quantization, pruning, distillation, sparsity, and speculative decoding to accelerate inference.
Primarily Python codebase with optional C++/CUDA extensions supporting PyTorch, ONNX, and Hugging Face/Megatron models.

> If a `CLAUDE.local.md` file exists alongside this file, read and respect it — it contains
> developer-specific overrides that supplement this shared guidance.

## Rules (Read First)

**CRITICAL (YOU MUST):**

- NVIDIA Apache 2.0 license header on ALL new Python/C++/CUDA files (see `LICENSE_HEADER`)
- `git commit -s -S` (DCO sign-off + cryptographic signing required). Never attribute AI tools in
  sign-off line
- `pre-commit` hooks run on commit — if files are modified by hooks, re-stage and commit again
- Rebase onto `main` (not merge): `git rebase origin/main`. We maintain linear history
- Use `git push --force-with-lease` (never `--force`) when pushing rebased branches
- PRs require CODEOWNERS review (auto-assigned based on `.github/CODEOWNERS`)
- After rebasing, always re-run tests locally before pushing
- Follow security coding practices (see Security section) — violations are blocked as pre-merge
  errors by CodeRabbit. Never use `# nosec` to bypass Bandit checks.

## Common Commands

| Task | Command |
|------|---------|
| Install (editable + dev) | `pip install -e ".[dev]"` |
| CPU unit tests | `python -m pytest tests/unit` |
| GPU unit tests | `python -m pytest tests/gpu` |
| Megatron GPU tests | `python -m pytest tests/gpu_megatron` |
| Pattern match | `pytest tests/unit -k "test_quantize"` |
| Lint + format (all files) | `pre-commit run --all-files` |
| Lint (diff only) | `pre-commit run --from-ref origin/main --to-ref HEAD` |
| Run via tox (CPU unit) | `tox -e py312-torch210-tf_latest-unit` |
| Build docs | `tox -e build-docs` |
| Build wheel | `tox -e build-wheel` |

### Installation Notes

GPU tests require CUDA and a compatible GPU. For features depending on TensorRT-LLM or
Megatron-Core, use a docker container:

```bash
# TensorRT-LLM release containers have ModelOpt pre-installed
nvcr.io/nvidia/tensorrt-llm/release:<version>
```

See [installation docs](https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html)
for alternative docker images and optional dependency groups.

## Architecture

ModelOpt is organized into three top-level namespaces:

| Namespace | Path | Role |
|-----------|------|------|
| `modelopt.torch` | `modelopt/torch/` | Core PyTorch optimization library |
| `modelopt.onnx` | `modelopt/onnx/` | ONNX model quantization and export |
| `modelopt.deploy` | `modelopt/deploy/` | Deployment utilities for LLMs |

### `modelopt.torch` Sub-packages

| Sub-package | Path | Role |
|-------------|------|------|
| `opt` | `modelopt/torch/opt/` | Core optimization infrastructure (modes, config, state dicts) |
| `quantization` | `modelopt/torch/quantization/` | PTQ, QAT, and quantization-aware algorithms |
| `prune` | `modelopt/torch/prune/` | Structured and unstructured pruning |
| `distill` | `modelopt/torch/distill/` | Knowledge distillation |
| `sparsity` | `modelopt/torch/sparsity/` | Weight and activation sparsity |
| `speculative` | `modelopt/torch/speculative/` | Speculative decoding (Medusa, EAGLE, etc.) |
| `nas` | `modelopt/torch/nas/` | Neural architecture search |
| `export` | `modelopt/torch/export/` | Checkpoint export for TRT-LLM / Megatron |
| `peft` | `modelopt/torch/peft/` | QLoRA and PEFT integration |
| `_deploy` | `modelopt/torch/_deploy/` | Internal deployment utilities |
| `utils` | `modelopt/torch/utils/` | Shared utilities and plugin infrastructure |

### Core Abstraction: Modes

A **mode** is the unit of model optimization in ModelOpt. Each algorithm (quantization, pruning,
etc.) is implemented as one or more modes. Modes are recorded in the model's `modelopt_state` so
optimization workflows can be composed, saved, and restored.

### Export Flow

```text
PyTorch/HF Model → modelopt.torch (optimize) → export checkpoint
    → TensorRT-LLM / vLLM / SGLang / TensorRT (deploy)
```

Unified HF export (`unified_export_hf.py`) supports both transformers and diffusers models.

## Key Files

| File | Role |
|------|------|
| `modelopt/torch/opt/mode.py` | Base class for all optimization modes |
| `modelopt/torch/opt/config.py` | Configuration system for modes |
| `modelopt/torch/opt/conversion.py` | `apply_mode()` / `restore()` entry points |
| `modelopt/torch/quantization/__init__.py` | PTQ/QAT public API |
| `modelopt/torch/export/unified_export_hf.py` | Unified HF checkpoint export |
| `modelopt/torch/export/model_config_export.py` | TRT-LLM model config export |
| `modelopt/deploy/llm/` | LLM deployment utilities |
| `pyproject.toml` | Optional dependency groups (`[onnx]`, `[hf]`, `[all]`, `[dev]`); ruff, mypy, pytest, bandit, and coverage config |
| `.pre-commit-config.yaml` | Pre-commit hooks (ruff, mypy, clang-format, license headers) |
| `tox.ini` | Test environment definitions |

## Design Patterns

| Pattern | Key Points |
|---------|------------|
| **Mode composition** | Optimization algorithms are composed as sequences of modes, each recorded in `modelopt_state` |
| **Plugin system** | Optional integrations (HuggingFace, Megatron, etc.) loaded lazily via `import_plugin()` |
| **Optional dependencies** | Features gated by install extras (`[onnx]`, `[hf]`, `[all]`); avoid hard imports at module level |
| **Config dataclasses** | Each mode has a typed config; use Pydantic or dataclass conventions |
| **State dict** | Models carry `modelopt_state` for checkpoint save/restore across optimization steps |

## Anti-Patterns / Gotchas

- **Pre-commit modifies files in-place** — if hooks fail, files are already fixed. Re-stage
  (`git add`) and commit again.
- **License header required** — all new `.py`, `.cpp`, `.cu`, `.sh` files need the NVIDIA Apache
  2.0 header from `LICENSE_HEADER`. Pre-commit will add it automatically, but re-stage after.
- **Optional imports must be guarded** — use `import_plugin()` or `try/except ImportError` for
  anything in `[onnx]`, `[hf]`, or other optional extras.
- **CPU unit tests must be fast** — `tests/unit/` should run in seconds (no GPU, no model weights).
  GPU tests go in `tests/gpu/`.
- **Coverage requirement** — Codecov check in PRs must pass (70% threshold on `modelopt/*`).
  New features need corresponding tests.
- **Rebase, don't merge** — always rebase onto `main`; force-push with `--force-with-lease`.
- **Sign every commit** — both DCO (`-s`) and cryptographic (`-S`) signing are required.
  Configure SSH signing key per `CONTRIBUTING.md`.
- **Changelog** — update `CHANGELOG.rst` for new features, API changes, critical fixes, or
  backward-incompatible changes.

## Security Coding Practices

> For full security policy, vulnerability reporting, and escalation paths, see `SECURITY.md`.

ModelOpt loads user-supplied model checkpoints and weights. These patterns are **forbidden** in all
non-test code and are enforced as blocking pre-merge errors by CodeRabbit and Bandit:

| Forbidden pattern | Rule |
|-------------------|------|
| `torch.load(..., weights_only=False)` | Forbidden unless an inline comment confirms the file is internally generated (not user-supplied) |
| `numpy.load(..., allow_pickle=True)` | Expose `allow_pickle` as a caller parameter defaulting to `False`; never hardcode `True` |
| `yaml.load()` | Always use `yaml.safe_load()` |
| `trust_remote_code=True` hardcoded | Expose as a caller parameter defaulting to `False`; never hardcode `True` |
| `subprocess.run(..., shell=True)` with string interpolation | Command-injection risk — pass args as a list instead |
| `eval()` / `exec()` on external input | Never on data that could originate outside the process |
| `# nosec` comments | Not allowed as a Bandit bypass — see exception process below |
| Hardcoded secrets / credentials | Never commit tokens, passwords, or API keys; use env vars |

**Security exception process:** If a sensitive pattern is genuinely required, add an inline comment
explaining why it is safe in this specific context, then request review from
`@NVIDIA/modelopt-setup-codeowners` with explicit justification in the PR description.

## Third-Party Code (IP)

Copying code from external sources requires OSRB (Open Source Review Board) authorization before
merging:

- **External contributors:** contact `@NVIDIA/modelopt-setup-codeowners` for guidance.
- **Internal contributors:** clone NVBug 2885977. Permissive licenses (MIT, Apache 2) are generally
  self-checkout; other licenses require expert review.
- **License header format** for copied files (in order): (1) source URL with commit hash,
  (2) original copyright/license, (3) NVIDIA Apache 2.0 header. See
  `modelopt/torch/speculative/eagle/utils.py` for an example.
- **Exclude from license hook:** add the file path to the `exclude` list in the `insert-license`
  hook in `.pre-commit-config.yaml` so the pre-commit hook does not prepend a second NVIDIA header.

## Development Workflow

> For detailed contribution guidelines, commit conventions, and PR requirements, see `CONTRIBUTING.md`.

1. Install in editable mode: `pip install -e ".[dev]"`
2. Install pre-commit hooks: `pre-commit install`
3. Make changes following project conventions
4. Run relevant tests: `pytest tests/unit` (CPU) or `pytest tests/gpu` (GPU)
5. Rebase onto `main`: `git rebase origin/main`
6. Commit with sign-off: `git commit -s -S -m "description"`
7. Push: `git push origin <branch> --force-with-lease`
8. Submit PR — fill in `.github/PULL_REQUEST_TEMPLATE.md` (type, overview, testing, changelog)

## CI / Testing

| Layer | Location | Notes |
|-------|----------|-------|
| CPU unit tests | `tests/unit/` | Fast, no GPU needed; run in pre-merge CI |
| GPU unit tests | `tests/gpu/` | Requires CUDA GPU |
| Megatron GPU tests | `tests/gpu_megatron/` | Requires Megatron-Core + GPU |
| TRT-LLM GPU tests | `tests/gpu_trtllm/` | Requires TensorRT-LLM + GPU |
| Example/integration tests | `tests/examples/` | Integration tests for examples; see `tests/examples/README.md` |
| Pre-commit / lint | `.pre-commit-config.yaml` | ruff, mypy, clang-format, license headers, bandit |
| Coverage | `pyproject.toml` | 70% minimum on `modelopt/*` |

## Key Documentation

| Topic | Path / URL |
|-------|------------|
| Full documentation | <https://nvidia.github.io/Model-Optimizer> |
| Installation guide | `docs/` or <https://nvidia.github.io/Model-Optimizer/getting_started/2_installation.html> |
| Quantization guide | <https://nvidia.github.io/Model-Optimizer/guides/1_quantization.html> |
| Pruning guide | <https://nvidia.github.io/Model-Optimizer/guides/3_pruning.html> |
| Distillation guide | <https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html> |
| Speculative decoding | <https://nvidia.github.io/Model-Optimizer/guides/5_speculative_decoding.html> |
| Sparsity guide | <https://nvidia.github.io/Model-Optimizer/guides/6_sparsity.html> |
| LLM PTQ examples | `examples/llm_ptq/` |
| Diffusers examples | `examples/diffusers/` |
| Contributing guide | `CONTRIBUTING.md` |
| Changelog | `CHANGELOG.rst` |
