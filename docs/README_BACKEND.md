# Backend Library README

`src/backend.py` centralizes backend selection and runtime configuration for:
- PyTorch training (`cpu`, `cuda`, `rocm`, `mps` via `auto`)
- ONNX Runtime inference providers
- Automatic mixed precision policy

## What This Module Does

The backend module solves four problems:
1. Resolve the PyTorch device from a requested backend.
2. Apply backend-specific tuning knobs (CUDA/ROCm/Metal).
3. Resolve AMP policy used by training.
4. Map requested backend to ONNX Runtime providers.

## Supported Backend Values

- `auto`: prefers GPU (`cuda`) when available, otherwise `mps`, otherwise `cpu`
- `cpu`: force CPU
- `cuda`: force CUDA build/device
- `rocm`: force ROCm build/device

Notes:
- In PyTorch, both CUDA and ROCm use `torch.device("cuda")`.
- ROCm is detected through `torch.version.hip`.

## Public API

### `resolve_torch_device(backend)`

Returns a `torch.device` for training.

- `auto` -> `cuda` > `mps` > `cpu`
- `cuda` -> error if ROCm build is detected
- `rocm` -> error if CUDA build is detected

Raises `BackendError` on invalid/unavailable choices.

### `configure_torch_backend(backend, device, verbose=True)`

Applies runtime tuning based on resolved device:

- CUDA:
  - enables `cudnn.benchmark`
  - enables TF32 flags (when available)
  - sets float32 matmul precision to `"high"` (when available)
- ROCm:
  - keeps TF32 disabled
  - sets float32 matmul precision to `"high"` (when available)
- Metal/MPS:
  - sets float32 matmul precision to `"high"` (when available)

If `verbose=True`, prints backend diagnostics (versions, device info, memory usage, tuning flags).

### `resolve_amp_config(device)`

Returns AMP policy as a dictionary:
- `enabled`: whether autocast should be enabled
- `dtype`: autocast dtype (`torch.float16` on GPU/MPS, `None` on CPU)
- `use_grad_scaler`: `True` for CUDA/ROCm, `False` for MPS/CPU

### `resolve_ort_providers(backend, available)`

Maps backend request to ONNX Runtime provider list with CPU fallback:
- `auto` -> CUDA provider, else ROCM, else CoreML, else CPU
- `cuda` -> requires `CUDAExecutionProvider`
- `rocm` -> requires `ROCMExecutionProvider`
- `cpu` -> CPU only

Raises `BackendError` if a requested provider is not available.

## Minimal Integration Example

```python
# --- Imports ---------------------------------------------------------------
# Backend utilities used by training code.
from backend import (
    BackendError,
    configure_torch_backend,
    resolve_amp_config,
    resolve_torch_device,
)

# --- Device resolution -----------------------------------------------------
# Resolve and validate the execution device.
backend = "auto"
try:
    device = resolve_torch_device(backend)
except BackendError as exc:
    raise SystemExit(f"Invalid backend: {exc}") from exc

# --- Runtime tuning --------------------------------------------------------
# Apply backend-specific tuning and print diagnostics.
configure_torch_backend(backend, device, verbose=True)

# --- AMP policy ------------------------------------------------------------
# Read AMP/autocast/scaler policy for the selected device.
amp_config = resolve_amp_config(device)
```

## Error Model

The module uses `BackendError` (a `RuntimeError`) for:
- unknown backend value
- backend mismatch (for example asking `cuda` on ROCm build)
- unavailable ONNX Runtime execution provider
- missing/unimportable PyTorch

## Design Notes

- Backend decisions are explicit and deterministic.
- Backend checks are isolated in one module to avoid duplication.
- Training and inference can share the same backend contract.
