"""Backend selection (PyTorch training / ONNX Runtime inference)."""

# --- Import ---------------------------------------------------------------
# Imports for CLI parser and hardware/software backend checks.

import argparse
import importlib


# --- Public types ---------------------------------------------------------
# Exposed classes to select and validate the backend.

class BackendError(RuntimeError):
    # 1) Explicit error for incompatible backend
    pass


# --- CLI arguments --------------------------------------------------------
# Argument parser definition (consistent with other modules).

def build_parser():
    parser = argparse.ArgumentParser(
        prog="backend",
        description="Utility for selecting training/inference backend.",
    )
    return parser


# --- Support: import ------------------------------------------------------
# Utility for lazy PyTorch import.

def _import_torch():
    # Lazy import: avoid requiring PyTorch when using ONNX inference only.
    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise BackendError("PyTorch is not installed or cannot be imported.") from exc


# --- PyTorch device resolution -------------------------------------------
# Select target device based on requested backend.

def resolve_torch_device(backend):
    torch = _import_torch()
    has_cuda = torch.cuda.is_available()
    has_rocm_build = torch.version.hip is not None
    has_mps = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    # 1) Choose device based on requested backend
    if backend == "auto":
        return (
            torch.device("cuda")
            if has_cuda
            else torch.device("mps")
            if has_mps
            else torch.device("cpu")
        )

    if backend == "cpu":
        return torch.device("cpu")

    if backend == "cuda":
        if has_cuda and has_rocm_build:
            raise BackendError("ROCm build detected: use 'rocm' backend.")
        if has_cuda:
            return torch.device("cuda")
        raise BackendError("CUDA is not available.")

    if backend == "rocm":
        if has_cuda and has_rocm_build:
            return torch.device("cuda")
        if has_cuda:
            raise BackendError("CUDA build detected: use 'cuda' backend.")
        raise BackendError("ROCm is not available.")

    raise BackendError(f"Unknown backend: {backend}")


# --- PyTorch backend configuration ---------------------------------------
# Apply automatic tuning and print diagnostics (if verbose).

def configure_torch_backend(
    backend,
    device,
    verbose=True,
):
    torch = _import_torch()
    is_cuda = device.type == "cuda"
    is_rocm = is_cuda and torch.version.hip is not None
    is_mps = (
        device.type == "mps"
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    matmul_precision = "default"

    # --- Automatic tuning --------------------------------------------------
    # Automatic configuration to improve training performance.
    if is_cuda and not is_rocm:
        # 1) CUDA tuning block
        # Enable cuDNN autotuner and TF32 on CUDA builds.
        torch.backends.cudnn.benchmark = True
        if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
            matmul_precision = "high"
    elif is_rocm:
        # 2) ROCm tuning block
        # Keep TF32 disabled and apply matmul precision policy.
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
            matmul_precision = "high"
    elif is_mps:
        # 3) Metal (MPS) tuning block
        # Use a higher matmul precision policy when available.
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
            matmul_precision = "high"

    if not verbose:
        return

    # --- Diagnostics -------------------------------------------------------
    # Collect and print useful info about the active backend.
    lines = []
    lines.append(f"[backend] requested={backend} resolved={device.type}")
    lines.append(f"[backend] torch={torch.__version__}")
    lines.append(
        f"[backend] cuda={torch.version.cuda} hip={torch.version.hip} "
        f"mps={getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}"
    )
    if is_cuda and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024**3)
        alloc_gb = torch.cuda.memory_allocated(idx) / (1024**3)
        reserv_gb = torch.cuda.memory_reserved(idx) / (1024**3)
        # CUDA/ROCm diagnostics block
        lines.append(
            f"[backend] device={props.name} cc={props.major}.{props.minor} "
            f"total={total_gb:.2f}GB alloc={alloc_gb:.2f}GB reserved={reserv_gb:.2f}GB"
        )
        lines.append(
            f"[backend] cudnn.benchmark={torch.backends.cudnn.benchmark} "
            f"tf32.matmul={getattr(torch.backends.cuda.matmul, 'allow_tf32', 'n/a')} "
            f"tf32.cudnn={getattr(torch.backends.cudnn, 'allow_tf32', 'n/a')} "
            f"matmul_precision={matmul_precision}"
        )
    elif (
        device.type == "mps"
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        # MPS diagnostics block
        lines.append("[backend] device=Apple MPS")
        lines.append("[backend] mps available")
    else:
        # CPU diagnostics block
        lines.append("[backend] cuda not available")

    print("\n".join(lines))


# --- ONNX Runtime provider resolution ------------------------------------
# Map requested backend to available ORT providers.

def resolve_ort_providers(backend, available):
    # 1) Resolve ORT providers with CPU fallback
    if backend == "auto":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "ROCMExecutionProvider" in available:
            return ["ROCMExecutionProvider", "CPUExecutionProvider"]
        if "CoreMLExecutionProvider" in available:
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    if backend == "cpu":
        return ["CPUExecutionProvider"]

    if backend == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise BackendError("CUDAExecutionProvider is not available in ONNX Runtime.")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if backend == "rocm":
        if "ROCMExecutionProvider" not in available:
            raise BackendError("ROCMExecutionProvider is not available in ONNX Runtime.")
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]

    raise BackendError(f"Unknown backend: {backend}")


# Note: this module is intended as a support library, without a script entrypoint.
