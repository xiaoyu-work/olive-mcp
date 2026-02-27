"""Olive MCP Server - Model optimization via Microsoft Olive."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

mcp = FastMCP(
    name="olive",
    instructions=(
        "Olive MCP server for model optimization. Use `list_supported_configs` first to understand "
        "available options, then use `optimize_model` for end-to-end optimization, `quantize_model` "
        "for quantization only, `finetune_model` for LoRA/QLoRA fine-tuning, or `benchmark_model` "
        "for evaluation."
    ),
)

OUTPUT_BASE = Path.home() / ".olive-mcp" / "outputs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DEVICES = ["cpu", "gpu", "npu"]

SUPPORTED_PROVIDERS = [
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "TensorrtExecutionProvider",
    "ROCMExecutionProvider",
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
    "WebGpuExecutionProvider",
    "NvTensorRTRTXExecutionProvider",
]

SUPPORTED_PRECISIONS = [
    "fp32", "fp16", "bf16",
    "int4", "int8", "int16", "int32",
    "uint4", "uint8", "uint16", "uint32",
]

SUPPORTED_ALGORITHMS = ["rtn", "gptq", "awq", "hqq"]

DEVICE_TO_DEFAULT_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "npu": "QNNExecutionProvider",
}

PROVIDER_PACKAGE_MAPPING = {
    "CPUExecutionProvider": "onnxruntime",
    "CUDAExecutionProvider": "onnxruntime-gpu",
    "TensorrtExecutionProvider": "onnxruntime-gpu",
    "ROCMExecutionProvider": "onnxruntime-gpu",
    "OpenVINOExecutionProvider": "onnxruntime-openvino",
    "DmlExecutionProvider": "onnxruntime-directml",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output_path(prefix: str, model_name: str) -> str:
    """Generate a unique output path under ~/.olive-mcp/outputs/."""
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_BASE / f"{prefix}_{safe_name}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _parse_output_dir(output_path: str) -> dict:
    """Parse Olive output directory for model files and configs."""
    out = Path(output_path)
    result = {"output_path": output_path}

    # Look for model files
    for ext in ("*.onnx", "*.pt", "*.safetensors", "*.bin"):
        files = list(out.rglob(ext))
        if files:
            result["model_files"] = [str(f) for f in files]
            break

    # Look for config files
    for name in ("model_config.json", "config.json"):
        cfg = out / name
        if cfg.exists():
            try:
                with open(cfg) as f:
                    result["model_config"] = json.load(f)
            except json.JSONDecodeError:
                pass
            break

    # Look for inference config
    inf_cfg = out / "inference_config.json"
    if inf_cfg.exists():
        try:
            with open(inf_cfg) as f:
                result["inference_config"] = json.load(f)
        except json.JSONDecodeError:
            pass

    return result


async def _run_olive_cli(
    cmd: list[str],
    ctx: Context[ServerSession, None],
    label: str,
) -> tuple[int, str, str]:
    """Run an Olive CLI command in a subprocess with progress updates."""
    await ctx.info(f"[olive-mcp] Starting: {label}")
    await ctx.info(f"[olive-mcp] Command: olive {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        "olive",
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()
    stdout_str = stdout.decode("utf-8", errors="replace")
    stderr_str = stderr.decode("utf-8", errors="replace")

    if process.returncode == 0:
        await ctx.info(f"[olive-mcp] Completed: {label}")
    else:
        await ctx.info(f"[olive-mcp] Failed (exit code {process.returncode}): {label}")

    return process.returncode, stdout_str, stderr_str


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def list_supported_configs() -> dict:
    """List all supported devices, execution providers, precisions, and quantization algorithms.

    Call this first to understand what options are available before running optimization.
    """
    return {
        "devices": SUPPORTED_DEVICES,
        "providers": SUPPORTED_PROVIDERS,
        "precisions": SUPPORTED_PRECISIONS,
        "quantization_algorithms": SUPPORTED_ALGORITHMS,
        "device_to_default_provider": DEVICE_TO_DEFAULT_PROVIDER,
        "provider_package_mapping": PROVIDER_PACKAGE_MAPPING,
        "notes": {
            "device": "Use 'cpu', 'gpu', or 'npu'. Provider is auto-selected if not specified.",
            "precision": "Common choices: 'int4' (smallest, fast), 'int8' (balanced), 'fp16' (high quality).",
            "provider": "Specific execution provider. Auto-detected from device if omitted.",
        },
    }


@mcp.tool()
async def optimize_model(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    device: str = "cpu",
    provider: str | None = None,
    precision: str = "int4",
    output_path: str | None = None,
) -> dict:
    """Optimize a model end-to-end using Olive's auto-optimization pipeline.

    This is the main tool. It automatically selects the best optimization passes
    (capture, convert, quantize, graph optimize) based on the model, device, and precision.

    Args:
        model_name_or_path: HuggingFace model name (e.g. "microsoft/Phi-3-mini-4k-instruct") or local path.
        device: Target device - "cpu", "gpu", or "npu".
        provider: Execution provider (e.g. "CUDAExecutionProvider"). Auto-detected from device if omitted.
        precision: Target precision - "fp16", "int4", "int8", etc.
        output_path: Directory to save optimized model. Auto-generated if omitted.
    """
    if not provider:
        provider = DEVICE_TO_DEFAULT_PROVIDER.get(device, "CPUExecutionProvider")

    if not output_path:
        output_path = _make_output_path("optimize", model_name_or_path)

    cmd = [
        "optimize",
        "--model_name_or_path", model_name_or_path,
        "--provider", provider,
        "--precision", precision,
        "--output_path", output_path,
        "--log_level", "1",
    ]
    if device:
        cmd.extend(["--device", device])

    returncode, stdout, stderr = await _run_olive_cli(
        cmd, ctx, f"optimize {model_name_or_path} → {device}/{provider}/{precision}"
    )

    if returncode != 0:
        return {
            "status": "error",
            "returncode": returncode,
            "error": stderr[-2000:] if len(stderr) > 2000 else stderr,
            "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
        }

    result = _parse_output_dir(output_path)
    result["status"] = "success"
    result["device"] = device
    result["provider"] = provider
    result["precision"] = precision
    return result


@mcp.tool()
async def quantize_model(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    precision: str = "int4",
    algorithm: str = "rtn",
    implementation: str = "olive",
    output_path: str | None = None,
) -> dict:
    """Quantize a model to reduce size and improve inference speed.

    Args:
        model_name_or_path: HuggingFace model name or local path (ONNX or PyTorch).
        precision: Target precision - "int4", "int8", etc.
        algorithm: Quantization algorithm - "rtn" (fast, no calibration), "gptq" (better quality), "awq" (activation-aware).
        implementation: Implementation backend - "olive", "ort", or "bnb".
        output_path: Directory to save quantized model. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("quantize", model_name_or_path)

    cmd = [
        "quantize",
        "--model_name_or_path", model_name_or_path,
        "--algorithm", algorithm,
        "--precision", precision,
        "--implementation", implementation,
        "--output_path", output_path,
        "--log_level", "1",
    ]

    returncode, stdout, stderr = await _run_olive_cli(
        cmd, ctx, f"quantize {model_name_or_path} → {algorithm}/{precision}"
    )

    if returncode != 0:
        return {
            "status": "error",
            "returncode": returncode,
            "error": stderr[-2000:] if len(stderr) > 2000 else stderr,
            "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
        }

    result = _parse_output_dir(output_path)
    result["status"] = "success"
    result["algorithm"] = algorithm
    result["precision"] = precision
    return result


@mcp.tool()
async def finetune_model(
    model_name_or_path: str,
    data_name: str,
    ctx: Context[ServerSession, None],
    method: str = "lora",
    max_steps: int = 100,
    output_path: str | None = None,
) -> dict:
    """Fine-tune a model using LoRA or QLoRA for a specific dataset.

    Args:
        model_name_or_path: HuggingFace model name (e.g. "microsoft/Phi-3-mini-4k-instruct").
        data_name: HuggingFace dataset name (e.g. "nampdn-ai/tiny-codes").
        method: Fine-tuning method - "lora" or "qlora" (4-bit quantized LoRA, uses less memory).
        max_steps: Maximum training steps.
        output_path: Directory to save fine-tuned adapter. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("finetune", model_name_or_path)

    cmd = [
        "finetune",
        "--model_name_or_path", model_name_or_path,
        "--data_name", data_name,
        "--method", method,
        "--max_steps", str(max_steps),
        "--output_path", output_path,
        "--log_level", "1",
    ]

    returncode, stdout, stderr = await _run_olive_cli(
        cmd, ctx, f"finetune {model_name_or_path} with {method} on {data_name}"
    )

    if returncode != 0:
        return {
            "status": "error",
            "returncode": returncode,
            "error": stderr[-2000:] if len(stderr) > 2000 else stderr,
            "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
        }

    result = _parse_output_dir(output_path)
    result["status"] = "success"
    result["method"] = method
    result["data_name"] = data_name
    return result


@mcp.tool()
async def benchmark_model(
    model_name_or_path: str,
    tasks: list[str],
    ctx: Context[ServerSession, None],
    device: str = "cpu",
    batch_size: int = 1,
    limit: float = 0.1,
    output_path: str | None = None,
) -> dict:
    """Benchmark/evaluate a model using lm-eval tasks.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        tasks: List of lm-eval tasks to run (e.g. ["hellaswag", "mmlu"]).
        device: Device for evaluation - "cpu" or "gpu".
        batch_size: Evaluation batch size.
        limit: Fraction of samples to use (0.0-1.0) or absolute number. Use small values for quick checks.
        output_path: Directory to save results. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("benchmark", model_name_or_path)

    cmd = [
        "benchmark",
        "--model_name_or_path", model_name_or_path,
        "--device", device,
        "--batch_size", str(batch_size),
        "--limit", str(limit),
        "--output_path", output_path,
        "--log_level", "1",
    ]
    for task in tasks:
        cmd.extend(["--tasks", task])

    returncode, stdout, stderr = await _run_olive_cli(
        cmd, ctx, f"benchmark {model_name_or_path} on {', '.join(tasks)}"
    )

    if returncode != 0:
        return {
            "status": "error",
            "returncode": returncode,
            "error": stderr[-2000:] if len(stderr) > 2000 else stderr,
            "stdout": stdout[-1000:] if len(stdout) > 1000 else stdout,
        }

    result = _parse_output_dir(output_path)
    result["status"] = "success"
    result["tasks"] = tasks
    result["device"] = device
    return result
