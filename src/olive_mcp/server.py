"""Olive MCP Server - Model optimization via Microsoft Olive Python API."""

import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession

mcp = FastMCP(
    name="olive",
    instructions="""Olive MCP server for Microsoft Olive model optimization.

## Choosing parameters based on user intent
When the user describes their goal instead of specific parameters, choose accordingly:
- **Smallest model / fastest inference / edge deployment** → precision="int4", algorithm="gptq" or "awq"
- **Balanced size and quality** → precision="int8"
- **Best quality / minimal degradation** → precision="fp16"
- **Training / fine-tuning** → suggest `finetune` with method="qlora" (less memory) or "lora"
- **Just convert to ONNX** → use `capture_onnx_graph`
- **Deploy to specific hardware** → match provider: GPU→CUDAExecutionProvider, NPU→QNNExecutionProvider, DirectML→DmlExecutionProvider

## Popular model recommendations
When the user doesn't specify a model, suggest based on their use case:
- **Text chat / general LLM**: microsoft/Phi-4-mini-instruct (small, fast), microsoft/Phi-4 (powerful)
- **Code generation**: microsoft/Phi-4-mini-instruct
- **Image generation**: runwayml/stable-diffusion-v1-5 (SD 1.5), stabilityai/stable-diffusion-xl-base-1.0 (SDXL), black-forest-labs/FLUX.1-dev (Flux)
- **Embedding / retrieval**: BAAI/bge-small-en-v1.5, sentence-transformers/all-MiniLM-L6-v2
- **Vision + language**: microsoft/Phi-4-multimodal-instruct

## Workflow
1. If the user is a beginner or unsure, ask what they want to achieve (chat, code, images, etc.) and recommend a model + settings.
2. For most optimization tasks, `optimize` is the recommended starting point — it auto-selects the best passes.
3. Use `quantize` only when the user wants fine-grained control over quantization algorithm/implementation.
4. After optimization, suggest `benchmark` to evaluate quality, or use `list_outputs` to review past results.
5. When the user asks to compare results, use `list_outputs` to find previous runs.

## Multi-step workflow guidance
- **Full pipeline**: optimize → benchmark → compare with original
- **Fine-tune then deploy**: finetune → optimize (with the fine-tuned model) → benchmark
- **Diffusion LoRA**: diffusion_lora → convert_adapters (if ONNX deployment needed)
- **ONNX deployment**: capture_onnx_graph → tune_session_params → benchmark
""",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VENV_BASE = Path.home() / ".olive-mcp" / "venvs"
OUTPUT_BASE = Path.home() / ".olive-mcp" / "outputs"
WORKER_PATH = Path(__file__).parent / "worker.py"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_PACKAGES = ["olive-ai"]

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

SUPPORTED_QUANT_ALGORITHMS = ["rtn", "gptq", "awq", "hqq"]

SUPPORTED_QUANT_IMPLEMENTATIONS = ["olive", "ort", "bnb", "nvmo", "inc", "spinquant", "quarot", "awq", "autogptq"]

PROVIDER_TO_ORT = {
    "CPUExecutionProvider": "onnxruntime",
    "CUDAExecutionProvider": "onnxruntime-gpu",
    "TensorrtExecutionProvider": "onnxruntime-gpu",
    "ROCMExecutionProvider": "onnxruntime-gpu",
    "OpenVINOExecutionProvider": "onnxruntime-openvino",
    "DmlExecutionProvider": "onnxruntime-directml",
    "QNNExecutionProvider": "onnxruntime",
    "VitisAIExecutionProvider": "onnxruntime",
    "WebGpuExecutionProvider": "onnxruntime",
    "NvTensorRTRTXExecutionProvider": "onnxruntime-gpu",
}

DEVICE_TO_DEFAULT_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "npu": "QNNExecutionProvider",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output_path(prefix: str, model_name: str) -> str:
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_BASE / f"{prefix}_{safe_name}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _get_python_path(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _build_kwargs(**kw) -> dict:
    """Filter out None values from kwargs."""
    return {k: v for k, v in kw.items() if v is not None}


# ---------------------------------------------------------------------------
# Venv management
# ---------------------------------------------------------------------------


async def _get_or_create_venv(
    packages: list[str],
    ctx: Context[ServerSession, None],
) -> Path:
    """Get or create a cached uv venv with the specified packages."""
    key = hashlib.md5("|".join(sorted(packages)).encode()).hexdigest()[:12]
    venv_path = VENV_BASE / key
    python_path = _get_python_path(venv_path)

    if not python_path.exists():
        await ctx.info(f"[olive-mcp] Creating venv with: {', '.join(packages)}")
        VENV_BASE.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            "uv", "venv", str(venv_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create venv: {stderr.decode()}")

        proc = await asyncio.create_subprocess_exec(
            "uv", "pip", "install", "--python", str(python_path), *packages,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to install packages: {stderr.decode()}")

        await ctx.info("[olive-mcp] Venv ready")
    else:
        await ctx.info(f"[olive-mcp] Reusing cached venv ({key})")

    return python_path


# ---------------------------------------------------------------------------
# Worker execution
# ---------------------------------------------------------------------------


async def _run_olive(
    command: str,
    kwargs: dict,
    extra_packages: list[str],
    ctx: Context[ServerSession, None],
) -> dict:
    """Run an olive command in an isolated venv via worker.py."""
    packages = [*BASE_PACKAGES, *extra_packages]
    python_path = await _get_or_create_venv(packages, ctx)

    await ctx.info(f"[olive-mcp] Running: {command}")

    proc = await asyncio.create_subprocess_exec(
        str(python_path), str(WORKER_PATH), command, json.dumps(kwargs),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_bytes, stderr_bytes = await proc.communicate()
    stdout_str = stdout_bytes.decode("utf-8", errors="replace")
    stderr_str = stderr_bytes.decode("utf-8", errors="replace")

    if proc.returncode != 0:
        await ctx.info(f"[olive-mcp] Failed: {command} (exit code {proc.returncode})")
        return {
            "status": "error",
            "returncode": proc.returncode,
            "error": stderr_str[-3000:],
            "stdout": stdout_str[-1000:],
        }

    await ctx.info(f"[olive-mcp] Completed: {command}")

    try:
        return json.loads(stdout_str)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error": "Failed to parse worker JSON output",
            "stdout": stdout_str[-3000:],
            "stderr": stderr_str[-1000:],
        }


# ---------------------------------------------------------------------------
# MCP Tools
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
        "quantization_algorithms": SUPPORTED_QUANT_ALGORITHMS,
        "quantization_implementations": SUPPORTED_QUANT_IMPLEMENTATIONS,
        "device_to_default_provider": DEVICE_TO_DEFAULT_PROVIDER,
        "provider_to_ort_package": PROVIDER_TO_ORT,
    }


@mcp.tool()
async def optimize(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    provider: str = "CPUExecutionProvider",
    device: str | None = None,
    precision: str = "fp32",
    act_precision: str | None = None,
    exporter: str | None = None,
    use_qdq_format: bool = False,
    num_split: int | None = None,
    memory: int | None = None,
    block_size: int | None = None,
    surgeries: list[str] | None = None,
    output_path: str | None = None,
) -> dict:
    """Optimize a model end-to-end using Olive's auto-optimization pipeline.

    Automatically selects the best passes (capture, convert, quantize, graph optimize)
    based on model, device, and precision.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        provider: Execution provider (e.g. "CUDAExecutionProvider"). Default: CPUExecutionProvider.
        device: Target device - "cpu", "gpu", or "npu". Auto-detected from provider if omitted.
        precision: Target precision - "fp32", "fp16", "int4", "int8", etc.
        act_precision: Activation precision for quantization (optional).
        exporter: Model exporter - "model_builder", "dynamo_exporter", "torchscript_exporter", "optimum_exporter".
        use_qdq_format: Use QDQ format for quantization instead of QOperator.
        num_split: Number of splits for model splitting.
        memory: Available device memory in MB.
        block_size: Block size for quantization (-1 for per-channel).
        surgeries: List of graph surgeries to apply.
        output_path: Directory to save optimized model. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("optimize", model_name_or_path)

    ort_package = PROVIDER_TO_ORT.get(provider, "onnxruntime")
    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        provider=provider,
        device=device,
        precision=precision,
        act_precision=act_precision,
        exporter=exporter,
        use_qdq_format=use_qdq_format if use_qdq_format else None,
        num_split=num_split,
        memory=memory,
        block_size=block_size,
        surgeries=surgeries,
        output_path=output_path,
    )
    return await _run_olive("optimize", kwargs, [ort_package], ctx)


@mcp.tool()
async def quantize(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    algorithm: str = "rtn",
    precision: str = "int8",
    act_precision: str = "int8",
    implementation: str = "olive",
    use_qdq_encoding: bool = False,
    data_name: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Quantize a model to reduce size and improve inference speed.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        algorithm: Quantization algorithm - "rtn", "gptq", "awq", "hqq".
        precision: Target precision - "int4", "int8", etc.
        act_precision: Activation precision for static quantization.
        implementation: Backend - "olive", "ort", "bnb", "nvmo", "inc", etc.
        use_qdq_encoding: Use QDQ encoding in ONNX model.
        data_name: HuggingFace dataset name for calibration (required by some algorithms).
        output_path: Directory to save quantized model. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("quantize", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        algorithm=algorithm,
        precision=precision,
        act_precision=act_precision,
        implementation=implementation,
        use_qdq_encoding=use_qdq_encoding if use_qdq_encoding else None,
        data_name=data_name,
        output_path=output_path,
    )
    return await _run_olive("quantize", kwargs, ["onnxruntime"], ctx)


@mcp.tool()
async def finetune(
    model_name_or_path: str,
    data_name: str,
    ctx: Context[ServerSession, None],
    method: str = "lora",
    lora_r: int = 64,
    lora_alpha: int = 16,
    target_modules: str | None = None,
    torch_dtype: str = "bfloat16",
    train_split: str = "train",
    eval_split: str | None = None,
    output_path: str | None = None,
) -> dict:
    """Fine-tune a model using LoRA or QLoRA.

    Args:
        model_name_or_path: HuggingFace model name (e.g. "microsoft/Phi-3-mini-4k-instruct").
        data_name: HuggingFace dataset name (e.g. "nampdn-ai/tiny-codes").
        method: Fine-tuning method - "lora" or "qlora" (4-bit quantized, less memory).
        lora_r: LoRA rank. Default: 64.
        lora_alpha: LoRA alpha scaling. Default: 16.
        target_modules: Comma-separated target modules for LoRA.
        torch_dtype: Torch dtype for training - "bfloat16", "float16", "float32".
        train_split: Dataset split for training. Default: "train".
        eval_split: Dataset split for evaluation (optional).
        output_path: Directory to save fine-tuned adapter. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("finetune", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        data_name=data_name,
        method=method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        torch_dtype=torch_dtype,
        train_split=train_split,
        eval_split=eval_split,
        output_path=output_path,
    )
    return await _run_olive("finetune", kwargs, [], ctx)


@mcp.tool()
async def capture_onnx_graph(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    use_model_builder: bool = False,
    use_dynamo_exporter: bool = False,
    precision: str = "fp16",
    conversion_device: str = "cpu",
    torch_dtype: str | None = None,
    target_opset: int = 20,
    use_ort_genai: bool = False,
    output_path: str | None = None,
) -> dict:
    """Capture ONNX graph from a HuggingFace or PyTorch model.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        use_model_builder: Use Model Builder to capture ONNX model.
        use_dynamo_exporter: Use dynamo_export API to export ONNX model.
        precision: Precision for Model Builder - "fp16", "fp32", "int4", "bf16".
        conversion_device: Device for conversion - "cpu" or "gpu".
        torch_dtype: Dtype to cast model before capture (e.g. "float16").
        target_opset: Target ONNX opset version. Default: 20.
        use_ort_genai: Use ORT generate() API to run the model.
        output_path: Directory to save ONNX model. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("capture", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        use_model_builder=use_model_builder if use_model_builder else None,
        use_dynamo_exporter=use_dynamo_exporter if use_dynamo_exporter else None,
        precision=precision,
        conversion_device=conversion_device,
        torch_dtype=torch_dtype,
        target_opset=target_opset,
        use_ort_genai=use_ort_genai if use_ort_genai else None,
        output_path=output_path,
    )
    return await _run_olive("capture_onnx_graph", kwargs, ["onnxruntime"], ctx)


@mcp.tool()
async def benchmark(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    tasks: list[str] | None = None,
    device: str = "cpu",
    batch_size: int = 1,
    max_length: int = 1024,
    limit: float = 1.0,
    output_path: str | None = None,
) -> dict:
    """Benchmark/evaluate a model using lm-eval tasks.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        tasks: List of lm-eval tasks. Default: ["hellaswag"].
        device: Device for evaluation - "cpu" or "gpu".
        batch_size: Evaluation batch size. Default: 1.
        max_length: Maximum length of input + output. Default: 1024.
        limit: Fraction of samples (0.0-1.0) or absolute number. Default: 1.0.
        output_path: Directory to save results. Auto-generated if omitted.
    """
    if not tasks:
        tasks = ["hellaswag"]
    if not output_path:
        output_path = _make_output_path("benchmark", model_name_or_path)

    ort_package = "onnxruntime-gpu" if device == "gpu" else "onnxruntime"
    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        tasks=tasks,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        limit=limit,
        output_path=output_path,
    )
    return await _run_olive("benchmark", kwargs, [ort_package], ctx)


@mcp.tool()
async def diffusion_lora(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    data_dir: str | None = None,
    data_name: str | None = None,
    model_variant: str = "auto",
    lora_r: int = 16,
    alpha: float | None = None,
    max_train_steps: int = 1000,
    learning_rate: float = 1e-4,
    train_batch_size: int = 1,
    mixed_precision: str = "bf16",
    dreambooth: bool = False,
    instance_prompt: str | None = None,
    merge_lora: bool = False,
    output_path: str | None = None,
) -> dict:
    """Train LoRA adapters for diffusion models (SD 1.5, SDXL, Flux).

    Args:
        model_name_or_path: HuggingFace model name (e.g. "runwayml/stable-diffusion-v1-5").
        data_dir: Path to local image folder with training images.
        data_name: HuggingFace dataset name (alternative to data_dir).
        model_variant: Model type - "auto", "sd", "sdxl", "flux".
        lora_r: LoRA rank. SD: 4-16, Flux: 16-64. Default: 16.
        alpha: LoRA alpha for scaling. Default: same as r.
        max_train_steps: Maximum training steps. Default: 1000.
        learning_rate: Learning rate. Default: 1e-4.
        train_batch_size: Training batch size. Default: 1.
        mixed_precision: Mixed precision - "bf16", "fp16", "no". Default: bf16.
        dreambooth: Enable DreamBooth training.
        instance_prompt: Fixed prompt for DreamBooth mode.
        merge_lora: Merge LoRA into base model instead of saving adapter only.
        output_path: Directory to save LoRA adapter. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("diffusion_lora", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        data_name=data_name,
        model_variant=model_variant,
        lora_r=lora_r,
        alpha=alpha,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        mixed_precision=mixed_precision,
        dreambooth=dreambooth if dreambooth else None,
        instance_prompt=instance_prompt,
        merge_lora=merge_lora if merge_lora else None,
        output_path=output_path,
    )
    return await _run_olive("diffusion_lora", kwargs, [], ctx)


@mcp.tool()
async def generate_adapter(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    adapter_type: str = "LoRA",
    adapter_format: str = "onnx_adapter",
    output_path: str | None = None,
) -> dict:
    """Generate ONNX model with adapters as inputs. Only accepts ONNX models.

    Args:
        model_name_or_path: Path to ONNX model.
        adapter_type: Type of adapter - "LoRA". Default: LoRA.
        adapter_format: Format to save weights - "onnx_adapter", "safetensors", "npz".
        output_path: Directory to save output. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("generate_adapter", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        adapter_type=adapter_type,
        adapter_format=adapter_format,
        output_path=output_path,
    )
    return await _run_olive("generate_adapter", kwargs, ["onnxruntime"], ctx)


@mcp.tool()
async def convert_adapters(
    adapter_path: str,
    output_path: str,
    ctx: Context[ServerSession, None],
    adapter_format: str = "onnx_adapter",
    dtype: str = "float32",
    quantize_int4: bool = False,
    int4_block_size: int = 32,
    int4_quantization_mode: str = "symmetric",
) -> dict:
    """Convert LoRA adapter weights to a format consumable by ONNX models.

    Args:
        adapter_path: Path to adapter weights (local folder or HuggingFace ID).
        output_path: Path to save exported weights.
        adapter_format: Format - "onnx_adapter", "safetensors", "npz".
        dtype: Data type - "float32" or "float16".
        quantize_int4: Quantize adapter weights to int4.
        int4_block_size: Block size for int4 quantization (16/32/64/128/256).
        int4_quantization_mode: Mode - "symmetric" or "asymmetric".
    """
    kwargs = _build_kwargs(
        adapter_path=adapter_path,
        output_path=output_path,
        adapter_format=adapter_format,
        dtype=dtype,
        quantize_int4=quantize_int4 if quantize_int4 else None,
        int4_block_size=int4_block_size,
        int4_quantization_mode=int4_quantization_mode,
    )
    return await _run_olive("convert_adapters", kwargs, ["onnxruntime"], ctx)


@mcp.tool()
async def extract_adapters(
    model_name_or_path: str,
    format: str,
    output: str,
    ctx: Context[ServerSession, None],
    dtype: str = "float32",
) -> dict:
    """Extract LoRA adapters from a PyTorch model to separate files.

    Args:
        model_name_or_path: Path to PyTorch model (local folder or HuggingFace ID).
        format: Format to save LoRAs - "onnx_adapter", "safetensors", "npz".
        output: Output folder to save the LoRAs.
        dtype: Data type - "float32" or "float16".
    """
    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        format=format,
        output=output,
        dtype=dtype,
    )
    return await _run_olive("extract_adapters", kwargs, [], ctx)


@mcp.tool()
async def tune_session_params(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    cpu_cores: int | None = None,
    io_bind: bool = False,
    enable_cuda_graph: bool = False,
    output_path: str | None = None,
) -> dict:
    """Tune ONNX Runtime session parameters for optimal inference performance.

    Args:
        model_name_or_path: Path to ONNX model.
        cpu_cores: CPU cores for thread tuning.
        io_bind: Enable IOBinding Search.
        enable_cuda_graph: Enable CUDA Graph for CUDA EP.
        output_path: Directory to save tuned parameters. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("tune_params", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        cpu_cores=cpu_cores,
        io_bind=io_bind if io_bind else None,
        enable_cuda_graph=enable_cuda_graph if enable_cuda_graph else None,
        output_path=output_path,
    )
    return await _run_olive("tune_session_params", kwargs, ["onnxruntime"], ctx)


@mcp.tool()
async def generate_cost_model(
    model_name_or_path: str,
    ctx: Context[ServerSession, None],
    weight_precision: str = "fp16",
    output_path: str | None = None,
) -> dict:
    """Generate a cost model for model splitting (HuggingFace models only).

    The cost model is saved as a CSV file consumed by the CaptureSplitInfo pass.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        weight_precision: Weight precision for cost estimation - "fp32", "fp16", "int4", etc.
        output_path: Path to save cost model CSV. Auto-generated if omitted.
    """
    if not output_path:
        output_path = _make_output_path("cost_model", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        weight_precision=weight_precision,
        output_path=output_path,
    )
    return await _run_olive("generate_cost_model", kwargs, [], ctx)


@mcp.tool()
async def run_workflow(
    run_config: str,
    ctx: Context[ServerSession, None],
    output_path: str | None = None,
) -> dict:
    """Run a custom Olive workflow from a JSON config file.

    Args:
        run_config: Path to Olive workflow config JSON file.
        output_path: Directory to save results. Uses config default if omitted.
    """
    kwargs = _build_kwargs(
        run_config=run_config,
        output_path=output_path,
    )
    return await _run_olive("run", kwargs, ["onnxruntime"], ctx)


@mcp.tool()
async def list_outputs(
    prefix: str | None = None,
    limit: int = 20,
) -> dict:
    """List previous optimization outputs saved by olive-mcp.

    Use this to review past results, find model paths, or compare runs.

    Args:
        prefix: Filter by operation type (e.g. "optimize", "quantize", "finetune"). Show all if omitted.
        limit: Maximum number of results to return. Default: 20.
    """
    if not OUTPUT_BASE.exists():
        return {"outputs": [], "message": "No outputs found. Run an optimization first."}

    entries = []
    for d in sorted(OUTPUT_BASE.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        if prefix and not d.name.startswith(prefix):
            continue

        entry = {"name": d.name, "path": str(d)}

        # Extract timestamp from dir name
        parts = d.name.rsplit("_", 2)
        if len(parts) >= 3:
            entry["operation"] = parts[0]
            entry["timestamp"] = f"{parts[-2]}_{parts[-1]}"

        # Find model files
        for ext in ("*.onnx", "*.pt", "*.safetensors", "*.bin"):
            files = list(d.rglob(ext))
            if files:
                entry["model_files"] = [str(f) for f in files[:5]]
                break

        # Check for config
        for cfg_name in ("model_config.json", "config.json", "inference_config.json"):
            cfg = d / cfg_name
            if cfg.exists():
                entry["has_config"] = True
                break

        entries.append(entry)
        if len(entries) >= limit:
            break

    return {"outputs": entries, "total": len(entries)}


# ---------------------------------------------------------------------------
# MCP Prompts — guided workflows for beginners
# ---------------------------------------------------------------------------


@mcp.prompt(
    name="optimize-model",
    description="Guided model optimization — helps you choose the right model, precision, and target device.",
)
def prompt_optimize_model() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "I want to optimize a model using Olive. Help me step by step:\n"
                "1. Ask what I want to use the model for (chat, code, images, etc.)\n"
                "2. Ask about my target device (CPU, GPU, NPU) and any size/speed/quality preference\n"
                "3. Recommend a model and optimization settings based on my answers\n"
                "4. Run the optimization with the `optimize` tool\n"
                "5. Show me the results and suggest next steps (benchmark, deploy, etc.)"
            ),
        }
    ]


@mcp.prompt(
    name="quantize-model",
    description="Guided model quantization — helps you choose precision and algorithm.",
)
def prompt_quantize_model() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "I want to quantize a model to make it smaller/faster. Help me:\n"
                "1. Ask which model I want to quantize (or recommend one for my use case)\n"
                "2. Ask about my priority: smallest size, best quality, or balanced\n"
                "3. Choose the right precision and algorithm based on my answers\n"
                "4. Run the quantization with the `quantize` tool\n"
                "5. Show me the results (size reduction, etc.)"
            ),
        }
    ]


@mcp.prompt(
    name="finetune-model",
    description="Guided model fine-tuning — helps you set up LoRA/QLoRA training.",
)
def prompt_finetune_model() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "I want to fine-tune a model on my own data. Help me:\n"
                "1. Ask what task I want the model to learn\n"
                "2. Ask about my training data (HuggingFace dataset name or local path)\n"
                "3. Ask about my GPU memory to decide between LoRA and QLoRA\n"
                "4. Recommend a base model and training settings\n"
                "5. Run fine-tuning with the `finetune` tool\n"
                "6. Suggest next steps (optimize the fine-tuned model, benchmark, etc.)"
            ),
        }
    ]


@mcp.prompt(
    name="compare-models",
    description="Compare previous optimization results — find the best model from past runs.",
)
def prompt_compare_models() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "I want to compare my previous optimization results. Help me:\n"
                "1. Use `list_outputs` to show my past optimization runs\n"
                "2. Summarize each run (model, precision, size, metrics if available)\n"
                "3. Recommend which result is best for my needs\n"
                "4. If I haven't benchmarked yet, suggest running `benchmark` on the candidates"
            ),
        }
    ]
