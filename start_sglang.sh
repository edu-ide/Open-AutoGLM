#!/bin/bash
# AutoGLM SGLang Server Startup Script

MODEL_PATH="/mnt/sda1/models/AutoGLM-Phone-9B-Multilingual"
PORT=8000
MODEL_NAME="autoglm-phone-9b"
ENV_PATH="/mnt/sda1/autoglm-env"

# Activate environment
source ${ENV_PATH}/bin/activate
export LD_LIBRARY_PATH=${ENV_PATH}/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

echo "========================================"
echo "AutoGLM SGLang Server"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "========================================"

# Parse arguments
QUANTIZATION=""
CPU_OFFLOAD=""
MAX_TOKENS="8192"

while [[ $# -gt 0 ]]; do
    case $1 in
        --fp8)
            QUANTIZATION="--quantization fp8"
            echo "Using FP8 quantization"
            shift
            ;;
        --int8)
            QUANTIZATION="--quantization compressed-tensors"
            echo "Using INT8 quantization"
            shift
            ;;
        --offload)
            CPU_OFFLOAD="--cpu-offload-gb ${2:-8}"
            echo "Using CPU offloading: ${2:-8}GB"
            shift 2
            ;;
        --tokens)
            MAX_TOKENS="$2"
            echo "Max tokens: $MAX_TOKENS"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --fp8           Use FP8 quantization (reduces memory ~50%)"
            echo "  --int8          Use INT8 quantization"
            echo "  --offload N     Offload N GB to CPU (default: 8)"
            echo "  --tokens N      Max total tokens (default: 8192)"
            echo "  --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --fp8                    # FP8 quantization (recommended for 24GB GPU)"
            echo "  $0 --offload 8              # CPU offloading 8GB"
            echo "  $0 --fp8 --tokens 16384     # FP8 with larger context"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default: use FP8 quantization (optimal for RTX 4090)
if [ -z "$QUANTIZATION" ] && [ -z "$CPU_OFFLOAD" ]; then
    QUANTIZATION="--quantization fp8"
    MAX_TOKENS="16384"
    echo "Default: Using FP8 quantization with 16384 tokens"
fi

echo ""
echo "Starting server..."
echo ""

python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --port $PORT \
    --host 0.0.0.0 \
    --served-model-name $MODEL_NAME \
    --trust-remote-code \
    --max-total-tokens $MAX_TOKENS \
    $QUANTIZATION \
    $CPU_OFFLOAD
