#!/bin/bash
# AutoGLM vLLM Server Startup Script

MODEL_PATH="/mnt/sda1/models/AutoGLM-Phone-9B-Multilingual"
PORT=8000
MODEL_NAME="autoglm-phone-9b"

echo "Starting AutoGLM vLLM Server..."
echo "Model: $MODEL_PATH"
echo "Port: $PORT"

python3 -m vllm.entrypoints.openai.api_server \
    --served-model-name $MODEL_NAME \
    --allowed-local-media-path / \
    --model $MODEL_PATH \
    --port $PORT \
    --trust-remote-code \
    --dtype bfloat16
