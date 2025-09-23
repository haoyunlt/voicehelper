#!/bin/bash

# VoiceHelper 增强版服务启动脚本
# 启动基于OpenAI Whisper、Edge-TTS、FAISS和Rasa的完整语音助手服务

set -e

echo "=========================================="
echo "VoiceHelper 增强版服务启动"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

# 检查虚拟环境
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  建议在虚拟环境中运行"
    echo "创建虚拟环境: python3 -m venv algo_venv"
    echo "激活虚拟环境: source algo_venv/bin/activate"
    echo ""
fi

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}

# 创建必要的目录
echo "创建数据目录..."
mkdir -p data/rag
mkdir -p data/tts_cache
mkdir -p logs

# 检查依赖
echo "检查依赖包..."
if ! python3 -c "import torch" &> /dev/null; then
    echo "❌ PyTorch 未安装，正在安装..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

if ! python3 -c "import whisper" &> /dev/null; then
    echo "❌ OpenAI Whisper 未安装，正在安装..."
    pip install openai-whisper
fi

if ! python3 -c "import edge_tts" &> /dev/null; then
    echo "❌ Edge-TTS 未安装，正在安装..."
    pip install edge-tts
fi

if ! python3 -c "import faiss" &> /dev/null; then
    echo "❌ FAISS 未安装，正在安装..."
    pip install faiss-cpu
fi

if ! python3 -c "import sentence_transformers" &> /dev/null; then
    echo "❌ Sentence Transformers 未安装，正在安装..."
    pip install sentence-transformers
fi

# 检查GPU支持
if python3 -c "import torch; print('GPU可用:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "✅ 检测到GPU支持"
    export TORCH_DEVICE="cuda"
else
    echo "ℹ️  使用CPU模式"
    export TORCH_DEVICE="cpu"
fi

# 检查Rasa服务 (可选)
echo "检查Rasa服务..."
if curl -s http://localhost:5005/status &> /dev/null; then
    echo "✅ Rasa服务运行中"
    export RASA_AVAILABLE="true"
else
    echo "⚠️  Rasa服务未运行，对话管理功能将受限"
    echo "启动Rasa: rasa run --enable-api --cors \"*\" --port 5005"
    export RASA_AVAILABLE="false"
fi

# 运行测试 (可选)
if [[ "$1" == "test" ]]; then
    echo ""
    echo "运行增强服务测试..."
    python3 test_enhanced_services.py
    exit 0
fi

# 启动服务
echo ""
echo "启动VoiceHelper增强版API服务..."
echo "服务地址: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo "健康检查: http://localhost:8000/health"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动FastAPI服务
if [[ -f "app/v2_api_enhanced.py" ]]; then
    cd app
    python3 -m uvicorn v2_api_enhanced:app --host 0.0.0.0 --port 8000 --reload
else
    echo "❌ 找不到 app/v2_api_enhanced.py"
    echo "请确保在正确的目录中运行此脚本"
    exit 1
fi
