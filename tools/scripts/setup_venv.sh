#!/bin/bash
# tools/scripts模块虚拟环境设置脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/tools_venv"

echo "=== 设置tools/scripts模块虚拟环境 ==="

# 删除旧的虚拟环境
if [ -d "$VENV_DIR" ]; then
    echo "删除旧的虚拟环境..."
    rm -rf "$VENV_DIR"
fi

# 创建新的虚拟环境
echo "创建虚拟环境: $VENV_DIR"
python3 -m venv "$VENV_DIR"

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装依赖..."
pip install -r "$SCRIPT_DIR/requirements.txt"

echo "=== tools/scripts模块虚拟环境设置完成 ==="
echo "激活命令: source $VENV_DIR/bin/activate"
echo "停用命令: deactivate"
