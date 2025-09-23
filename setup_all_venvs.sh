#!/bin/bash
# 统一设置所有Python模块的虚拟环境

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== VoiceHelper 多模块虚拟环境设置 ==="
echo "项目根目录: $PROJECT_ROOT"

# 定义模块列表
declare -a MODULES=(
    "algo"
    "backend/app"
    "platforms/admin"
    "shared/sdks/python"
    "tools/scripts"
)

# 设置每个模块的虚拟环境
for module in "${MODULES[@]}"; do
    echo ""
    echo "=== 设置模块: $module ==="
    
    module_path="$PROJECT_ROOT/$module"
    setup_script="$module_path/setup_venv.sh"
    
    if [ -f "$setup_script" ]; then
        echo "执行设置脚本: $setup_script"
        cd "$module_path"
        bash setup_venv.sh
        cd "$PROJECT_ROOT"
        echo "✅ $module 虚拟环境设置完成"
    else
        echo "❌ 未找到设置脚本: $setup_script"
    fi
done

echo ""
echo "=== 所有模块虚拟环境设置完成 ==="
echo ""
echo "各模块激活命令:"
echo "  algo:                source algo/algo_venv/bin/activate"
echo "  backend/app:         source backend/app/backend_venv/bin/activate"
echo "  platforms/admin:     source platforms/admin/admin_venv/bin/activate"
echo "  shared/sdks/python:  source shared/sdks/python/sdk_venv/bin/activate"
echo "  tools/scripts:       source tools/scripts/tools_venv/bin/activate"
echo ""
echo "停用虚拟环境: deactivate"
