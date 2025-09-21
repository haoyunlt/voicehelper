#!/bin/bash

# 便捷部署助手脚本 - 调用deploy目录下的实际脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 调用实际的部署助手脚本
exec "$SCRIPT_DIR/deploy/scripts/deploy-helper.sh" "$@"
