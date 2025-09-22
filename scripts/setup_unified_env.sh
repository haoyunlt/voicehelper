#!/bin/bash

# VoiceHelper 统一环境配置设置脚本
# 自动设置统一的 .env 配置文件

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/.env"
TEMPLATE_FILE="$PROJECT_ROOT/env.unified.new"

echo "🚀 VoiceHelper 统一环境配置设置"
echo "=================================="
echo "项目路径: $PROJECT_ROOT"
echo ""

# 检查是否存在现有的 .env 文件
if [ -f "$ENV_FILE" ]; then
    echo "⚠️  发现现有的 .env 文件"
    echo "当前配置文件: $ENV_FILE"
    echo ""
    read -p "是否要备份现有配置并创建新的配置文件? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # 备份现有文件
        BACKUP_FILE="$ENV_FILE.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$ENV_FILE" "$BACKUP_FILE"
        echo "✅ 已备份现有配置到: $BACKUP_FILE"
    else
        echo "❌ 取消操作，保持现有配置"
        exit 0
    fi
fi

# 检查模板文件是否存在
if [ ! -f "$TEMPLATE_FILE" ]; then
    echo "❌ 错误: 未找到配置模板文件 $TEMPLATE_FILE"
    exit 1
fi

# 复制模板文件
echo "📋 复制配置模板..."
cp "$TEMPLATE_FILE" "$ENV_FILE"
echo "✅ 已创建配置文件: $ENV_FILE"
echo ""

# 提示用户配置 API 密钥
echo "🔑 API 密钥配置"
echo "================"
echo ""
echo "请配置以下 API 密钥以启用 AI 功能:"
echo ""

# GLM-4 配置
echo "1. GLM-4 (智谱AI) - 推荐，成本最低 (0.1元/百万tokens)"
echo "   获取地址: https://open.bigmodel.cn/"
echo "   当前配置: $(grep '^GLM_API_KEY=' "$ENV_FILE" | cut -d'=' -f2)"
echo ""
read -p "   输入新的 GLM-4 API 密钥 (回车跳过): " glm_key
if [ ! -z "$glm_key" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^GLM_API_KEY=.*/GLM_API_KEY=$glm_key/" "$ENV_FILE"
    else
        # Linux
        sed -i "s/^GLM_API_KEY=.*/GLM_API_KEY=$glm_key/" "$ENV_FILE"
    fi
    echo "   ✅ 已更新 GLM-4 API 密钥"
fi
echo ""

# 豆包配置
echo "2. 豆包 (字节跳动) - 备选方案 (2.8元/百万tokens)"
echo "   获取地址: https://console.volcengine.com/"
echo "   当前配置: $(grep '^ARK_API_KEY=' "$ENV_FILE" | cut -d'=' -f2)"
echo ""
read -p "   输入新的豆包 API 密钥 (回车跳过): " ark_key
if [ ! -z "$ark_key" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^ARK_API_KEY=.*/ARK_API_KEY=$ark_key/" "$ENV_FILE"
    else
        # Linux
        sed -i "s/^ARK_API_KEY=.*/ARK_API_KEY=$ark_key/" "$ENV_FILE"
    fi
    echo "   ✅ 已更新豆包 API 密钥"
fi
echo ""

# OpenAI 配置 (可选)
echo "3. OpenAI - 可选 (~200元/百万tokens)"
echo "   获取地址: https://platform.openai.com/"
echo ""
read -p "   输入 OpenAI API 密钥 (回车跳过): " openai_key
if [ ! -z "$openai_key" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" "$ENV_FILE"
    else
        # Linux
        sed -i "s/^OPENAI_API_KEY=.*/OPENAI_API_KEY=$openai_key/" "$ENV_FILE"
    fi
    echo "   ✅ 已更新 OpenAI API 密钥"
fi
echo ""

# 安全配置
echo "🔐 安全配置"
echo "==========="
echo ""
echo "为了安全，建议修改默认的密钥和密码:"
echo ""

# JWT Secret
echo "1. JWT 密钥 (用于用户认证)"
read -p "   是否生成新的 JWT 密钥? (Y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    # 生成随机 JWT 密钥
    jwt_secret=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || echo "$(date +%s | sha256sum | head -c 64)")
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^JWT_SECRET=.*/JWT_SECRET=$jwt_secret/" "$ENV_FILE"
    else
        # Linux
        sed -i "s/^JWT_SECRET=.*/JWT_SECRET=$jwt_secret/" "$ENV_FILE"
    fi
    echo "   ✅ 已生成新的 JWT 密钥"
fi
echo ""

# 数据库密码
echo "2. 数据库密码"
read -p "   是否修改默认数据库密码? (y/N): " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -p "   输入新的 PostgreSQL 密码: " postgres_pass
    read -p "   输入新的 Redis 密码: " redis_pass
    
    if [ ! -z "$postgres_pass" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$postgres_pass/" "$ENV_FILE"
            sed -i '' "s|postgresql://voicehelper:voicehelper123@|postgresql://voicehelper:$postgres_pass@|g" "$ENV_FILE"
        else
            # Linux
            sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$postgres_pass/" "$ENV_FILE"
            sed -i "s|postgresql://voicehelper:voicehelper123@|postgresql://voicehelper:$postgres_pass@|g" "$ENV_FILE"
        fi
        echo "   ✅ 已更新 PostgreSQL 密码"
    fi
    
    if [ ! -z "$redis_pass" ]; then
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/^REDIS_PASSWORD=.*/REDIS_PASSWORD=$redis_pass/" "$ENV_FILE"
            sed -i '' "s|redis://:redis123@|redis://:$redis_pass@|g" "$ENV_FILE"
        else
            # Linux
            sed -i "s/^REDIS_PASSWORD=.*/REDIS_PASSWORD=$redis_pass/" "$ENV_FILE"
            sed -i "s|redis://:redis123@|redis://:$redis_pass@|g" "$ENV_FILE"
        fi
        echo "   ✅ 已更新 Redis 密码"
    fi
fi
echo ""

# 运行配置验证
echo "🔍 配置验证"
echo "==========="
echo ""
echo "正在验证配置..."

if command -v python3 &> /dev/null; then
    python3 "$PROJECT_ROOT/scripts/validate_env_config.py"
else
    echo "⚠️  未找到 python3，跳过自动验证"
    echo "请手动运行: python3 scripts/validate_env_config.py"
fi
echo ""

# 完成提示
echo "🎉 配置设置完成!"
echo "================"
echo ""
echo "📁 配置文件位置: $ENV_FILE"
echo ""
echo "🚀 下一步操作:"
echo "1. 检查配置: python3 scripts/validate_env_config.py"
echo "2. 启动服务: docker-compose -f docker-compose.local.yml up -d"
echo "3. 访问应用:"
echo "   - 前端应用: http://localhost:3000"
echo "   - API 文档: http://localhost:8000/docs"
echo "   - 管理后台: http://localhost:5001"
echo ""
echo "📚 更多信息请查看: docs/UNIFIED_ENV_CONFIG_GUIDE.md"
echo ""
echo "✨ 祝你使用愉快!"
