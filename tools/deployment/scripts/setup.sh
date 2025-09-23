#!/bin/bash

# 聊天机器人项目初始化脚本

set -e

echo "🚀 初始化聊天机器人项目..."

# 检查 Docker 和 Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker 未安装，请先安装 Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose 未安装，请先安装 Docker Compose"
    exit 1
fi

# 创建环境变量文件
if [ ! -f .env ]; then
    echo "📝 创建环境变量文件..."
    cp env.example .env
    echo "✅ 请编辑 .env 文件，填入您的豆包 API Key"
fi

# 创建必要的目录
echo "📁 创建必要的目录..."
mkdir -p logs
mkdir -p data/{uploads,exports}

# 构建镜像
echo "🔨 构建 Docker 镜像..."
docker-compose build

# 启动基础设施服务
echo "🏗️ 启动基础设施服务..."

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 检查服务状态
echo "🔍 检查服务状态..."
docker-compose ps

echo ""
echo "✅ 项目初始化完成！"
echo ""
echo "📋 下一步操作："
echo "1. 编辑 .env 文件，填入您的豆包 API Key"
echo "2. 运行 'make up' 启动所有服务"
echo "3. 访问 http://localhost:3000 使用聊天机器人"
echo ""
echo "🔧 常用命令："
echo "  make up      - 启动所有服务"
echo "  make down    - 停止所有服务"
echo "  make logs    - 查看日志"
echo "  make clean   - 清理所有数据"
echo ""
