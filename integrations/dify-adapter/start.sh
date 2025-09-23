#!/bin/bash

# Dify-VoiceHelper 集成适配器启动脚本

set -e

echo "🚀 启动 Dify-VoiceHelper 集成适配器..."

# 等待依赖服务启动
echo "⏳ 等待依赖服务启动..."

# 等待PostgreSQL
echo "等待 PostgreSQL..."
while ! nc -z ${POSTGRES_HOST:-postgres} ${POSTGRES_PORT:-5432}; do
  sleep 1
done
echo "✅ PostgreSQL 已就绪"

# 等待Redis
echo "等待 Redis..."
while ! nc -z ${REDIS_HOST:-redis} ${REDIS_PORT:-6379}; do
  sleep 1
done
echo "✅ Redis 已就绪"

# 等待VoiceHelper Gateway
echo "等待 VoiceHelper Gateway..."
while ! nc -z gateway 8080; do
  sleep 1
done
echo "✅ VoiceHelper Gateway 已就绪"

# 等待VoiceHelper Algo Service
echo "等待 VoiceHelper Algo Service..."
while ! nc -z algo-service 8000; do
  sleep 1
done
echo "✅ VoiceHelper Algo Service 已就绪"

# 等待Dify API
echo "等待 Dify API..."
while ! nc -z dify-api 5001; do
  sleep 1
done
echo "✅ Dify API 已就绪"

# 运行数据库迁移
echo "🔄 运行数据库迁移..."
python -c "
import asyncio
from core.database import DatabaseManager
from core.config import Settings

async def migrate():
    settings = Settings()
    db_manager = DatabaseManager(settings.database_url)
    await db_manager.initialize()
    await db_manager.migrate()
    await db_manager.close()
    print('✅ 数据库迁移完成')

asyncio.run(migrate())
"

# 启动应用
echo "🎯 启动集成适配器服务..."
exec python main.py
