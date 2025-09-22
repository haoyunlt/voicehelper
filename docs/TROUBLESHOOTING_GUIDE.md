# VoiceHelper 故障排除指南

## 📋 目录

- [快速诊断](#快速诊断)
- [服务启动问题](#服务启动问题)
- [数据库连接问题](#数据库连接问题)
- [API 接口问题](#api-接口问题)
- [语音功能问题](#语音功能问题)
- [前端界面问题](#前端界面问题)
- [性能问题](#性能问题)
- [网络连接问题](#网络连接问题)
- [安全和权限问题](#安全和权限问题)
- [数据备份恢复](#数据备份恢复)
- [监控和日志](#监控和日志)
- [常见错误代码](#常见错误代码)

## 🔍 快速诊断

### 系统健康检查脚本

```bash
#!/bin/bash
# scripts/health-check.sh

echo "🏥 VoiceHelper 系统健康检查"
echo "================================"

# 检查服务状态
echo "📊 检查服务状态..."

for service in "${services[@]}"; do
    if docker-compose ps | grep -q "$service.*Up"; then
        echo "✅ $service: 运行正常"
    else
        echo "❌ $service: 服务异常"
    fi
done

# 检查端口连通性
echo -e "\n🔌 检查端口连通性..."

for port_info in "${ports[@]}"; do
    port=$(echo $port_info | cut -d: -f1)
    name=$(echo $port_info | cut -d: -f2)
    
    if nc -z localhost $port 2>/dev/null; then
        echo "✅ $name ($port): 连接正常"
    else
        echo "❌ $name ($port): 连接失败"
    fi
done

# 检查API健康状态
echo -e "\n🩺 检查API健康状态..."
if curl -f -s http://localhost:8080/health > /dev/null; then
    echo "✅ 后端API: 健康"
else
    echo "❌ 后端API: 异常"
fi

if curl -f -s http://localhost:8000/health > /dev/null; then
    echo "✅ 算法服务: 健康"
else
    echo "❌ 算法服务: 异常"
fi

# 检查磁盘空间
echo -e "\n💾 检查磁盘空间..."
disk_usage=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $disk_usage -lt 80 ]; then
    echo "✅ 磁盘空间: ${disk_usage}% (正常)"
elif [ $disk_usage -lt 90 ]; then
    echo "⚠️ 磁盘空间: ${disk_usage}% (警告)"
else
    echo "❌ 磁盘空间: ${disk_usage}% (严重不足)"
fi

# 检查内存使用
echo -e "\n🧠 检查内存使用..."
memory_usage=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
if (( $(echo "$memory_usage < 80" | bc -l) )); then
    echo "✅ 内存使用: ${memory_usage}% (正常)"
elif (( $(echo "$memory_usage < 90" | bc -l) )); then
    echo "⚠️ 内存使用: ${memory_usage}% (警告)"
else
    echo "❌ 内存使用: ${memory_usage}% (过高)"
fi

echo -e "\n✅ 健康检查完成！"
```

### 一键诊断命令

```bash
# 运行健康检查
./scripts/health-check.sh

# 查看所有服务状态
docker-compose ps

# 查看系统资源使用
docker stats --no-stream

# 检查最近的错误日志
docker-compose logs --tail=50 | grep -i error
```

## 🚀 服务启动问题

### 问题1: Docker 容器无法启动

**症状**:
```bash
$ docker-compose up -d
ERROR: for backend  Cannot start service backend: driver failed programming external connectivity
```

**可能原因**:
- 端口被占用
- Docker 服务未启动
- 权限不足
- 配置文件错误

**解决方案**:

1. **检查端口占用**:
```bash
# 检查端口使用情况
sudo lsof -i :8080
sudo lsof -i :8000
sudo lsof -i :3000

# 杀死占用端口的进程
sudo kill -9 PID

# 或者修改端口配置
vim docker-compose.yml
# 将 "8080:8080" 改为 "8081:8080"
```

2. **重启 Docker 服务**:
```bash
# Ubuntu/Debian
sudo systemctl restart docker

# macOS
# 重启 Docker Desktop

# 检查 Docker 状态
docker info
```

3. **检查权限**:
```bash
# 添加用户到 docker 组
sudo usermod -aG docker $USER

# 重新登录或执行
newgrp docker

# 检查权限
docker ps
```

4. **清理并重新启动**:
```bash
# 停止所有容器
docker-compose down

# 清理系统
docker system prune -f

# 重新启动
docker-compose up -d
```

### 问题2: 服务启动后立即退出

**症状**:
```bash
$ docker-compose ps
Name    Command    State    Ports
backend   /app/server   Exit 1
```

**解决方案**:

1. **查看详细日志**:
```bash
# 查看服务日志
docker-compose logs backend

# 实时查看日志
docker-compose logs -f backend
```

2. **检查配置文件**:
```bash
# 验证环境变量
docker-compose config

# 检查配置文件语法
docker-compose -f docker-compose.yml config
```

3. **手动运行容器调试**:
```bash
# 交互式运行容器
docker run -it --rm voicehelper/backend:latest /bin/sh

# 检查应用启动
./server --help
```

### 问题3: 依赖服务启动顺序问题

**症状**:
```
backend_1  | Error: dial tcp 172.18.0.3:5432: connect: connection refused
postgres_1 | database system is ready to accept connections
```

**解决方案**:

1. **添加健康检查**:
```yaml
# docker-compose.yml
services:
  postgres:
    image: postgres:15-alpine
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  backend:
    depends_on:
      postgres:
        condition: service_healthy
```

2. **使用等待脚本**:
```bash
#!/bin/bash
# scripts/wait-for-it.sh
# 等待服务可用

host="$1"
port="$2"
timeout="${3:-30}"

echo "等待 $host:$port 可用..."

for i in $(seq 1 $timeout); do
    if nc -z "$host" "$port" 2>/dev/null; then
        echo "$host:$port 已可用"
        exit 0
    fi
    echo "等待中... ($i/$timeout)"
    sleep 1
done

echo "超时: $host:$port 不可用"
exit 1
```

3. **分步启动**:
```bash
# 先启动基础服务

# 等待服务就绪
sleep 30

# 启动应用服务
docker-compose up -d backend algo-service frontend
```

## 🗄️ 数据库连接问题

### 问题1: PostgreSQL 连接被拒绝

**症状**:
```
pq: password authentication failed for user "postgres"
```

**解决方案**:

1. **检查密码配置**:
```bash
# 检查环境变量
docker-compose exec backend env | grep DATABASE

# 重置密码
docker-compose exec postgres psql -U postgres -c "ALTER USER postgres PASSWORD 'new_password';"
```

2. **检查连接字符串**:
```bash
# 正确的连接字符串格式
DATABASE_URL=postgresql://postgres:password@postgres:5432/voicehelper

# 检查主机名（容器内应使用服务名）
# ❌ 错误: localhost
# ✅ 正确: postgres
```

3. **重置数据库**:
```bash
# 停止服务
docker-compose down

# 删除数据卷
docker volume rm deploy_postgres_data

# 重新启动
docker-compose up -d postgres

# 等待初始化完成
sleep 30
```

### 问题2: 数据库连接数过多

**症状**:
```
pq: sorry, too many clients already
```

**解决方案**:

1. **检查连接数**:
```sql
-- 查看当前连接数
SELECT count(*) FROM pg_stat_activity;

-- 查看最大连接数
SHOW max_connections;

-- 查看连接详情
SELECT pid, usename, application_name, client_addr, state 
FROM pg_stat_activity;
```

2. **优化连接池配置**:
```go
// backend 连接池配置
db.SetMaxOpenConns(25)      // 最大打开连接数
db.SetMaxIdleConns(10)      // 最大空闲连接数
db.SetConnMaxLifetime(5 * time.Minute)  // 连接最大生存时间
```

3. **增加数据库最大连接数**:
```bash
# 修改 PostgreSQL 配置
docker-compose exec postgres psql -U postgres -c "ALTER SYSTEM SET max_connections = 200;"
docker-compose restart postgres
```

### 问题3: Redis 连接问题

**症状**:
```
dial tcp 127.0.0.1:6379: connect: connection refused
```

**解决方案**:

1. **检查 Redis 状态**:
```bash
# 检查 Redis 服务
docker-compose exec redis redis-cli ping

# 查看 Redis 日志
docker-compose logs redis
```

2. **检查连接配置**:
```bash
# 检查 Redis URL
REDIS_URL=redis://redis:6379

# 如果有密码
REDIS_URL=redis://:password@redis:6379
```

3. **重启 Redis**:
```bash
# 重启 Redis 服务
docker-compose restart redis

# 清理 Redis 数据（谨慎操作）
docker-compose exec redis redis-cli FLUSHALL
```

## 🔌 API 接口问题

### 问题1: API 返回 404 错误

**症状**:
```json
{
  "error": "404 Not Found",
  "message": "The requested resource was not found"
}
```

**解决方案**:

1. **检查 API 路径**:
```bash
# 正确的 API 路径
curl http://localhost:8080/api/v1/health

# 检查可用路由
curl http://localhost:8080/api/v1/routes
```

2. **检查服务状态**:
```bash
# 检查后端服务
docker-compose logs backend

# 检查路由注册
grep -r "router\." backend/
```

### 问题2: API 响应超时

**症状**:
```
curl: (28) Operation timed out after 30000 milliseconds
```

**解决方案**:

1. **检查服务负载**:
```bash
# 查看容器资源使用
docker stats

# 查看系统负载
top
htop
```

2. **增加超时时间**:
```bash
# 增加 curl 超时
curl --connect-timeout 60 --max-time 120 http://localhost:8080/api/v1/chat

# 检查服务器超时配置
grep -r "timeout" backend/
```

3. **优化性能**:
```yaml
# docker-compose.yml 增加资源限制
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### 问题3: API 认证失败

**症状**:
```json
{
  "error": "Unauthorized",
  "message": "Invalid API key"
}
```

**解决方案**:

1. **检查 API Key**:
```bash
# 检查 API Key 格式
echo "YOUR_API_KEY" | wc -c  # 应该是合理长度

# 测试 API Key
curl -H "Authorization: Bearer YOUR_API_KEY" \
     http://localhost:8080/api/v1/health
```

2. **生成新的 API Key**:
```bash
# 如果有管理接口
curl -X POST http://localhost:8080/api/v1/admin/api-keys \
  -H "Content-Type: application/json" \
  -d '{"name": "test-key"}'
```

## 🎤 语音功能问题

### 问题1: 语音识别不工作

**症状**:
- 上传音频文件无响应
- 识别结果为空
- 返回错误信息

**解决方案**:

1. **检查音频格式**:
```bash
# 支持的格式: wav, mp3, webm
# 推荐格式: 16kHz, 16bit, mono WAV

# 转换音频格式
ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 output.wav

# 检查音频信息
ffprobe input.wav
```

2. **测试语音识别服务**:
```bash
# 检查算法服务状态
curl http://localhost:8000/health

# 测试语音识别接口
curl -X POST http://localhost:8000/api/v1/voice/asr \
  -F "audio=@test.wav" \
  -F "language=zh-CN"
```

3. **检查服务配置**:
```bash
# 查看算法服务日志
docker-compose logs algo-service

# 检查环境变量
docker-compose exec algo-service env | grep -E "(OPENAI|ASR)"
```

### 问题2: 语音合成失败

**症状**:
```json
{
  "error": "TTS service unavailable",
  "message": "Text-to-speech conversion failed"
}
```

**解决方案**:

1. **检查文本内容**:
```bash
# 测试简单文本
curl -X POST http://localhost:8000/api/v1/voice/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "你好", "voice": "zh-CN-XiaoxiaoNeural"}'

# 检查文本长度限制
echo "your_text" | wc -c
```

2. **检查语音模型**:
```bash
# 查看可用语音
curl http://localhost:8000/api/v1/voice/voices

# 测试不同语音
curl -X POST http://localhost:8000/api/v1/voice/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "voice": "en-US-JennyNeural"}'
```

### 问题3: 语音质量问题

**症状**:
- 识别准确率低
- 合成语音不自然
- 音频有噪音

**解决方案**:

1. **优化音频质量**:
```bash
# 降噪处理
ffmpeg -i noisy.wav -af "highpass=f=200,lowpass=f=3000" clean.wav

# 音量标准化
ffmpeg -i input.wav -af "loudnorm" normalized.wav
```

2. **调整识别参数**:
```json
{
  "audio": "base64_audio_data",
  "language": "zh-CN",
  "model": "whisper-large",
  "temperature": 0.0,
  "no_speech_threshold": 0.6
}
```

3. **优化合成参数**:
```json
{
  "text": "你好，欢迎使用VoiceHelper",
  "voice": "zh-CN-XiaoxiaoNeural",
  "speed": 1.0,
  "pitch": 0,
  "volume": 1.0,
  "style": "cheerful"
}
```

## 🖥️ 前端界面问题

### 问题1: 页面无法加载

**症状**:
- 浏览器显示 "无法访问此网站"
- 页面一直加载中
- 显示 502/503 错误

**解决方案**:

1. **检查前端服务**:
```bash
# 检查前端容器状态
docker-compose ps frontend

# 查看前端日志
docker-compose logs frontend

# 检查端口映射
docker port $(docker-compose ps -q frontend)
```

2. **检查网络连接**:
```bash
# 测试端口连通性
curl -I http://localhost:3000

# 检查防火墙设置
sudo ufw status
```

3. **重新构建前端**:
```bash
# 重新构建前端镜像
docker-compose build --no-cache frontend

# 重启前端服务
docker-compose up -d frontend
```

### 问题2: JavaScript 错误

**症状**:
- 浏览器控制台显示错误
- 功能按钮无响应
- 页面显示不完整

**解决方案**:

1. **检查浏览器控制台**:
```javascript
// 打开浏览器开发者工具 (F12)
// 查看 Console 标签页的错误信息

// 常见错误类型:
// - CORS 错误
// - API 请求失败
// - 资源加载失败
```

2. **检查 API 连接**:
```bash
# 检查 API 配置
grep -r "API_URL" frontend/

# 测试 API 连接
curl http://localhost:8080/api/v1/health
```

3. **清除浏览器缓存**:
```bash
# Chrome: Ctrl+Shift+R (强制刷新)
# Firefox: Ctrl+F5
# Safari: Cmd+Shift+R

# 或者在开发者工具中禁用缓存
```

### 问题3: 样式显示异常

**症状**:
- 页面布局混乱
- 样式丢失
- 响应式布局不工作

**解决方案**:

1. **检查 CSS 加载**:
```bash
# 查看网络标签页，检查 CSS 文件是否加载成功
# 检查 404 错误

# 查看前端构建日志
docker-compose logs frontend | grep -i css
```

2. **重新构建样式**:
```bash
# 进入前端容器
docker-compose exec frontend /bin/sh

# 重新构建
npm run build

# 或者重新安装依赖
npm install
```

## ⚡ 性能问题

### 问题1: 响应速度慢

**症状**:
- API 响应时间超过 5 秒
- 页面加载缓慢
- 用户体验差

**诊断方法**:

1. **性能监控**:
```bash
# 检查 API 响应时间
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8080/api/v1/health

# curl-format.txt 内容:
#      time_namelookup:  %{time_namelookup}\n
#         time_connect:  %{time_connect}\n
#      time_appconnect:  %{time_appconnect}\n
#     time_pretransfer:  %{time_pretransfer}\n
#        time_redirect:  %{time_redirect}\n
#   time_starttransfer:  %{time_starttransfer}\n
#                      ----------\n
#           time_total:  %{time_total}\n
```

2. **系统资源监控**:
```bash
# 查看系统负载
uptime

# 查看内存使用
free -h

# 查看磁盘 I/O
iostat -x 1

# 查看容器资源使用
docker stats
```

**优化方案**:

1. **数据库优化**:
```sql
-- 添加索引
CREATE INDEX CONCURRENTLY idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX CONCURRENTLY idx_messages_created_at ON messages(created_at);

-- 分析查询性能
EXPLAIN ANALYZE SELECT * FROM messages WHERE conversation_id = 'xxx';

-- 更新统计信息
ANALYZE;
```

2. **缓存优化**:
```bash
# 检查 Redis 缓存命中率
docker-compose exec redis redis-cli info stats | grep keyspace

# 优化缓存配置
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
```

3. **应用优化**:
```go
// 连接池优化
db.SetMaxOpenConns(25)
db.SetMaxIdleConns(25)
db.SetConnMaxLifetime(5 * time.Minute)

// 启用 gzip 压缩
router.Use(gzip.Gzip(gzip.DefaultCompression))
```

### 问题2: 内存泄漏

**症状**:
- 内存使用持续增长
- 系统变慢
- 容器被 OOM 杀死

**诊断方法**:

1. **监控内存使用**:
```bash
# 持续监控容器内存
watch -n 5 'docker stats --no-stream'

# 查看内存使用趋势
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

2. **分析内存泄漏**:
```bash
# Go 应用内存分析
curl http://localhost:8080/debug/pprof/heap > heap.prof
go tool pprof heap.prof

# Python 应用内存分析
pip install memory_profiler
python -m memory_profiler your_script.py
```

**解决方案**:

1. **设置内存限制**:
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

2. **优化代码**:
```go
// 及时关闭资源
defer rows.Close()
defer resp.Body.Close()

// 使用对象池
var bufferPool = sync.Pool{
    New: func() interface{} {
        return make([]byte, 1024)
    },
}
```

## 🌐 网络连接问题

### 问题1: 容器间网络不通

**症状**:
```
dial tcp: lookup postgres on 127.0.0.11:53: no such host
```

**解决方案**:

1. **检查网络配置**:
```bash
# 查看 Docker 网络
docker network ls

# 检查容器网络
docker-compose exec backend nslookup postgres

# 查看网络详情
docker network inspect deploy_default
```

2. **重建网络**:
```bash
# 停止服务
docker-compose down

# 删除网络
docker network prune

# 重新启动
docker-compose up -d
```

### 问题2: 外部网络访问问题

**症状**:
- 无法访问外部 API
- DNS 解析失败
- 网络超时

**解决方案**:

1. **检查 DNS 配置**:
```bash
# 测试 DNS 解析
docker-compose exec backend nslookup google.com

# 检查 DNS 配置
docker-compose exec backend cat /etc/resolv.conf
```

2. **检查防火墙**:
```bash
# 检查防火墙规则
sudo iptables -L

# 检查 Docker 防火墙规则
sudo iptables -L DOCKER
```

3. **配置代理**:
```yaml
# docker-compose.yml
services:
  backend:
    environment:
      - HTTP_PROXY=http://proxy:8080
      - HTTPS_PROXY=http://proxy:8080
      - NO_PROXY=localhost,127.0.0.1
```

## 🔒 安全和权限问题

### 问题1: 权限被拒绝

**症状**:
```
permission denied while trying to connect to the Docker daemon socket
```

**解决方案**:

1. **添加用户到 docker 组**:
```bash
# 添加当前用户到 docker 组
sudo usermod -aG docker $USER

# 重新登录或执行
newgrp docker

# 验证权限
docker ps
```

2. **检查文件权限**:
```bash
# 检查 Docker socket 权限
ls -la /var/run/docker.sock

# 修复权限
sudo chmod 666 /var/run/docker.sock
```

### 问题2: SSL/TLS 证书问题

**症状**:
```
x509: certificate signed by unknown authority
```

**解决方案**:

1. **更新证书**:
```bash
# 更新系统证书
sudo apt update && sudo apt install ca-certificates

# 重新生成 Let's Encrypt 证书
sudo certbot renew --force-renewal
```

2. **配置自签名证书**:
```bash
# 生成自签名证书
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# 配置 Nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
}
```

## 💾 数据备份恢复

### 问题1: 数据丢失

**症状**:
- 数据库数据丢失
- 用户对话历史消失
- 知识库文档丢失

**预防措施**:

1. **设置自动备份**:
```bash
#!/bin/bash
# scripts/auto-backup.sh

BACKUP_DIR="/backup/voicehelper"
DATE=$(date +%Y%m%d_%H%M%S)

# 备份数据库
docker-compose exec -T postgres pg_dump -U postgres voicehelper | gzip > $BACKUP_DIR/postgres_$DATE.sql.gz

# 备份 Redis
docker-compose exec redis redis-cli BGSAVE
docker cp $(docker-compose ps -q redis):/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb


# 上传到云存储
aws s3 sync $BACKUP_DIR s3://your-backup-bucket/voicehelper/
```

2. **设置定时任务**:
```bash
# 添加到 crontab
crontab -e

# 每天凌晨 2 点备份
0 2 * * * /path/to/scripts/auto-backup.sh >> /var/log/voicehelper-backup.log 2>&1
```

**恢复数据**:

1. **恢复数据库**:
```bash
#!/bin/bash
# scripts/restore-data.sh

BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "用法: $0 <backup_date>"
    exit 1
fi

# 停止服务
docker-compose down

# 恢复 PostgreSQL
docker-compose up -d postgres
sleep 30
zcat /backup/voicehelper/postgres_$BACKUP_FILE.sql.gz | docker-compose exec -T postgres psql -U postgres voicehelper

# 恢复 Redis
docker-compose up -d redis
sleep 10
docker cp /backup/voicehelper/redis_$BACKUP_FILE.rdb $(docker-compose ps -q redis):/data/dump.rdb
docker-compose restart redis

# 启动所有服务
docker-compose up -d
```

## 📊 监控和日志

### 日志分析

1. **集中日志查看**:
```bash
# 查看所有服务日志
docker-compose logs -f

# 查看特定服务日志
docker-compose logs -f backend

# 查看最近的错误
docker-compose logs --tail=100 | grep -i error

# 按时间过滤日志
docker-compose logs --since="2025-01-21T10:00:00"
```

2. **日志分析脚本**:
```bash
#!/bin/bash
# scripts/analyze-logs.sh

echo "📊 日志分析报告"
echo "================"

# 错误统计
echo "🔴 错误统计:"
docker-compose logs --since="24h" | grep -i error | wc -l

# 最频繁的错误
echo -e "\n🔍 最频繁的错误:"
docker-compose logs --since="24h" | grep -i error | sort | uniq -c | sort -nr | head -5

# API 请求统计
echo -e "\n📈 API 请求统计:"
docker-compose logs backend --since="24h" | grep "GET\|POST\|PUT\|DELETE" | wc -l

# 响应时间分析
echo -e "\n⏱️ 平均响应时间:"
docker-compose logs backend --since="24h" | grep "duration" | awk '{print $NF}' | awk '{sum+=$1; count++} END {print sum/count "ms"}'
```

### 性能监控

1. **系统监控脚本**:
```bash
#!/bin/bash
# scripts/monitor-system.sh

while true; do
    echo "$(date): 系统监控报告"
    echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)"
    echo "内存: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
    echo "磁盘: $(df / | tail -1 | awk '{print $5}')"
    echo "负载: $(uptime | awk -F'load average:' '{print $2}')"
    echo "---"
    sleep 60
done
```

## 🚨 常见错误代码

### HTTP 状态码

| 状态码 | 错误类型 | 可能原因 | 解决方案 |
|--------|----------|----------|----------|
| **400** | Bad Request | 请求参数错误 | 检查请求格式和参数 |
| **401** | Unauthorized | 认证失败 | 检查 API Key 或 Token |
| **403** | Forbidden | 权限不足 | 检查用户权限设置 |
| **404** | Not Found | 资源不存在 | 检查 URL 路径和资源 ID |
| **429** | Too Many Requests | 请求过于频繁 | 实施限流或减少请求频率 |
| **500** | Internal Server Error | 服务器内部错误 | 查看服务器日志 |
| **502** | Bad Gateway | 网关错误 | 检查上游服务状态 |
| **503** | Service Unavailable | 服务不可用 | 检查服务健康状态 |

### 应用错误代码

| 错误代码 | 说明 | 解决方案 |
|----------|------|----------|
| `DB_CONNECTION_FAILED` | 数据库连接失败 | 检查数据库服务和连接配置 |
| `REDIS_CONNECTION_FAILED` | Redis 连接失败 | 检查 Redis 服务状态 |
| `OPENAI_API_ERROR` | OpenAI API 错误 | 检查 API Key 和配额 |
| `ASR_SERVICE_ERROR` | 语音识别服务错误 | 检查音频格式和服务状态 |
| `TTS_SERVICE_ERROR` | 语音合成服务错误 | 检查文本内容和语音模型 |
| `DOCUMENT_PROCESSING_FAILED` | 文档处理失败 | 检查文档格式和大小 |
| `VECTOR_SEARCH_FAILED` | 向量搜索失败 | 检查查询参数和索引状态 |

### 紧急恢复流程

```bash
#!/bin/bash
# scripts/emergency-recovery.sh

echo "🚨 启动紧急恢复流程..."

# 1. 停止所有服务
echo "🛑 停止所有服务..."
docker-compose down

# 2. 检查系统资源
echo "🔍 检查系统资源..."
df -h
free -h

# 3. 清理系统
echo "🧹 清理系统..."
docker system prune -f
docker volume prune -f

# 4. 恢复最近备份
echo "💾 恢复最近备份..."
LATEST_BACKUP=$(ls -t /backup/voicehelper/postgres_*.sql.gz | head -1)
if [ -n "$LATEST_BACKUP" ]; then
    echo "恢复备份: $LATEST_BACKUP"
    ./scripts/restore-data.sh $(basename $LATEST_BACKUP .sql.gz | cut -d'_' -f2-)
fi

# 5. 重启服务
echo "🚀 重启服务..."
docker-compose up -d

# 6. 健康检查
echo "🏥 执行健康检查..."
sleep 60
./scripts/health-check.sh

echo "✅ 紧急恢复完成！"
```

---

## 📞 获取帮助

如果以上解决方案都无法解决你的问题，可以通过以下方式获取帮助：

### 收集诊断信息

在寻求帮助前，请收集以下信息：

```bash
#!/bin/bash
# scripts/collect-diagnostic-info.sh

echo "📋 收集诊断信息..."

# 系统信息
echo "=== 系统信息 ===" > diagnostic-info.txt
uname -a >> diagnostic-info.txt
docker --version >> diagnostic-info.txt
docker-compose --version >> diagnostic-info.txt

# 服务状态
echo -e "\n=== 服务状态 ===" >> diagnostic-info.txt
docker-compose ps >> diagnostic-info.txt

# 最近日志
echo -e "\n=== 最近日志 ===" >> diagnostic-info.txt
docker-compose logs --tail=100 >> diagnostic-info.txt

# 系统资源
echo -e "\n=== 系统资源 ===" >> diagnostic-info.txt
free -h >> diagnostic-info.txt
df -h >> diagnostic-info.txt

echo "✅ 诊断信息已保存到 diagnostic-info.txt"
```

### 联系方式

- **GitHub Issues**: [提交问题](https://github.com/your-org/voicehelper/issues)
- **技术支持**: support@voicehelper.com
- **社区讨论**: [Discord](https://discord.gg/voicehelper)
- **文档中心**: [https://docs.voicehelper.com](https://docs.voicehelper.com)

### 问题模板

提交问题时，请使用以下模板：

```markdown
## 问题描述
简要描述遇到的问题

## 环境信息
- 操作系统: 
- Docker 版本: 
- VoiceHelper 版本: 

## 复现步骤
1. 
2. 
3. 

## 预期行为
描述期望的正常行为

## 实际行为
描述实际发生的情况

## 错误日志
```
粘贴相关的错误日志
```

## 已尝试的解决方案
列出已经尝试过的解决方法
```

---

**故障排除指南完成！** 🎉

希望这个指南能帮助你快速解决 VoiceHelper 使用过程中遇到的问题。如果有其他问题，欢迎随时联系我们！
