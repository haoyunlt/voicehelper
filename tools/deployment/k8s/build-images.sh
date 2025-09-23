#!/bin/bash

# VoiceHelper Docker 镜像构建脚本
set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
REGISTRY="voicehelper"
VERSION="latest"
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

echo -e "${BLUE}🚀 VoiceHelper Docker 镜像构建开始${NC}"
echo -e "${BLUE}项目根目录: ${PROJECT_ROOT}${NC}"
echo -e "${BLUE}镜像版本: ${VERSION}${NC}"
echo -e "${BLUE}构建时间: ${BUILD_DATE}${NC}"
echo -e "${BLUE}Git提交: ${GIT_COMMIT}${NC}"

# 函数：构建镜像
build_image() {
    local service_name=$1
    local dockerfile_path=$2
    local context_path=$3
    local image_name="${REGISTRY}/${service_name}:${VERSION}"
    
    echo -e "\n${YELLOW}📦 构建 ${service_name} 镜像...${NC}"
    
    if [ ! -f "${dockerfile_path}" ]; then
        echo -e "${RED}❌ Dockerfile 不存在: ${dockerfile_path}${NC}"
        return 1
    fi
    
    # 构建镜像
    docker build \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg GIT_COMMIT="${GIT_COMMIT}" \
        --build-arg VERSION="${VERSION}" \
        -t "${image_name}" \
        -f "${dockerfile_path}" \
        "${context_path}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ ${service_name} 镜像构建成功: ${image_name}${NC}"
        
        # 显示镜像信息
        docker images "${image_name}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
        
        return 0
    else
        echo -e "${RED}❌ ${service_name} 镜像构建失败${NC}"
        return 1
    fi
}

# 创建 Dockerfile - Go 后端网关
create_gateway_dockerfile() {
    cat > "${PROJECT_ROOT}/backend/Dockerfile" << 'EOF'
# 多阶段构建 - Go 后端网关服务
FROM golang:1.21-alpine AS builder

# 安装必要工具
RUN apk add --no-cache git ca-certificates tzdata

# 设置工作目录
WORKDIR /app

# 复制 go mod 文件
COPY go.mod go.sum ./

# 下载依赖
RUN go mod download

# 复制源代码
COPY . .

# 构建应用
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags '-extldflags "-static" -X main.version=${VERSION} -X main.buildDate=${BUILD_DATE} -X main.gitCommit=${GIT_COMMIT}' \
    -o gateway ./cmd/gateway

RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags '-extldflags "-static"' \
    -o persistence ./cmd/persistence

RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo \
    -ldflags '-extldflags "-static"' \
    -o monitoring ./cmd/monitoring

# 最终镜像
FROM alpine:3.18

# 安装运行时依赖
RUN apk --no-cache add ca-certificates tzdata curl

# 创建非root用户
RUN addgroup -g 1000 voicehelper && \
    adduser -D -s /bin/sh -u 1000 -G voicehelper voicehelper

# 设置工作目录
WORKDIR /app

# 从构建阶段复制二进制文件
COPY --from=builder /app/gateway /app/
COPY --from=builder /app/persistence /app/
COPY --from=builder /app/monitoring /app/

# 创建日志目录
RUN mkdir -p /var/log/voicehelper && \
    chown -R voicehelper:voicehelper /app /var/log/voicehelper

# 切换到非root用户
USER voicehelper

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 暴露端口
EXPOSE 8080 8081

# 启动命令
CMD ["./gateway"]
EOF
}

# 创建 Dockerfile - Python 算法服务
create_algo_dockerfile() {
    cat > "${PROJECT_ROOT}/algo/Dockerfile" << 'EOF'
# Python 算法服务
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd -r voicehelper && useradd -r -g voicehelper voicehelper

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements*.txt ./

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置权限
RUN chown -R voicehelper:voicehelper /app

# 创建缓存目录
RUN mkdir -p /root/.cache && chown -R voicehelper:voicehelper /root/.cache

# 切换到非root用户
USER voicehelper

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
EOF
}

# 创建 Dockerfile - 前端服务
create_frontend_dockerfile() {
    cat > "${PROJECT_ROOT}/platforms/web/Dockerfile" << 'EOF'
# 多阶段构建 - Next.js 前端应用
FROM node:18-alpine AS builder

# 安装依赖
RUN apk add --no-cache libc6-compat

# 设置工作目录
WORKDIR /app

# 复制package文件
COPY package*.json ./

# 安装依赖
RUN npm ci --only=production

# 复制源代码
COPY . .

# 构建应用
RUN npm run build

# 生产镜像
FROM node:18-alpine AS runner

# 安装运行时依赖
RUN apk add --no-cache curl

# 创建非root用户
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# 设置工作目录
WORKDIR /app

# 复制构建产物
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# 设置权限
RUN chown -R nextjs:nodejs /app

# 切换到非root用户
USER nextjs

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/ || exit 1

# 暴露端口
EXPOSE 3000

# 启动命令
CMD ["node", "server.js"]
EOF
}

# 创建 Dockerfile - 语音服务
create_voice_dockerfile() {
    cat > "${PROJECT_ROOT}/platforms/voice/Dockerfile" << 'EOF'
# Python 语音服务
FROM python:3.11-slim

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN groupadd -r voicehelper && useradd -r -g voicehelper voicehelper

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt ./

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 设置权限
RUN chown -R voicehelper:voicehelper /app

# 创建临时目录
RUN mkdir -p /tmp/audio && chown -R voicehelper:voicehelper /tmp/audio

# 切换到非root用户
USER voicehelper

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# 暴露端口
EXPOSE 8001 8002

# 启动命令
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
EOF
}

# 主构建流程
main() {
    echo -e "\n${BLUE}📝 创建 Dockerfile 文件...${NC}"
    
    # 创建各服务的 Dockerfile
    create_gateway_dockerfile
    create_algo_dockerfile
    create_frontend_dockerfile
    create_voice_dockerfile
    
    echo -e "${GREEN}✅ Dockerfile 文件创建完成${NC}"
    
    # 构建镜像
    echo -e "\n${BLUE}🔨 开始构建镜像...${NC}"
    
    # 构建 Go 后端网关
    if build_image "gateway" "${PROJECT_ROOT}/backend/Dockerfile" "${PROJECT_ROOT}/backend"; then
        echo -e "${GREEN}✅ Gateway 镜像构建成功${NC}"
    else
        echo -e "${RED}❌ Gateway 镜像构建失败${NC}"
        exit 1
    fi
    
    # 构建 Python 算法服务
    if build_image "algo-service" "${PROJECT_ROOT}/algo/Dockerfile" "${PROJECT_ROOT}/algo"; then
        echo -e "${GREEN}✅ Algo Service 镜像构建成功${NC}"
    else
        echo -e "${RED}❌ Algo Service 镜像构建失败${NC}"
        exit 1
    fi
    
    # 构建前端服务
    if build_image "frontend" "${PROJECT_ROOT}/platforms/web/Dockerfile" "${PROJECT_ROOT}/platforms/web"; then
        echo -e "${GREEN}✅ Frontend 镜像构建成功${NC}"
    else
        echo -e "${RED}❌ Frontend 镜像构建失败${NC}"
        exit 1
    fi
    
    # 构建语音服务（如果存在）
    if [ -d "${PROJECT_ROOT}/platforms/voice" ]; then
        if build_image "voice-service" "${PROJECT_ROOT}/platforms/voice/Dockerfile" "${PROJECT_ROOT}/platforms/voice"; then
            echo -e "${GREEN}✅ Voice Service 镜像构建成功${NC}"
        else
            echo -e "${RED}❌ Voice Service 镜像构建失败${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️  Voice Service 目录不存在，跳过构建${NC}"
    fi
    
    echo -e "\n${GREEN}🎉 所有镜像构建完成！${NC}"
    
    # 显示构建的镜像
    echo -e "\n${BLUE}📋 构建的镜像列表:${NC}"
    docker images "${REGISTRY}/*:${VERSION}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    
    # 保存镜像信息
    echo -e "\n${BLUE}💾 保存镜像信息...${NC}"
    docker images "${REGISTRY}/*:${VERSION}" --format "{{.Repository}}:{{.Tag}}" > "${PROJECT_ROOT}/tools/deployment/k8s/built-images.txt"
    
    echo -e "\n${GREEN}✅ 镜像构建完成！可以使用以下命令部署到 Kubernetes:${NC}"
    echo -e "${YELLOW}cd ${PROJECT_ROOT}/tools/deployment/k8s${NC}"
    echo -e "${YELLOW}./deploy.sh${NC}"
}

# 检查 Docker 是否运行
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker 未运行或无法访问${NC}"
    exit 1
fi

# 检查项目根目录
if [ ! -f "${PROJECT_ROOT}/backend/go.mod" ]; then
    echo -e "${RED}❌ 项目根目录不正确: ${PROJECT_ROOT}${NC}"
    exit 1
fi

# 运行主函数
main "$@"
