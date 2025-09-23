#!/bin/bash

# V2架构部署脚本
# 基于父类/子类设计模式的VoiceHelper部署

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    log_info "检查系统依赖..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker未安装，请先安装Docker"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose未安装，请先安装Docker Compose"
        exit 1
    fi
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装，请先安装Python3"
        exit 1
    fi
    
    # 检查Go
    if ! command -v go &> /dev/null; then
        log_error "Go未安装，请先安装Go"
        exit 1
    fi
    
    # 检查Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js未安装，请先安装Node.js"
        exit 1
    fi
    
    log_success "所有依赖检查通过"
}

# 构建算法服务
build_algo_service() {
    log_info "构建V2架构算法服务..."
    
    cd algo
    
    # 安装Python依赖
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    
    # 运行测试
    log_info "运行算法服务测试..."
    python -m pytest tests/ -v || log_warning "部分测试失败，继续部署"
    
    deactivate
    cd ..
    
    log_success "算法服务构建完成"
}

# 构建网关服务
build_gateway_service() {
    log_info "构建V2架构网关服务..."
    
    cd backend
    
    # 下载Go依赖
    go mod tidy
    
    # 构建服务
    go build -o server ./cmd/server/main.go
    
    # 运行测试
    log_info "运行网关服务测试..."
    go test ./... -v || log_warning "部分测试失败，继续部署"
    
    cd ..
    
    log_success "网关服务构建完成"
}

# 构建前端服务
build_frontend_service() {
    log_info "构建V2架构前端服务..."
    
    cd frontend
    
    # 安装Node.js依赖
    npm install
    
    # 构建前端
    npm run build
    
    # 运行测试
    log_info "运行前端测试..."
    npm test -- --watchAll=false || log_warning "部分测试失败，继续部署"
    
    cd ..
    
    log_success "前端服务构建完成"
}

# 创建V2架构配置
create_v2_config() {
    log_info "创建V2架构配置..."
    
    # 创建环境配置文件
    cat > .env.v2 << EOF
# V2架构环境配置

# 算法服务配置
ALGO_SERVICE_URL=http://localhost:8070
BGE_MODEL_NAME=BAAI/bge-large-zh-v1.5
BGE_DEVICE=cpu
FAISS_INDEX_TYPE=HNSW32,Flat
FAISS_EF_CONSTRUCTION=200
FAISS_EF_SEARCH=64

# 网关服务配置
GATEWAY_PORT=8080
ALGO_SERVICE_URL=http://localhost:8070

# 前端服务配置
NEXT_PUBLIC_API_URL=http://localhost:8080

# 数据库配置
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=voicehelper_v2
POSTGRES_USER=voicehelper
POSTGRES_PASSWORD=voicehelper123

# Redis配置
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# OpenAI配置（可选）
OPENAI_API_KEY=your_openai_api_key_here

# 日志配置
LOG_LEVEL=info
EOF

    log_success "V2架构配置创建完成"
}

# 创建Docker Compose配置
create_docker_compose() {
    log_info "创建V2架构Docker Compose配置..."
    
    cat > docker-compose.v2.yml << EOF
version: '3.8'

services:
  # PostgreSQL数据库
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: voicehelper_v2
      POSTGRES_USER: voicehelper
      POSTGRES_PASSWORD: voicehelper123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U voicehelper"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Redis缓存
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  # V2算法服务
  algo-v2:
    build:
      context: ./algo
      dockerfile: Dockerfile
    ports:
      - "8070:8070"
    environment:
      - BGE_MODEL_NAME=BAAI/bge-large-zh-v1.5
      - BGE_DEVICE=cpu
      - FAISS_INDEX_TYPE=HNSW32,Flat
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8070/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # V2网关服务
  gateway-v2:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8080:8080"
    environment:
      - ALGO_SERVICE_URL=http://algo-v2:8070
      - POSTGRES_HOST=postgres
      - REDIS_HOST=redis
    depends_on:
      algo-v2:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/api/v2/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # V2前端服务
  frontend-v2:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8080
    depends_on:
      gateway-v2:
        condition: service_healthy

volumes:
  postgres_data:
  redis_data:
EOF

    log_success "Docker Compose配置创建完成"
}

# 启动V2架构服务
start_v2_services() {
    log_info "启动V2架构服务..."
    
    # 启动基础设施服务
    docker-compose -f docker-compose.v2.yml up -d postgres redis
    
    # 等待数据库就绪
    log_info "等待数据库就绪..."
    sleep 10
    
    # 启动应用服务
    docker-compose -f docker-compose.v2.yml up -d
    
    log_success "V2架构服务启动完成"
}

# 验证部署
verify_deployment() {
    log_info "验证V2架构部署..."
    
    # 等待服务启动
    sleep 30
    
    # 检查算法服务
    if curl -f http://localhost:8070/api/v1/health > /dev/null 2>&1; then
        log_success "算法服务健康检查通过"
    else
        log_error "算法服务健康检查失败"
        return 1
    fi
    
    # 检查网关服务
    if curl -f http://localhost:8080/api/v2/health > /dev/null 2>&1; then
        log_success "网关服务健康检查通过"
    else
        log_error "网关服务健康检查失败"
        return 1
    fi
    
    # 检查前端服务
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        log_success "前端服务健康检查通过"
    else
        log_warning "前端服务可能需要更多时间启动"
    fi
    
    log_success "V2架构部署验证完成"
}

# 显示部署信息
show_deployment_info() {
    log_info "V2架构部署信息:"
    echo ""
    echo "🌐 服务地址:"
    echo "  - 前端服务: http://localhost:3000"
    echo "  - 网关API: http://localhost:8080"
    echo "  - 算法API: http://localhost:8070"
    echo ""
    echo "📊 监控地址:"
    echo "  - 网关健康检查: http://localhost:8080/api/v2/health"
    echo "  - 算法健康检查: http://localhost:8070/api/v1/health"
    echo ""
    echo "🔧 管理命令:"
    echo "  - 查看日志: docker-compose -f docker-compose.v2.yml logs -f"
    echo "  - 停止服务: docker-compose -f docker-compose.v2.yml down"
    echo "  - 重启服务: docker-compose -f docker-compose.v2.yml restart"
    echo ""
    echo "📝 配置文件:"
    echo "  - 环境配置: .env.v2"
    echo "  - Docker配置: docker-compose.v2.yml"
    echo ""
}

# 主函数
main() {
    log_info "开始部署VoiceHelper V2架构..."
    
    # 检查依赖
    check_dependencies
    
    # 构建服务
    build_algo_service
    build_gateway_service
    build_frontend_service
    
    # 创建配置
    create_v2_config
    create_docker_compose
    
    # 启动服务
    start_v2_services
    
    # 验证部署
    if verify_deployment; then
        show_deployment_info
        log_success "🎉 V2架构部署成功！"
    else
        log_error "❌ V2架构部署失败，请检查日志"
        exit 1
    fi
}

# 脚本入口
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
