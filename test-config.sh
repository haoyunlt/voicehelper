#!/bin/bash

# VoiceHelper AI - 配置文件测试脚本
# 验证 Docker Compose 配置文件的正确性

set -euo pipefail

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

echo -e "${BLUE}VoiceHelper AI - 配置文件测试${NC}"
echo "=================================="

# 检查配置文件是否存在
log_info "检查配置文件..."

files=(
    "docker-compose.yml"
    "docker-compose.dev.yml" 
    "docker-compose.prod.yml"
    "docker-compose.local.yml"
    "env.unified"
    "deploy.sh"
    "quick-start.sh"
    "Makefile"
)

missing_files=()
for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        log_success "✓ $file"
    else
        log_error "✗ $file"
        missing_files+=("$file")
    fi
done

if [[ ${#missing_files[@]} -gt 0 ]]; then
    log_error "缺少以下文件: ${missing_files[*]}"
    exit 1
fi

# 检查 Docker Compose 文件语法
log_info "检查 Docker Compose 文件语法..."

compose_files=(
    "docker-compose.yml"
    "docker-compose.dev.yml"
    "docker-compose.prod.yml"
    "docker-compose.local.yml"
)

for file in "${compose_files[@]}"; do
    if command -v docker-compose &> /dev/null; then
        if docker-compose -f "$file" config &> /dev/null; then
            log_success "✓ $file 语法正确"
        else
            log_error "✗ $file 语法错误"
            docker-compose -f "$file" config
        fi
    elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
        if docker compose -f "$file" config &> /dev/null; then
            log_success "✓ $file 语法正确"
        else
            log_error "✗ $file 语法错误"
            docker compose -f "$file" config
        fi
    else
        log_warning "Docker Compose 未安装，跳过语法检查"
        break
    fi
done

# 检查环境变量文件
log_info "检查环境变量配置..."

if [[ -f "env.unified" ]]; then
    # 检查必要的环境变量
    required_vars=(
        "POSTGRES_DB"
        "POSTGRES_USER" 
        "POSTGRES_PASSWORD"
        "REDIS_PASSWORD"
        "JWT_SECRET"
        "FRONTEND_PORT"
        "GATEWAY_PORT"
        "ALGO_PORT"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if grep -q "^${var}=" env.unified; then
            log_success "✓ $var"
        else
            log_error "✗ $var"
            missing_vars+=("$var")
        fi
    done
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        log_error "env.unified 中缺少以下变量: ${missing_vars[*]}"
    fi
else
    log_error "env.unified 文件不存在"
fi

# 检查脚本权限
log_info "检查脚本权限..."

scripts=("deploy.sh" "quick-start.sh")
for script in "${scripts[@]}"; do
    if [[ -x "$script" ]]; then
        log_success "✓ $script 可执行"
    else
        log_warning "✗ $script 不可执行，运行: chmod +x $script"
    fi
done

# 检查目录结构
log_info "检查目录结构..."

required_dirs=(
    "backend"
    "algo" 
    "platforms/web"
    "platforms/admin"
    "developer-portal"
    "tools/deployment/config"
    "tools/deployment/database"
)

missing_dirs=()
for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        log_success "✓ $dir/"
    else
        log_error "✗ $dir/"
        missing_dirs+=("$dir")
    fi
done

if [[ ${#missing_dirs[@]} -gt 0 ]]; then
    log_error "缺少以下目录: ${missing_dirs[*]}"
fi

# 检查 Dockerfile
log_info "检查 Dockerfile..."

dockerfiles=(
    "backend/Dockerfile"
    "algo/Dockerfile"
    "platforms/web/Dockerfile"
    "platforms/admin/Dockerfile"
    "developer-portal/Dockerfile"
)

for dockerfile in "${dockerfiles[@]}"; do
    if [[ -f "$dockerfile" ]]; then
        log_success "✓ $dockerfile"
    else
        log_error "✗ $dockerfile"
    fi
done

# 检查配置文件
log_info "检查配置文件..."

config_files=(
    "tools/deployment/config/nginx/nginx.conf"
    "tools/deployment/config/redis-optimized.conf"
    "tools/deployment/config/prometheus.yml"
    "tools/deployment/config/milvus.yaml"
    "tools/deployment/database/schema.sql"
)

for config in "${config_files[@]}"; do
    if [[ -f "$config" ]]; then
        log_success "✓ $config"
    else
        log_error "✗ $config"
    fi
done

# 生成测试报告
log_info "生成测试报告..."

cat > config-test-report.txt << EOF
VoiceHelper AI 配置文件测试报告
生成时间: $(date)

配置文件检查:
$(for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ $file"
    else
        echo "✗ $file"
    fi
done)

目录结构检查:
$(for dir in "${required_dirs[@]}"; do
    if [[ -d "$dir" ]]; then
        echo "✓ $dir/"
    else
        echo "✗ $dir/"
    fi
done)

建议的下一步:
1. 如果有缺失文件，请检查项目完整性
2. 运行 'cp env.unified .env' 创建环境配置
3. 运行 './deploy.sh check' 进行完整环境检查
4. 运行 './quick-start.sh' 开始部署

EOF

log_success "测试报告已保存到: config-test-report.txt"

echo ""
echo "=================================="
log_success "配置文件测试完成！"
echo ""
echo "下一步:"
echo "1. 查看测试报告: cat config-test-report.txt"
echo "2. 启动 Docker 服务"
echo "3. 运行: ./deploy.sh check"
echo "4. 运行: ./quick-start.sh"
