#!/bin/bash

# 跨平台编译脚本 - 支持 Linux 和 macOS
# Usage: ./scripts/build.sh [service] [platform]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目信息
PROJECT_NAME="chatbot"
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME=$(date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# 构建信息
LDFLAGS="-X main.Version=${VERSION} -X main.BuildTime=${BUILD_TIME} -X main.GitCommit=${GIT_COMMIT} -s -w"

# 支持的平台
PLATFORMS=(
    "linux/amd64"
    "linux/arm64"
    "darwin/amd64"
    "darwin/arm64"
)

# 服务列表
SERVICES=(
    "backend"
    "algo"
    "admin"
)

# 输出目录
BUILD_DIR="build"
DIST_DIR="dist"

# 帮助信息
show_help() {
    echo -e "${BLUE}跨平台编译脚本${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] [SERVICE] [PLATFORM]"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help     显示帮助信息"
    echo "  -c, --clean    清理构建目录"
    echo "  -a, --all      构建所有服务和平台"
    echo "  -r, --release  发布模式（优化编译）"
    echo "  -v, --verbose  详细输出"
    echo ""
    echo "SERVICE:"
    echo "  backend        Go后端服务"
    echo "  algo           Python算法服务"
    echo "  admin          Flask管理后台"
    echo "  all            所有服务"
    echo ""
    echo "PLATFORM:"
    echo "  linux/amd64    Linux x86_64"
    echo "  linux/arm64    Linux ARM64"
    echo "  darwin/amd64   macOS x86_64"
    echo "  darwin/arm64   macOS ARM64 (Apple Silicon)"
    echo "  all            所有平台"
    echo ""
    echo "Examples:"
    echo "  $0 backend                    # 构建后端服务（当前平台）"
    echo "  $0 backend linux/amd64        # 构建后端服务（Linux x64）"
    echo "  $0 --all                      # 构建所有服务和平台"
    echo "  $0 --clean                    # 清理构建目录"
}

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 清理构建目录
clean_build() {
    log_info "清理构建目录..."
    rm -rf ${BUILD_DIR}
    rm -rf ${DIST_DIR}
    log_info "构建目录已清理"
}

# 检查依赖
check_dependencies() {
    log_info "检查构建依赖..."
    
    # 检查Go
    if ! command -v go &> /dev/null; then
        log_error "Go未安装，请先安装Go 1.19+"
        exit 1
    fi
    
    GO_VERSION=$(go version | awk '{print $3}' | sed 's/go//')
    log_info "Go版本: ${GO_VERSION}"
    
    # 检查Python（用于算法服务）
    if ! command -v python3 &> /dev/null; then
        log_warn "Python3未安装，将跳过算法服务构建"
    else
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        log_info "Python版本: ${PYTHON_VERSION}"
    fi
    
    # 检查Docker（用于容器化构建）
    if command -v docker &> /dev/null; then
        log_info "Docker可用，支持容器化构建"
    fi
}

# 获取当前平台
get_current_platform() {
    local os=$(uname -s | tr '[:upper:]' '[:lower:]')
    local arch=$(uname -m)
    
    case $os in
        linux)
            os="linux"
            ;;
        darwin)
            os="darwin"
            ;;
        *)
            log_error "不支持的操作系统: $os"
            exit 1
            ;;
    esac
    
    case $arch in
        x86_64|amd64)
            arch="amd64"
            ;;
        arm64|aarch64)
            arch="arm64"
            ;;
        *)
            log_error "不支持的架构: $arch"
            exit 1
            ;;
    esac
    
    echo "${os}/${arch}"
}

# 构建Go服务
build_go_service() {
    local service=$1
    local platform=$2
    local output_dir=$3
    
    local goos=$(echo $platform | cut -d'/' -f1)
    local goarch=$(echo $platform | cut -d'/' -f2)
    
    log_info "构建 ${service} for ${platform}..."
    
    # 设置输出文件名
    local binary_name="${service}"
    if [ "$goos" = "linux" ]; then
        binary_name="${service}-linux-${goarch}"
    elif [ "$goos" = "darwin" ]; then
        binary_name="${service}-darwin-${goarch}"
    fi
    
    # 构建命令
    local build_cmd="GOOS=${goos} GOARCH=${goarch} CGO_ENABLED=0 go build"
    
    if [ "$RELEASE_MODE" = "true" ]; then
        build_cmd="${build_cmd} -ldflags='${LDFLAGS}' -trimpath"
    else
        build_cmd="${build_cmd} -ldflags='${LDFLAGS}'"
    fi
    
    # 根据服务类型设置源码路径
    case $service in
        backend)
            build_cmd="(cd backend && ${build_cmd} -o ../${output_dir}/${binary_name} ./cmd/server)"
            ;;
        *)
            log_error "未知的Go服务: $service"
            return 1
            ;;
    esac
    
    # 执行构建
    if [ "$VERBOSE" = "true" ]; then
        log_info "执行命令: $build_cmd"
    fi
    
    eval $build_cmd
    
    if [ $? -eq 0 ]; then
        log_info "✓ ${service} (${platform}) 构建成功"
        
        # 显示文件信息
        if [ -f "${output_dir}/${binary_name}" ]; then
            local file_size=$(ls -lh ${output_dir}/${binary_name} | awk '{print $5}')
            log_info "  文件大小: ${file_size}"
            
            # 设置执行权限
            chmod +x ${output_dir}/${binary_name}
        else
            log_warn "  构建文件未找到: ${output_dir}/${binary_name}"
        fi
        
        return 0
    else
        log_error "✗ ${service} (${platform}) 构建失败"
        return 1
    fi
}

# 构建Python服务
build_python_service() {
    local service=$1
    local platform=$2
    local output_dir=$3
    
    log_info "构建 ${service} for ${platform}..."
    
    case $service in
        algo)
            # 创建Python应用包
            local app_dir="${output_dir}/${service}-${platform//\//-}"
            mkdir -p ${app_dir}
            
            # 复制源码
            cp -r algo/* ${app_dir}/
            
            # 创建启动脚本
            cat > ${app_dir}/start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8001
EOF
            chmod +x ${app_dir}/start.sh
            
            # 创建requirements.txt（如果不存在）
            if [ ! -f ${app_dir}/requirements.txt ]; then
                cat > ${app_dir}/requirements.txt << 'EOF'
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
langchain>=0.0.350
langchain-community>=0.0.10
redis>=5.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentence-transformers>=2.2.0
openai>=1.0.0
pydantic>=2.0.0
python-multipart>=0.0.6
aiofiles>=23.0.0
EOF
            fi
            
            log_info "✓ ${service} (${platform}) 打包成功"
            ;;
        admin)
            # 创建Flask应用包
            local app_dir="${output_dir}/${service}-${platform//\//-}"
            mkdir -p ${app_dir}
            
            # 复制源码
            cp -r admin/* ${app_dir}/
            
            # 创建启动脚本
            cat > ${app_dir}/start.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
python3 app.py
EOF
            chmod +x ${app_dir}/start.sh
            
            # 创建requirements.txt（如果不存在）
            if [ ! -f ${app_dir}/requirements.txt ]; then
                cat > ${app_dir}/requirements.txt << 'EOF'
Flask>=3.0.0
Flask-CORS>=4.0.0
redis>=5.0.0
psycopg2-binary>=2.9.0
requests>=2.31.0
pandas>=2.0.0
plotly>=5.17.0
gunicorn>=21.2.0
EOF
            fi
            
            log_info "✓ ${service} (${platform}) 打包成功"
            ;;
        *)
            log_error "未知的Python服务: $service"
            return 1
            ;;
    esac
    
    return 0
}

# 构建单个服务
build_service() {
    local service=$1
    local platform=$2
    
    # 创建输出目录
    local output_dir="${BUILD_DIR}/${platform//\//-}"
    mkdir -p ${output_dir}
    
    case $service in
        backend)
            build_go_service $service $platform $output_dir
            ;;
        algo|admin)
            build_python_service $service $platform $output_dir
            ;;
        *)
            log_error "未知服务: $service"
            return 1
            ;;
    esac
}

# 创建发布包
create_release_package() {
    log_info "创建发布包..."
    
    mkdir -p ${DIST_DIR}
    
    for platform in "${PLATFORMS[@]}"; do
        local platform_dir="${BUILD_DIR}/${platform//\//-}"
        
        if [ -d "$platform_dir" ]; then
            local archive_name="${PROJECT_NAME}-${VERSION}-${platform//\//-}"
            
            log_info "打包 ${archive_name}..."
            
            # 创建临时目录
            local temp_dir="/tmp/${archive_name}"
            rm -rf ${temp_dir}
            mkdir -p ${temp_dir}
            
            # 复制二进制文件
            cp -r ${platform_dir}/* ${temp_dir}/
            
            # 复制配置文件和文档
            cp -r deploy/config ${temp_dir}/ 2>/dev/null || true
            cp README.md ${temp_dir}/ 2>/dev/null || true
            cp README-DEPLOY.md ${temp_dir}/ 2>/dev/null || true
            
            # 创建启动脚本
            cat > ${temp_dir}/start.sh << 'EOF'
#!/bin/bash
# 启动脚本

set -e

# 检测平台
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case $ARCH in
    x86_64|amd64)
        ARCH="amd64"
        ;;
    arm64|aarch64)
        ARCH="arm64"
        ;;
    *)
        echo "不支持的架构: $ARCH"
        exit 1
        ;;
esac

PLATFORM="${OS}-${ARCH}"

# 启动后端服务
if [ -f "backend-${PLATFORM}" ]; then
    echo "启动后端服务..."
    ./backend-${PLATFORM} &
    BACKEND_PID=$!
fi

# 启动算法服务
if [ -d "algo-${PLATFORM}" ]; then
    echo "启动算法服务..."
    cd algo-${PLATFORM}
    ./start.sh &
    ALGO_PID=$!
    cd ..
fi

# 启动管理后台
if [ -d "admin-${PLATFORM}" ]; then
    echo "启动管理后台..."
    cd admin-${PLATFORM}
    ./start.sh &
    ADMIN_PID=$!
    cd ..
fi

echo "所有服务已启动"
echo "后端服务: http://localhost:8080"
echo "算法服务: http://localhost:8001"
echo "管理后台: http://localhost:5000"

# 等待信号
trap 'kill $BACKEND_PID $ALGO_PID $ADMIN_PID 2>/dev/null; exit' INT TERM

wait
EOF
            chmod +x ${temp_dir}/start.sh
            
            # 创建压缩包
            cd /tmp
            tar -czf "${DIST_DIR}/${archive_name}.tar.gz" ${archive_name}
            
            # 清理临时目录
            rm -rf ${temp_dir}
            
            log_info "✓ ${archive_name}.tar.gz 创建成功"
        fi
    done
}

# 显示构建结果
show_build_results() {
    log_info "构建完成！"
    echo ""
    echo "构建产物："
    
    if [ -d "$BUILD_DIR" ]; then
        find $BUILD_DIR -type f -executable -o -name "*.tar.gz" | while read file; do
            local size=$(ls -lh "$file" | awk '{print $5}')
            echo "  $file ($size)"
        done
    fi
    
    if [ -d "$DIST_DIR" ]; then
        echo ""
        echo "发布包："
        find $DIST_DIR -name "*.tar.gz" | while read file; do
            local size=$(ls -lh "$file" | awk '{print $5}')
            echo "  $file ($size)"
        done
    fi
}

# 主函数
main() {
    local service=""
    local platform=""
    local build_all=false
    local clean_only=false
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--clean)
                clean_only=true
                shift
                ;;
            -a|--all)
                build_all=true
                shift
                ;;
            -r|--release)
                RELEASE_MODE=true
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            *)
                if [ -z "$service" ]; then
                    service=$1
                elif [ -z "$platform" ]; then
                    platform=$1
                else
                    log_error "未知参数: $1"
                    show_help
                    exit 1
                fi
                shift
                ;;
        esac
    done
    
    # 清理模式
    if [ "$clean_only" = true ]; then
        clean_build
        exit 0
    fi
    
    # 检查依赖
    check_dependencies
    
    # 创建构建目录
    mkdir -p ${BUILD_DIR}
    
    log_info "开始构建 ${PROJECT_NAME} v${VERSION}"
    log_info "构建时间: ${BUILD_TIME}"
    log_info "Git提交: ${GIT_COMMIT}"
    
    # 构建所有
    if [ "$build_all" = true ]; then
        for service in "${SERVICES[@]}"; do
            for platform in "${PLATFORMS[@]}"; do
                build_service $service $platform
            done
        done
        create_release_package
    else
        # 设置默认值
        if [ -z "$service" ]; then
            service="backend"
        fi
        
        if [ -z "$platform" ]; then
            platform=$(get_current_platform)
        fi
        
        # 构建指定服务和平台
        if [ "$service" = "all" ]; then
            for svc in "${SERVICES[@]}"; do
                if [ "$platform" = "all" ]; then
                    for plt in "${PLATFORMS[@]}"; do
                        build_service $svc $plt
                    done
                else
                    build_service $svc $platform
                fi
            done
        else
            if [ "$platform" = "all" ]; then
                for plt in "${PLATFORMS[@]}"; do
                    build_service $service $plt
                done
            else
                build_service $service $platform
            fi
        fi
        
        # 如果构建了多个平台，创建发布包
        if [ "$platform" = "all" ] || [ "$build_all" = true ]; then
            create_release_package
        fi
    fi
    
    show_build_results
}

# 执行主函数
main "$@"
