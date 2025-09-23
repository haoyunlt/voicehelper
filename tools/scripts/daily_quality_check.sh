#!/bin/bash

# VoiceHelper 每日质量检查脚本
# 执行各种代码质量检查和生成报告

set -e

# 配置
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT_DIR="$PROJECT_ROOT/reports/quality/$(date +%Y-%m-%d)"
LOG_FILE="$REPORT_DIR/quality_check.log"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 创建报告目录
mkdir -p "$REPORT_DIR"

# 日志函数
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') $1" | tee -a "$LOG_FILE"
}

log_info() {
    log "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    log "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    log "${RED}[ERROR]${NC} $1"
}

log_success() {
    log "${GREEN}[SUCCESS]${NC} $1"
}

# 检查工具是否存在
check_tool() {
    local tool=$1
    if ! command -v "$tool" &> /dev/null; then
        log_warn "$tool 未安装，跳过相关检查"
        return 1
    fi
    return 0
}

# 开始质量检查
log_info "开始每日质量检查 - $(date)"
log_info "项目根目录: $PROJECT_ROOT"
log_info "报告目录: $REPORT_DIR"

cd "$PROJECT_ROOT"

# 1. 代码统计
log_info "=== 代码统计 ==="
if check_tool "cloc"; then
    cloc --exclude-dir=node_modules,vendor,dist,build,.git --json . > "$REPORT_DIR/code_stats.json"
    cloc --exclude-dir=node_modules,vendor,dist,build,.git . > "$REPORT_DIR/code_stats.txt"
    log_success "代码统计完成"
else
    log_warn "cloc 未安装，跳过代码统计"
fi

# 2. Go代码质量检查
log_info "=== Go代码质量检查 ==="
if [ -d "backend" ]; then
    cd backend
    
    # Go vet
    if check_tool "go"; then
        log_info "运行 go vet..."
        go vet ./... > "$REPORT_DIR/go_vet.txt" 2>&1 || log_warn "go vet 发现问题"
        
        # Go fmt检查
        log_info "检查 go fmt..."
        gofmt -l . > "$REPORT_DIR/go_fmt.txt" 2>&1
        if [ -s "$REPORT_DIR/go_fmt.txt" ]; then
            log_warn "发现未格式化的Go文件"
        else
            log_success "所有Go文件格式正确"
        fi
        
        # Go mod tidy检查
        log_info "检查 go mod..."
        go mod tidy
        if git diff --quiet go.mod go.sum; then
            log_success "go.mod 和 go.sum 是最新的"
        else
            log_warn "go.mod 或 go.sum 需要更新"
        fi
    fi
    
    # golangci-lint
    if check_tool "golangci-lint"; then
        log_info "运行 golangci-lint..."
        golangci-lint run --out-format json > "$REPORT_DIR/golangci-lint.json" 2>&1 || true
        golangci-lint run > "$REPORT_DIR/golangci-lint.txt" 2>&1 || true
        log_success "golangci-lint 检查完成"
    fi
    
    # 安全检查
    if check_tool "gosec"; then
        log_info "运行安全检查 (gosec)..."
        gosec -fmt json -out "$REPORT_DIR/gosec.json" ./... 2>&1 || true
        gosec -fmt text -out "$REPORT_DIR/gosec.txt" ./... 2>&1 || true
        log_success "安全检查完成"
    fi
    
    cd ..
fi

# 3. Python代码质量检查
log_info "=== Python代码质量检查 ==="
if [ -d "algo" ]; then
    cd algo
    
    # 激活虚拟环境（如果存在）
    if [ -f "../voice_venv/bin/activate" ]; then
        source ../voice_venv/bin/activate
    fi
    
    # flake8
    if check_tool "flake8"; then
        log_info "运行 flake8..."
        flake8 --output-file="$REPORT_DIR/flake8.txt" . || log_warn "flake8 发现问题"
    fi
    
    # pylint
    if check_tool "pylint"; then
        log_info "运行 pylint..."
        pylint --output-format=json . > "$REPORT_DIR/pylint.json" 2>&1 || true
        pylint . > "$REPORT_DIR/pylint.txt" 2>&1 || true
    fi
    
    # bandit (安全检查)
    if check_tool "bandit"; then
        log_info "运行安全检查 (bandit)..."
        bandit -r . -f json -o "$REPORT_DIR/bandit.json" 2>&1 || true
        bandit -r . -f txt -o "$REPORT_DIR/bandit.txt" 2>&1 || true
    fi
    
    # mypy (类型检查)
    if check_tool "mypy"; then
        log_info "运行类型检查 (mypy)..."
        mypy . > "$REPORT_DIR/mypy.txt" 2>&1 || true
    fi
    
    cd ..
fi

# 4. JavaScript/TypeScript代码质量检查
log_info "=== JavaScript/TypeScript代码质量检查 ==="
if [ -d "frontend" ]; then
    cd frontend
    
    if [ -f "package.json" ]; then
        # ESLint
        if [ -f "node_modules/.bin/eslint" ]; then
            log_info "运行 ESLint..."
            ./node_modules/.bin/eslint . --ext .js,.jsx,.ts,.tsx --format json > "$REPORT_DIR/eslint.json" 2>&1 || true
            ./node_modules/.bin/eslint . --ext .js,.jsx,.ts,.tsx > "$REPORT_DIR/eslint.txt" 2>&1 || true
        fi
        
        # TypeScript检查
        if [ -f "node_modules/.bin/tsc" ]; then
            log_info "运行 TypeScript 检查..."
            ./node_modules/.bin/tsc --noEmit > "$REPORT_DIR/typescript.txt" 2>&1 || true
        fi
        
        # Prettier检查
        if [ -f "node_modules/.bin/prettier" ]; then
            log_info "检查代码格式 (Prettier)..."
            ./node_modules/.bin/prettier --check . > "$REPORT_DIR/prettier.txt" 2>&1 || true
        fi
    fi
    
    cd ..
fi

# 5. 依赖安全检查
log_info "=== 依赖安全检查 ==="

# Go依赖检查
if [ -d "backend" ] && check_tool "go"; then
    cd backend
    log_info "检查Go依赖安全性..."
    if check_tool "nancy"; then
        go list -json -m all | nancy sleuth > "$REPORT_DIR/go_deps_security.txt" 2>&1 || true
    fi
    cd ..
fi

# Python依赖检查
if [ -d "algo" ] && check_tool "safety"; then
    cd algo
    log_info "检查Python依赖安全性..."
    safety check --json > "$REPORT_DIR/python_deps_security.json" 2>&1 || true
    safety check > "$REPORT_DIR/python_deps_security.txt" 2>&1 || true
    cd ..
fi

# Node.js依赖检查
if [ -d "frontend" ] && [ -f "frontend/package.json" ]; then
    cd frontend
    log_info "检查Node.js依赖安全性..."
    if check_tool "npm"; then
        npm audit --json > "$REPORT_DIR/npm_audit.json" 2>&1 || true
        npm audit > "$REPORT_DIR/npm_audit.txt" 2>&1 || true
    fi
    cd ..
fi

# 6. 测试覆盖率检查
log_info "=== 测试覆盖率检查 ==="

# Go测试覆盖率
if [ -d "backend" ] && check_tool "go"; then
    cd backend
    log_info "运行Go测试并生成覆盖率报告..."
    go test -coverprofile="$REPORT_DIR/go_coverage.out" ./... > "$REPORT_DIR/go_test.txt" 2>&1 || true
    if [ -f "$REPORT_DIR/go_coverage.out" ]; then
        go tool cover -html="$REPORT_DIR/go_coverage.out" -o "$REPORT_DIR/go_coverage.html"
        go tool cover -func="$REPORT_DIR/go_coverage.out" > "$REPORT_DIR/go_coverage.txt"
    fi
    cd ..
fi

# Python测试覆盖率
if [ -d "algo" ] && check_tool "pytest"; then
    cd algo
    log_info "运行Python测试并生成覆盖率报告..."
    pytest --cov=. --cov-report=html:"$REPORT_DIR/python_coverage_html" --cov-report=term > "$REPORT_DIR/python_test.txt" 2>&1 || true
    cd ..
fi

# 7. 性能检查
log_info "=== 性能检查 ==="

# 检查大文件
log_info "检查大文件..."
find . -type f -size +10M -not -path "./.git/*" -not -path "./node_modules/*" -not -path "./vendor/*" > "$REPORT_DIR/large_files.txt" 2>&1 || true

# 检查重复文件
if check_tool "fdupes"; then
    log_info "检查重复文件..."
    fdupes -r . > "$REPORT_DIR/duplicate_files.txt" 2>&1 || true
fi

# 8. Git检查
log_info "=== Git检查 ==="
if [ -d ".git" ]; then
    # 检查未提交的更改
    log_info "检查Git状态..."
    git status --porcelain > "$REPORT_DIR/git_status.txt"
    
    # 检查最近的提交
    git log --oneline -10 > "$REPORT_DIR/recent_commits.txt"
    
    # 检查分支信息
    git branch -a > "$REPORT_DIR/git_branches.txt"
    
    # 检查代码贡献统计
    git shortlog -sn --since="1 week ago" > "$REPORT_DIR/contributors_week.txt"
fi

# 9. 文档检查
log_info "=== 文档检查 ==="

# 检查README文件
log_info "检查文档完整性..."
{
    echo "=== README文件检查 ==="
    find . -name "README*" -type f | head -10
    echo ""
    echo "=== API文档检查 ==="
    find . -name "*.md" -path "*/docs/*" -type f | head -10
    echo ""
    echo "=== 配置文件检查 ==="
    find . -name "*.yml" -o -name "*.yaml" -o -name "*.json" -type f | grep -E "(config|env)" | head -10
} > "$REPORT_DIR/documentation.txt"

# 10. 生成汇总报告
log_info "=== 生成汇总报告 ==="

cat > "$REPORT_DIR/summary.md" << EOF
# VoiceHelper 每日质量检查报告

**检查日期**: $(date '+%Y-%m-%d %H:%M:%S')
**项目根目录**: $PROJECT_ROOT

## 检查项目

### 1. 代码统计
- 代码行数统计: [code_stats.txt](code_stats.txt)

### 2. Go代码质量
- Go Vet: [go_vet.txt](go_vet.txt)
- Go Fmt: [go_fmt.txt](go_fmt.txt)
- GolangCI-Lint: [golangci-lint.txt](golangci-lint.txt)
- 安全检查: [gosec.txt](gosec.txt)

### 3. Python代码质量
- Flake8: [flake8.txt](flake8.txt)
- Pylint: [pylint.txt](pylint.txt)
- Bandit: [bandit.txt](bandit.txt)
- MyPy: [mypy.txt](mypy.txt)

### 4. JavaScript/TypeScript代码质量
- ESLint: [eslint.txt](eslint.txt)
- TypeScript: [typescript.txt](typescript.txt)
- Prettier: [prettier.txt](prettier.txt)

### 5. 依赖安全检查
- Go依赖: [go_deps_security.txt](go_deps_security.txt)
- Python依赖: [python_deps_security.txt](python_deps_security.txt)
- Node.js依赖: [npm_audit.txt](npm_audit.txt)

### 6. 测试覆盖率
- Go测试: [go_test.txt](go_test.txt), [覆盖率报告](go_coverage.html)
- Python测试: [python_test.txt](python_test.txt)

### 7. 性能检查
- 大文件: [large_files.txt](large_files.txt)
- 重复文件: [duplicate_files.txt](duplicate_files.txt)

### 8. Git检查
- Git状态: [git_status.txt](git_status.txt)
- 最近提交: [recent_commits.txt](recent_commits.txt)
- 分支信息: [git_branches.txt](git_branches.txt)
- 贡献统计: [contributors_week.txt](contributors_week.txt)

### 9. 文档检查
- 文档完整性: [documentation.txt](documentation.txt)

## 建议

$(if [ -s "$REPORT_DIR/go_fmt.txt" ]; then echo "- 🔧 运行 \`go fmt\` 格式化Go代码"; fi)
$(if [ -s "$REPORT_DIR/git_status.txt" ]; then echo "- 📝 提交未保存的更改"; fi)
$(if [ -s "$REPORT_DIR/large_files.txt" ]; then echo "- 🗂️ 检查并清理大文件"; fi)

---
*报告生成时间: $(date)*
EOF

# 11. 发送通知（如果配置了）
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    log_info "发送Slack通知..."
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"VoiceHelper 每日质量检查完成 - $(date '+%Y-%m-%d')\"}" \
        "$SLACK_WEBHOOK_URL" || true
fi

if [ -n "$TEAMS_WEBHOOK_URL" ]; then
    log_info "发送Teams通知..."
    curl -X POST -H 'Content-type: application/json' \
        --data "{\"text\":\"VoiceHelper 每日质量检查完成 - $(date '+%Y-%m-%d')\"}" \
        "$TEAMS_WEBHOOK_URL" || true
fi

log_success "每日质量检查完成！"
log_info "报告位置: $REPORT_DIR"
log_info "汇总报告: $REPORT_DIR/summary.md"

# 清理旧报告（保留最近7天）
find "$PROJECT_ROOT/reports/quality" -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true

echo -e "\n${GREEN}质量检查完成！查看报告：${NC}"
echo -e "${BLUE}file://$REPORT_DIR/summary.md${NC}"
