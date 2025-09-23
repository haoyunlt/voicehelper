# VoiceHelper AI - 完整的开发和部署 Makefile
# 版本: 2.0.0

.PHONY: help install dev prod local test build clean logs status health backup restore

# 默认目标
.DEFAULT_GOAL := help

# 颜色定义
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# 项目配置
PROJECT_NAME := voicehelper

help: ## 显示帮助信息
	@echo -e "${BLUE}VoiceHelper AI - 开发和部署工具${NC}"
	@echo ""
	@echo -e "${GREEN}🚀 快速启动:${NC}"
	@echo "  make quick-start    一键启动完整系统"
	@echo ""
	@echo -e "${GREEN}🏗️  环境管理:${NC}"
	@echo "  make dev           启动开发环境"
	@echo "  make prod          启动生产环境"
	@echo "  make local         启动本地环境"
	@echo "  make stop          停止所有服务"
	@echo "  make restart       重启所有服务"
	@echo ""
	@echo -e "${GREEN}🔧 开发工具:${NC}"
	@echo "  make build         构建所有镜像"
	@echo "  make pull          拉取最新镜像"
	@echo "  make logs          查看服务日志"
	@echo "  make status        查看服务状态"
	@echo "  make health        健康检查"
	@echo ""
	@echo -e "${GREEN}💾 数据管理:${NC}"
	@echo "  make backup        备份数据"
	@echo "  make restore       恢复数据"
	@echo "  make clean         清理资源"
	@echo ""
	@echo -e "${GREEN}🧪 测试和质量:${NC}"
	@echo "  make test          运行测试"
	@echo "  make lint          代码检查"
	@echo "  make format        代码格式化"
	@echo ""

quick-start: ## 一键启动完整系统
	@echo -e "${GREEN}🚀 启动 VoiceHelper AI...${NC}"
	@./quick-start.sh

dev: ## 启动开发环境
	@echo -e "${GREEN}🔧 启动开发环境...${NC}"
	@./deploy.sh -e dev up -d
	@make _show-dev-info

prod: ## 启动生产环境
	@echo -e "${GREEN}🏭 启动生产环境...${NC}"
	@./deploy.sh -e prod up -d
	@make _show-prod-info

local: ## 启动本地环境
	@echo -e "${GREEN}💻 启动本地环境...${NC}"
	@docker-compose -f docker-compose.local.yml up -d
	@make _show-local-info

stop: ## 停止所有服务
	@echo -e "${YELLOW}⏹️  停止所有服务...${NC}"
	@./deploy.sh down

restart: ## 重启所有服务
	@echo -e "${YELLOW}🔄 重启所有服务...${NC}"
	@./deploy.sh restart

build: ## 构建所有镜像
	@echo -e "${BLUE}🔨 构建所有镜像...${NC}"
	@./deploy.sh build

pull: ## 拉取最新镜像
	@echo -e "${BLUE}📥 拉取最新镜像...${NC}"
	@./deploy.sh pull

logs: ## 查看服务日志
	@echo -e "${BLUE}📋 查看服务日志...${NC}"
	@./deploy.sh logs

status: ## 查看服务状态
	@echo -e "${BLUE}📊 查看服务状态...${NC}"
	@./deploy.sh status

health: ## 健康检查
	@echo -e "${BLUE}🏥 执行健康检查...${NC}"
	@./deploy.sh health

backup: ## 备份数据
	@echo -e "${BLUE}💾 备份数据...${NC}"
	@./deploy.sh backup

restore: ## 恢复数据
	@echo -e "${BLUE}🔄 恢复数据...${NC}"
	@if [ -z "$(BACKUP_DIR)" ]; then \
		echo -e "${RED}错误: 请指定备份目录${NC}"; \
		echo "示例: make restore BACKUP_DIR=./backups/20231201_120000"; \
		exit 1; \
	fi
	@./deploy.sh restore $(BACKUP_DIR)

clean: ## 清理未使用的资源
	@echo -e "${YELLOW}🧹 清理未使用的资源...${NC}"
	@./deploy.sh -f clean

test: ## 运行测试
	@echo -e "${BLUE}🧪 运行测试...${NC}"
	@./tools/scripts/run_tests.sh

lint: ## 代码检查
	@echo -e "${BLUE}🔍 执行代码检查...${NC}"
	@echo "检查 Python 代码..."
	@flake8 algo/ --max-line-length=88 --exclude=__pycache__,venv || true
	@echo "检查 Go 代码..."
	@cd backend && golangci-lint run || true

format: ## 代码格式化
	@echo -e "${BLUE}✨ 格式化代码...${NC}"
	@echo "格式化 Python 代码..."
	@black algo/ --line-length=88 || true
	@echo "格式化 Go 代码..."
	@cd backend && gofmt -w . || true

# 核心服务
core: ## 仅启动核心服务
	@echo -e "${GREEN}🎯 启动核心服务...${NC}"
	@./deploy.sh -p core up -d

# 监控服务
monitoring: ## 仅启动监控服务
	@echo -e "${GREEN}📊 启动监控服务...${NC}"
	@./deploy.sh -p monitoring up -d

# 开发工具
tools: ## 仅启动开发工具
	@echo -e "${GREEN}🛠️  启动开发工具...${NC}"
	@./deploy.sh -p tools up -d

# 扩缩容
scale-algo: ## 扩容算法服务
	@echo -e "${BLUE}📈 扩容算法服务到 $(REPLICAS) 个实例...${NC}"
	@./deploy.sh scale algo-service=$(REPLICAS)

scale-gateway: ## 扩容网关服务
	@echo -e "${BLUE}📈 扩容网关服务到 $(REPLICAS) 个实例...${NC}"
	@./deploy.sh scale gateway=$(REPLICAS)

# 内部辅助函数
_show-dev-info:
	@echo ""
	@echo -e "${GREEN}✅ 开发环境启动完成！${NC}"
	@echo ""
	@echo "🌐 访问地址:"
	@echo "  Web 应用:    http://localhost:3000"
	@echo "  API 网关:    http://localhost:8080"
	@echo "  算法服务:    http://localhost:8000"
	@echo "  管理后台:    http://localhost:5001"
	@echo ""

_show-prod-info:
	@echo ""
	@echo -e "${GREEN}✅ 生产环境启动完成！${NC}"
	@echo ""
	@echo "🌐 访问地址:"
	@echo "  Web 应用:    http://localhost:80"
	@echo "  API 网关:    http://localhost:8080"
	@echo ""

_show-local-info:
	@echo ""
	@echo -e "${GREEN}✅ 本地环境启动完成！${NC}"
	@echo ""
	@echo "🌐 访问地址:"
	@echo "  Web 应用:    http://localhost:3000"
	@echo "  API 网关:    http://localhost:8080"
	@echo ""

# 安装依赖
install: ## 安装开发依赖
	@echo -e "${BLUE}📦 安装开发依赖...${NC}"
	@if command -v brew >/dev/null 2>&1; then \
		brew install docker docker-compose; \
	elif command -v apt-get >/dev/null 2>&1; then \
		sudo apt-get update && sudo apt-get install -y docker.io docker-compose; \
	else \
		echo -e "${RED}请手动安装 Docker 和 Docker Compose${NC}"; \
	fi

# 环境检查
check: ## 检查环境配置
	@echo -e "${BLUE}🔍 检查环境配置...${NC}"
	@echo "Docker 版本:"
	@docker --version
	@echo "Docker Compose 版本:"
	@docker-compose --version || docker compose version
	@echo "环境配置文件:"
	@if [ -f .env ]; then echo "✅ .env 文件存在"; else echo "❌ .env 文件不存在"; fi