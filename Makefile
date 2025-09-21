# Chatbot 项目 Makefile
# 支持 Linux 和 macOS 跨平台构建

.PHONY: help build clean test docker release install deps lint format check

# 项目信息
PROJECT_NAME := chatbot
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BUILD_TIME := $(shell date -u '+%Y-%m-%d_%H:%M:%S')
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# 目录定义
BUILD_DIR := build
DIST_DIR := dist
SCRIPTS_DIR := scripts

# Go 构建参数
LDFLAGS := -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME) -X main.GitCommit=$(GIT_COMMIT) -s -w
GO_BUILD_FLAGS := -ldflags="$(LDFLAGS)" -trimpath

# 平台检测
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
    OS := linux
endif
ifeq ($(UNAME_S),Darwin)
    OS := darwin
endif

ifeq ($(UNAME_M),x86_64)
    ARCH := amd64
endif
ifeq ($(UNAME_M),arm64)
    ARCH := arm64
endif
ifeq ($(UNAME_M),aarch64)
    ARCH := arm64
endif

CURRENT_PLATFORM := $(OS)/$(ARCH)

# 颜色输出
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

# 默认目标
help: ## 显示帮助信息
	@echo "$(PROJECT_NAME) 构建系统"
	@echo ""
	@echo "可用命令:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "平台信息:"
	@echo "  当前平台: $(CURRENT_PLATFORM)"
	@echo "  版本: $(VERSION)"
	@echo "  构建时间: $(BUILD_TIME)"
	@echo "  Git提交: $(GIT_COMMIT)"

# 依赖检查
deps: ## 安装和检查依赖
	@echo "$(GREEN)[INFO]$(NC) 检查构建依赖..."
	@command -v go >/dev/null 2>&1 || { echo "$(RED)[ERROR]$(NC) Go未安装"; exit 1; }
	@echo "$(GREEN)[INFO]$(NC) Go版本: $$(go version | awk '{print $$3}')"
	@command -v python3 >/dev/null 2>&1 || echo "$(YELLOW)[WARN]$(NC) Python3未安装，将跳过Python服务"
	@command -v docker >/dev/null 2>&1 && echo "$(GREEN)[INFO]$(NC) Docker可用" || echo "$(YELLOW)[WARN]$(NC) Docker不可用"

# 清理
clean: ## 清理构建产物
	@echo "$(GREEN)[INFO]$(NC) 清理构建目录..."
	@rm -rf $(BUILD_DIR) $(DIST_DIR)
	@echo "$(GREEN)[INFO]$(NC) 清理完成"

# 代码检查
lint: ## 代码检查
	@echo "$(GREEN)[INFO]$(NC) 运行代码检查..."
	@cd backend && go vet ./...
	@cd backend && go fmt ./...
	@echo "$(GREEN)[INFO]$(NC) 代码检查完成"

# 代码格式化
format: ## 格式化代码
	@echo "$(GREEN)[INFO]$(NC) 格式化Go代码..."
	@cd backend && go fmt ./...
	@echo "$(GREEN)[INFO]$(NC) 格式化完成"

# 测试
test: ## 运行测试
	@echo "$(GREEN)[INFO]$(NC) 运行测试..."
	@cd backend && go test -v ./...
	@echo "$(GREEN)[INFO]$(NC) 测试完成"

# 构建当前平台
build: deps ## 构建当前平台的所有服务
	@$(SCRIPTS_DIR)/build.sh all $(CURRENT_PLATFORM)

# 构建后端服务
build-backend: deps ## 构建后端服务（当前平台）
	@$(SCRIPTS_DIR)/build.sh backend $(CURRENT_PLATFORM)

# 构建算法服务
build-algo: ## 构建算法服务（当前平台）
	@$(SCRIPTS_DIR)/build.sh algo $(CURRENT_PLATFORM)

# 构建管理后台
build-admin: ## 构建管理后台（当前平台）
	@$(SCRIPTS_DIR)/build.sh admin $(CURRENT_PLATFORM)

# 构建所有平台
build-all: deps ## 构建所有支持的平台
	@$(SCRIPTS_DIR)/build.sh --all

# 发布模式构建
release: deps ## 发布模式构建（优化编译）
	@$(SCRIPTS_DIR)/build.sh --all --release

# Linux构建
build-linux: deps ## 构建Linux版本
	@$(SCRIPTS_DIR)/build.sh all linux/amd64
	@$(SCRIPTS_DIR)/build.sh all linux/arm64

# macOS构建
build-macos: deps ## 构建macOS版本
	@$(SCRIPTS_DIR)/build.sh all darwin/amd64
	@$(SCRIPTS_DIR)/build.sh all darwin/arm64

# Docker构建
docker: ## 构建Docker镜像
	@echo "$(GREEN)[INFO]$(NC) 构建Docker镜像..."
	@docker build -t $(PROJECT_NAME)-backend:$(VERSION) -f backend/Dockerfile .
	@docker build -t $(PROJECT_NAME)-algo:$(VERSION) -f algo/Dockerfile .
	@echo "$(GREEN)[INFO]$(NC) Docker镜像构建完成"

# 本地开发环境
dev: ## 启动本地开发环境
	@echo "$(GREEN)[INFO]$(NC) 启动本地开发环境..."
	@./deploy.sh --chatbot

# 安装到系统
install: build ## 安装到系统（需要sudo权限）
	@echo "$(GREEN)[INFO]$(NC) 安装到系统..."
	@sudo mkdir -p /usr/local/bin
	@sudo cp $(BUILD_DIR)/$(OS)-$(ARCH)/backend-$(OS)-$(ARCH) /usr/local/bin/chatbot-backend
	@sudo chmod +x /usr/local/bin/chatbot-backend
	@echo "$(GREEN)[INFO]$(NC) 安装完成"

# 卸载
uninstall: ## 从系统卸载
	@echo "$(GREEN)[INFO]$(NC) 从系统卸载..."
	@sudo rm -f /usr/local/bin/chatbot-backend
	@echo "$(GREEN)[INFO]$(NC) 卸载完成"

# 检查构建结果
check: ## 检查构建结果
	@echo "$(GREEN)[INFO]$(NC) 检查构建结果..."
	@if [ -d "$(BUILD_DIR)" ]; then \
		echo "构建产物:"; \
		find $(BUILD_DIR) -type f -perm +111 | while read file; do \
			size=$$(ls -lh "$$file" | awk '{print $$5}'); \
			echo "  $$file ($$size)"; \
		done; \
	fi
	@if [ -d "$(DIST_DIR)" ]; then \
		echo "发布包:"; \
		find $(DIST_DIR) -name "*.tar.gz" | while read file; do \
			size=$$(ls -lh "$$file" | awk '{print $$5}'); \
			echo "  $$file ($$size)"; \
		done; \
	fi

# 快速构建（跳过测试）
quick: ## 快速构建（跳过依赖检查和测试）
	@$(SCRIPTS_DIR)/build.sh backend $(CURRENT_PLATFORM)

# 性能测试构建
bench: ## 构建性能测试版本
	@echo "$(GREEN)[INFO]$(NC) 构建性能测试版本..."
	@cd backend && CGO_ENABLED=0 GOOS=$(OS) GOARCH=$(ARCH) go build \
		-ldflags="$(LDFLAGS)" \
		-gcflags="-m -l" \
		-o ../$(BUILD_DIR)/$(OS)-$(ARCH)/backend-bench \
		./cmd/server
	@echo "$(GREEN)[INFO]$(NC) 性能测试版本构建完成"

# 调试版本构建
debug: ## 构建调试版本
	@echo "$(GREEN)[INFO]$(NC) 构建调试版本..."
	@cd backend && CGO_ENABLED=0 GOOS=$(OS) GOARCH=$(ARCH) go build \
		-ldflags="-X main.Version=$(VERSION)-debug -X main.BuildTime=$(BUILD_TIME) -X main.GitCommit=$(GIT_COMMIT)" \
		-gcflags="all=-N -l" \
		-o ../$(BUILD_DIR)/$(OS)-$(ARCH)/backend-debug \
		./cmd/server
	@echo "$(GREEN)[INFO]$(NC) 调试版本构建完成"

# 生成版本信息
version: ## 显示版本信息
	@echo "项目: $(PROJECT_NAME)"
	@echo "版本: $(VERSION)"
	@echo "构建时间: $(BUILD_TIME)"
	@echo "Git提交: $(GIT_COMMIT)"
	@echo "平台: $(CURRENT_PLATFORM)"

# 打包源码
package-source: ## 打包源码
	@echo "$(GREEN)[INFO]$(NC) 打包源码..."
	@mkdir -p $(DIST_DIR)
	@git archive --format=tar.gz --prefix=$(PROJECT_NAME)-$(VERSION)/ HEAD > $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)-source.tar.gz
	@echo "$(GREEN)[INFO]$(NC) 源码包: $(DIST_DIR)/$(PROJECT_NAME)-$(VERSION)-source.tar.gz"

# 完整构建流程
all: clean deps lint test build-all check ## 完整构建流程（清理、依赖、检查、测试、构建、验证）

# CI/CD构建
ci: clean deps lint test build ## CI/CD构建流程

# 本地测试
local-test: build ## 本地测试构建结果
	@echo "$(GREEN)[INFO]$(NC) 测试构建结果..."
	@if [ -f "$(BUILD_DIR)/$(OS)-$(ARCH)/backend-$(OS)-$(ARCH)" ]; then \
		echo "测试后端服务..."; \
		$(BUILD_DIR)/$(OS)-$(ARCH)/backend-$(OS)-$(ARCH) --version || true; \
	fi

# 显示构建信息
info: ## 显示构建信息
	@echo "构建配置:"
	@echo "  项目名称: $(PROJECT_NAME)"
	@echo "  版本: $(VERSION)"
	@echo "  构建时间: $(BUILD_TIME)"
	@echo "  Git提交: $(GIT_COMMIT)"
	@echo "  当前平台: $(CURRENT_PLATFORM)"
	@echo "  构建目录: $(BUILD_DIR)"
	@echo "  发布目录: $(DIST_DIR)"
	@echo "  Go构建参数: $(GO_BUILD_FLAGS)"