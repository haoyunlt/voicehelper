# VoiceHelper 项目 Makefile
# 提供常用的开发和测试命令

.PHONY: help install test test-unit test-integration test-e2e test-performance test-quick clean lint format docs

# 默认目标
help:
	@echo "VoiceHelper 项目命令"
	@echo "===================="
	@echo ""
	@echo "安装和设置:"
	@echo "  make install          安装所有依赖"
	@echo "  make install-test     安装测试依赖"
	@echo ""
	@echo "测试命令:"
	@echo "  make test             运行所有测试"
	@echo "  make test-unit        运行单元测试"
	@echo "  make test-integration 运行集成测试"
	@echo "  make test-e2e         运行端到端测试"
	@echo "  make test-performance 运行性能测试"
	@echo "  make test-quick       快速测试验证"
	@echo "  make test-demo        测试演示"
	@echo ""
	@echo "代码质量:"
	@echo "  make lint             代码检查"
	@echo "  make format           代码格式化"
	@echo "  make coverage         生成覆盖率报告"
	@echo ""
	@echo "文档和清理:"
	@echo "  make docs             生成文档"
	@echo "  make clean            清理临时文件"
	@echo ""
	@echo "服务管理:"
	@echo "  make start-backend    启动后端服务"
	@echo "  make start-algo       启动算法服务"
	@echo "  make start-frontend   启动前端服务"
	@echo "  make stop-services    停止所有服务"

# 安装依赖
install:
	@echo "📦 安装项目依赖..."
	pip install -r requirements.txt
	cd platforms/web && npm install
	cd backend && go mod tidy

install-test:
	@echo "📦 安装测试依赖..."
	pip install -r requirements-test.txt

# 测试命令
test: install-test
	@echo "🧪 运行所有测试..."
	./tools/scripts/run_tests.sh

test-unit: install-test
	@echo "🧪 运行单元测试..."
	python -m pytest tools/testing/unit/ -v

test-integration: install-test
	@echo "🔗 运行集成测试..."
	python -m pytest tools/testing/integration/ -v

test-e2e: install-test
	@echo "🌐 运行端到端测试..."
	python -m pytest tools/testing/e2e/ -v

test-performance:
	@echo "⚡ 启动性能测试..."
	@echo "选择测试类型:"
	@echo "1. 负载测试: locust -f scripts/performance/load_test.py --web-host 0.0.0.0"
	@echo "2. 压力测试: locust -f scripts/performance/stress_test.py --web-host 0.0.0.0"
	@echo "3. 基准测试: locust -f scripts/performance/benchmark_test.py --web-host 0.0.0.0"
	@echo "然后访问 http://localhost:8089"

test-quick:
	@echo "⚡ 快速测试验证..."
	./tools/scripts/simple_validation.py

test-demo:
	@echo "🎯 测试演示..."
	python tools/scripts/simple_test_logging.py

# 代码质量
lint:
	@echo "🔍 代码检查..."
	# Python代码检查
	flake8 algo/ tests/ --max-line-length=120 || true
	pylint algo/ tests/ --disable=all --enable=E,W || true
	# Markdown文档检查
	markdownlint docs/ || true

format:
	@echo "✨ 代码格式化..."
	# Python代码格式化
	black algo/ tests/ --line-length=120 || true
	isort algo/ tests/ || true
	# Go代码格式化
	cd backend && go fmt ./... || true

coverage: install-test
	@echo "📊 生成覆盖率报告..."
	python -m pytest tests/unit/ --cov=algo --cov=backend --cov-report=html:reports/coverage --cov-report=term
	@echo "📁 覆盖率报告: reports/coverage/index.html"

# 服务管理
start-backend:
	@echo "🚀 启动后端服务..."
	cd backend && go run cmd/gateway/main.go &

start-algo:
	@echo "🚀 启动算法服务..."
	cd algo && python app/main.py &

start-frontend:
	@echo "🚀 启动前端服务..."
	cd platforms/web && npm run dev &

stop-services:
	@echo "🛑 停止所有服务..."
	pkill -f "go run cmd/gateway/main.go" || true
	pkill -f "python app/main.py" || true
	pkill -f "npm run dev" || true

# 文档
docs:
	@echo "📚 生成文档..."
	# 生成API文档
	cd backend && swag init -g cmd/gateway/main.go || true
	# 生成测试报告
	python -m pytest tools/testing/ --html=reports/test_report.html --self-contained-html || true
	@echo "📁 文档位置:"
	@echo "  API文档: backend/docs/"
	@echo "  测试报告: reports/test_report.html"

# 清理
clean:
	@echo "🧹 清理临时文件..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + || true
	rm -rf reports/ || true
	rm -rf .coverage || true
	rm -rf htmlcov/ || true
	rm -rf temp_* || true

# 开发环境设置
dev-setup: install install-test
	@echo "🛠️ 开发环境设置完成"
	@echo ""
	@echo "下一步:"
	@echo "1. 启动服务: make start-backend start-algo start-frontend"
	@echo "2. 运行测试: make test-quick"
	@echo "3. 查看文档: docs/TESTING_GUIDE.md"

# 持续集成
ci: install-test lint test coverage
	@echo "✅ CI流程完成"