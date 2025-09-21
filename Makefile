.PHONY: help build up down logs clean dev

help: ## 显示帮助信息
	@echo "可用命令："
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## 构建所有服务镜像
	docker-compose build

up: ## 启动所有服务
	docker-compose up -d

down: ## 停止所有服务
	docker-compose down

logs: ## 查看所有服务日志
	docker-compose logs -f

clean: ## 清理所有容器和卷
	docker-compose down -v
	docker system prune -f

dev: ## 开发模式启动（仅基础设施）
	docker-compose up -d milvus redis postgres etcd minio

status: ## 查看服务状态
	docker-compose ps

restart: ## 重启所有服务
	docker-compose restart

backend-logs: ## 查看后端日志
	docker-compose logs -f backend

frontend-logs: ## 查看前端日志
	docker-compose logs -f frontend

algo-logs: ## 查看算法服务日志
	docker-compose logs -f algo-service
