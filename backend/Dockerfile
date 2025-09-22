# 多阶段构建
FROM golang:1.21-alpine AS builder

WORKDIR /app

# 安装依赖
RUN apk add --no-cache git

# 复制 go mod 文件
COPY go.mod go.sum ./
RUN go mod download

# 复制源代码
COPY . .

# 构建应用
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o main ./cmd/server

# 运行阶段
FROM alpine:latest

RUN apk --no-cache add ca-certificates tzdata
WORKDIR /root/

# 复制二进制文件
COPY --from=builder /app/main .

# 设置时区
ENV TZ=Asia/Shanghai

# 环境变量通过 .env 文件或 docker-compose 环境变量传入
# 包括但不限于：
# PORT, GIN_MODE, DATABASE_URL, REDIS_URL, ALGO_SERVICE_URL
# JWT_SECRET, ADMIN_SECRET_KEY, LOG_LEVEL 等

EXPOSE 8080

CMD ["./main"]
