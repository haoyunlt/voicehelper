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

# 设置环境变量
ENV PORT=8080
ENV GIN_MODE=release
ENV DATABASE_URL=postgres://chatbot:chatbot123@postgres:5432/chatbot?sslmode=disable
ENV REDIS_URL=redis://redis:6379
ENV ALGO_SERVICE_URL=http://algo-service:8000
ENV JWT_SECRET=11e8ba7f6690ebbe069afd0bf43cfcc3ad5bacefb1fe816c04f90bb022749995
ENV ADMIN_SECRET_KEY=b29d40c1ea82b3331a76b7479526e67fcbf7e5d09719ddf8e6c607dae81756fa
ENV LOG_LEVEL=info

EXPOSE 8080

CMD ["./main"]
