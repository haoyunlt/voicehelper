FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# 环境变量通过 .env 文件或 docker-compose 环境变量传入
# 包括但不限于：
# NODE_ENV, PORT, NEXT_PUBLIC_API_URL 等

# Expose port
EXPOSE 3002

# Start the application in development mode
CMD ["npm", "run", "dev"]
