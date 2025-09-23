/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  typescript: {
    ignoreBuildErrors: false, // 启用TypeScript类型检查
  },
  eslint: {
    ignoreDuringBuilds: false, // 启用ESLint检查
  },
  experimental: {
    typedRoutes: true, // 启用类型化路由
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:8080/api/:path*', // 代理到后端服务
      },
    ]
  },
}

module.exports = nextConfig
