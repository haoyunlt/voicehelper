import Link from 'next/link'
import { MessageCircle, Database, BarChart3 } from 'lucide-react'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            智能聊天机器人
          </h1>
          <p className="text-xl text-gray-600 mb-8">
            基于豆包大模型的企业级知识问答系统
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-4xl mx-auto">
          <Link href="/chat" className="group">
            <div className="bg-white rounded-lg shadow-lg p-8 hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-center w-16 h-16 bg-blue-100 rounded-lg mb-6 mx-auto group-hover:bg-blue-200 transition-colors">
                <MessageCircle className="w-8 h-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3 text-center">
                开始对话
              </h3>
              <p className="text-gray-600 text-center">
                与智能助手对话，获取准确的知识问答
              </p>
            </div>
          </Link>

          <Link href="/datasets" className="group">
            <div className="bg-white rounded-lg shadow-lg p-8 hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-center w-16 h-16 bg-green-100 rounded-lg mb-6 mx-auto group-hover:bg-green-200 transition-colors">
                <Database className="w-8 h-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3 text-center">
                知识库管理
              </h3>
              <p className="text-gray-600 text-center">
                上传和管理文档，构建专属知识库
              </p>
            </div>
          </Link>

          <Link href="/analytics" className="group">
            <div className="bg-white rounded-lg shadow-lg p-8 hover:shadow-xl transition-shadow">
              <div className="flex items-center justify-center w-16 h-16 bg-purple-100 rounded-lg mb-6 mx-auto group-hover:bg-purple-200 transition-colors">
                <BarChart3 className="w-8 h-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3 text-center">
                数据分析
              </h3>
              <p className="text-gray-600 text-center">
                查看使用统计和性能指标
              </p>
            </div>
          </Link>
        </div>

        <div className="mt-16 text-center">
          <div className="bg-white rounded-lg shadow-lg p-8 max-w-2xl mx-auto">
            <h2 className="text-2xl font-semibold text-gray-900 mb-4">
              功能特性
            </h2>
            <div className="grid md:grid-cols-2 gap-6 text-left">
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">🚀 高性能</h3>
                <p className="text-gray-600">基于 Milvus 向量数据库，毫秒级检索响应</p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">🎯 精准问答</h3>
                <p className="text-gray-600">豆包大模型驱动，提供准确的知识问答</p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">📚 多格式支持</h3>
                <p className="text-gray-600">支持 PDF、Word、Markdown 等多种文档格式</p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">🔒 企业级安全</h3>
                <p className="text-gray-600">租户隔离、权限控制、审计日志</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
