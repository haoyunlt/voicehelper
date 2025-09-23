'use client';

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { 
  CodeBracketIcon, 
  CpuChipIcon, 
  SpeakerWaveIcon,
  EyeIcon,
  PuzzlePieceIcon,
  RocketLaunchIcon,
  DocumentTextIcon,
  PlayIcon
} from '@heroicons/react/24/outline';

const features = [
  {
    name: '多模态交互',
    description: '支持文本、语音、图像、视频等5种模态的统一处理和融合',
    icon: PuzzlePieceIcon,
    stats: '5种模态',
    color: 'from-blue-500 to-cyan-500'
  },
  {
    name: '超低延迟语音',
    description: 'v1.8.0优化：语音处理延迟降至150ms，支持情感化TTS',
    icon: SpeakerWaveIcon,
    stats: '150ms延迟',
    color: 'from-purple-500 to-pink-500'
  },
  {
    name: '智能视觉理解',
    description: '支持12种图像类型识别，95%准确率，包含OCR和品牌识别',
    icon: EyeIcon,
    stats: '95%准确率',
    color: 'from-green-500 to-emerald-500'
  },
  {
    name: '500+服务集成',
    description: 'v1.9.0生态建设：集成办公套件、开发工具、社交平台等',
    icon: CpuChipIcon,
    stats: '500+服务',
    color: 'from-orange-500 to-red-500'
  }
];

const quickStart = [
  {
    title: 'JavaScript SDK',
    description: '快速集成到Web应用',
    code: `import { VoiceHelperSDK } from 'voicehelper-sdk';

const sdk = new VoiceHelperSDK({
  apiKey: 'your_api_key'
});

const response = await sdk.createChatCompletion([
  { role: 'user', content: '你好' }
]);`,
    language: 'javascript'
  },
  {
    title: 'Python SDK',
    description: '适用于AI/ML项目',
    code: `from voicehelper_sdk import VoiceHelperSDK

sdk = VoiceHelperSDK(api_key="your_api_key")

response = await sdk.create_chat_completion([
    {"role": "user", "content": "你好"}
])`,
    language: 'python'
  },
  {
    title: 'REST API',
    description: '直接HTTP调用',
    code: `curl -X POST https://api.voicehelper.com/v1/chat/completions \\
  -H "X-API-Key: your_api_key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "model": "gpt-4-turbo"
  }'`,
    language: 'bash'
  }
];

const stats = [
  { name: '开发者', value: '10,000+', description: '活跃开发者社区' },
  { name: 'API调用', value: '100M+', value_suffix: '/月', description: '月度API调用量' },
  { name: '响应时间', value: '150ms', description: '平均语音处理延迟' },
  { name: '可用性', value: '99.9%', description: '服务可用性保证' }
];

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-800/20 to-cyan-800/20 backdrop-blur-3xl"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center"
          >
            <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
              <span className="bg-gradient-to-r from-blue-400 via-purple-400 to-cyan-400 bg-clip-text text-transparent">
                VoiceHelper
              </span>
              <br />
              <span className="text-3xl md:text-5xl text-gray-300">开发者平台</span>
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-4xl mx-auto">
              v1.9.0 生态建设版 - 构建下一代多模态AI应用
              <br />
              <span className="text-lg text-gray-400">
                150ms语音延迟 • 95%视觉准确率 • 500+服务集成 • 5种模态融合
              </span>
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href={"/docs/quickstart" as any}
                className="inline-flex items-center px-8 py-4 text-lg font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                <RocketLaunchIcon className="w-5 h-5 mr-2" />
                快速开始
              </Link>
              <Link
                href={"/docs/api" as any}
                className="inline-flex items-center px-8 py-4 text-lg font-medium text-gray-300 bg-white/10 backdrop-blur-sm rounded-lg hover:bg-white/20 transition-all duration-200 border border-white/20"
              >
                <DocumentTextIcon className="w-5 h-5 mr-2" />
                API文档
              </Link>
              <Link
                href={"/playground" as any}
                className="inline-flex items-center px-8 py-4 text-lg font-medium text-gray-300 bg-white/10 backdrop-blur-sm rounded-lg hover:bg-white/20 transition-all duration-200 border border-white/20"
              >
                <PlayIcon className="w-5 h-5 mr-2" />
                在线体验
              </Link>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="relative py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-3xl md:text-4xl font-bold text-white mb-2">
                  {stat.value}
                  <span className="text-xl text-gray-400">{stat.value_suffix}</span>
                </div>
                <div className="text-sm text-gray-400 mb-1">{stat.name}</div>
                <div className="text-xs text-gray-500">{stat.description}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="relative py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              强大的AI能力
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              基于v1.8.0体验升级版的技术突破，提供业界领先的多模态AI能力
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className="relative group"
              >
                <div className="absolute inset-0 bg-gradient-to-r opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl blur-xl"
                     style={{
                       background: `linear-gradient(135deg, ${feature.color.split(' ')[1]}, ${feature.color.split(' ')[3]})`
                     }}
                ></div>
                <div className="relative bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20 hover:border-white/40 transition-all duration-300">
                  <div className={`inline-flex p-3 rounded-lg bg-gradient-to-r ${feature.color} mb-4`}>
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-semibold text-white mb-2">{feature.name}</h3>
                  <p className="text-gray-300 text-sm mb-4">{feature.description}</p>
                  <div className="text-2xl font-bold text-white">{feature.stats}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* Quick Start Section */}
      <div className="relative py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              5分钟快速集成
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              选择您喜欢的开发语言，几行代码即可开始使用VoiceHelper的强大功能
            </p>
          </motion.div>

          <div className="grid lg:grid-cols-3 gap-8">
            {quickStart.map((example, index) => (
              <motion.div
                key={example.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                className="bg-slate-800/50 backdrop-blur-sm rounded-2xl p-6 border border-slate-700"
              >
                <div className="flex items-center mb-4">
                  <CodeBracketIcon className="w-6 h-6 text-blue-400 mr-3" />
                  <h3 className="text-xl font-semibold text-white">{example.title}</h3>
                </div>
                <p className="text-gray-300 text-sm mb-4">{example.description}</p>
                <div className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
                  <pre className="text-sm text-gray-300">
                    <code>{example.code}</code>
                  </pre>
                </div>
                <div className="mt-4">
                  <Link
                    href={`/docs/sdk/${example.language === 'bash' ? 'rest-api' : example.language}` as any}
                    className="text-blue-400 hover:text-blue-300 text-sm font-medium"
                  >
                    查看完整文档 →
                  </Link>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="relative py-20">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              准备开始构建了吗？
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              加入10,000+开发者社区，使用VoiceHelper构建下一代AI应用
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link
                href={"/auth/signup" as any}
                className="inline-flex items-center px-8 py-4 text-lg font-medium text-white bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg hover:from-blue-700 hover:to-purple-700 transition-all duration-200 shadow-lg hover:shadow-xl"
              >
                免费注册
              </Link>
              <Link
                href={"/docs" as any}
                className="inline-flex items-center px-8 py-4 text-lg font-medium text-gray-300 bg-white/10 backdrop-blur-sm rounded-lg hover:bg-white/20 transition-all duration-200 border border-white/20"
              >
                浏览文档
              </Link>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
