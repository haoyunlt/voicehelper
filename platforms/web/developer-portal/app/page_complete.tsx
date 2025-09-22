'use client';

import React, { useState, useEffect } from 'react';
import { 
  Code, 
  Book, 
  Zap, 
  Shield, 
  Globe, 
  Smartphone, 
  Monitor, 
  Headphones,
  Brain,
  MessageSquare,
  FileText,
  Settings,
  Play,
  Copy,
  Check,
  ExternalLink,
  Github,
  Download,
  Star,
  Users,
  TrendingUp,
  Clock,
  CheckCircle
} from 'lucide-react';

// 类型定义
interface CodeExample {
  language: string;
  title: string;
  code: string;
}

interface Feature {
  icon: React.ReactNode;
  title: string;
  description: string;
  status: 'available' | 'beta' | 'coming_soon';
}

interface SDK {
  name: string;
  language: string;
  version: string;
  downloads: string;
  stars: number;
  description: string;
  installCommand: string;
  quickStart: string;
}

interface Service {
  name: string;
  category: string;
  description: string;
  status: 'active' | 'beta' | 'deprecated';
  popularity: number;
}

// 主页面组件
export default function DeveloperPortal() {
  const [activeTab, setActiveTab] = useState('overview');
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  // 复制代码功能
  const copyCode = async (code: string, id: string) => {
    await navigator.clipboard.writeText(code);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  // 代码示例
  const codeExamples: CodeExample[] = [
    {
      language: 'javascript',
      title: 'JavaScript/TypeScript',
      code: `import { VoiceHelperSDK } from '@voicehelper/sdk';

const client = new VoiceHelperSDK({
  apiKey: 'your-api-key'
});

// 简单对话
const response = await client.createChatCompletion({
  messages: [{ role: 'user', content: '你好！' }],
  model: 'gpt-4'
});

console.log(response.choices[0].message.content);

// 语音转文字
const transcription = await client.transcribeAudio({
  file: audioFile,
  model: 'whisper-1'
});

// 文字转语音
const audioBuffer = await client.synthesizeText({
  text: '欢迎使用VoiceHelper AI',
  voice: 'zh-female-1',
  emotion: 'happy'
});`
    },
    {
      language: 'python',
      title: 'Python',
      code: `from voicehelper_sdk import VoiceHelperSDK, VoiceHelperConfig

# 创建客户端
config = VoiceHelperConfig(api_key="your-api-key")
client = VoiceHelperSDK(config)

# 简单对话
response = await client.create_chat_completion({
    "messages": [{"role": "user", "content": "你好！"}],
    "model": "gpt-4"
})

print(response.choices[0]["message"]["content"])

# 语音转文字
with open("audio.wav", "rb") as audio_file:
    transcription = await client.transcribe_audio(audio_file)
    print(transcription.text)

# 文字转语音
audio_data = await client.synthesize_text({
    "text": "欢迎使用VoiceHelper AI",
    "voice": "zh-female-1",
    "emotion": "happy"
})`
    },
    {
      language: 'curl',
      title: 'REST API',
      code: `# 对话接口
curl -X POST "https://api.voicehelper.ai/v1/chat/completions" \\
  -H "Authorization: Bearer your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "你好！"}],
    "model": "gpt-4"
  }'

# 语音转文字
curl -X POST "https://api.voicehelper.ai/v1/voice/transcribe" \\
  -H "Authorization: Bearer your-api-key" \\
  -F "file=@audio.wav" \\
  -F "model=whisper-1"

# 文字转语音
curl -X POST "https://api.voicehelper.ai/v1/voice/synthesize" \\
  -H "Authorization: Bearer your-api-key" \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "欢迎使用VoiceHelper AI",
    "voice": "zh-female-1",
    "emotion": "happy"
  }' \\
  --output speech.mp3`
    }
  ];

  // 核心功能
  const features: Feature[] = [
    {
      icon: <MessageSquare className="w-6 h-6" />,
      title: "智能对话",
      description: "支持多模型的智能对话，包括GPT-4、Claude-3、文心一言等",
      status: "available"
    },
    {
      icon: <Headphones className="w-6 h-6" />,
      title: "语音处理",
      description: "150ms超低延迟的语音识别和合成，支持多语言和情感表达",
      status: "available"
    },
    {
      icon: <Brain className="w-6 h-6" />,
      title: "知识检索",
      description: "GraphRAG知识图谱检索，支持多跳推理和复杂查询",
      status: "available"
    },
    {
      icon: <Globe className="w-6 h-6" />,
      title: "服务集成",
      description: "500+第三方服务集成，覆盖办公、开发、社交、电商等15个分类",
      status: "available"
    },
    {
      icon: <Smartphone className="w-6 h-6" />,
      title: "移动端SDK",
      description: "iOS和Android原生SDK，支持离线缓存和推送通知",
      status: "beta"
    },
    {
      icon: <Monitor className="w-6 h-6" />,
      title: "桌面应用",
      description: "跨平台桌面应用，支持系统托盘和全局快捷键",
      status: "beta"
    }
  ];

  // SDK列表
  const sdks: SDK[] = [
    {
      name: "JavaScript SDK",
      language: "TypeScript/JavaScript",
      version: "1.9.0",
      downloads: "50K+",
      stars: 1200,
      description: "完整的TypeScript SDK，支持浏览器和Node.js环境",
      installCommand: "npm install @voicehelper/sdk",
      quickStart: "import { VoiceHelperSDK } from '@voicehelper/sdk';"
    },
    {
      name: "Python SDK", 
      language: "Python",
      version: "1.9.0",
      downloads: "30K+",
      stars: 800,
      description: "异步和同步双版本Python SDK，支持Python 3.8+",
      installCommand: "pip install voicehelper-sdk",
      quickStart: "from voicehelper_sdk import VoiceHelperSDK"
    },
    {
      name: "Go SDK",
      language: "Go",
      version: "1.8.0",
      downloads: "15K+", 
      stars: 450,
      description: "高性能Go SDK，适合后端服务集成",
      installCommand: "go get github.com/voicehelper/go-sdk",
      quickStart: "import \"github.com/voicehelper/go-sdk\""
    },
    {
      name: "Java SDK",
      language: "Java",
      version: "1.8.0",
      downloads: "20K+",
      stars: 600,
      description: "企业级Java SDK，支持Spring Boot集成",
      installCommand: "implementation 'ai.voicehelper:java-sdk:1.8.0'",
      quickStart: "import ai.voicehelper.VoiceHelperClient;"
    }
  ];

  // 热门服务
  const popularServices: Service[] = [
    { name: "微信", category: "社交平台", description: "微信消息发送和接收", status: "active", popularity: 95 },
    { name: "钉钉", category: "办公套件", description: "钉钉工作通知和审批", status: "active", popularity: 92 },
    { name: "GitHub", category: "开发工具", description: "代码仓库管理和CI/CD", status: "active", popularity: 90 },
    { name: "淘宝", category: "电商平台", description: "商品搜索和订单管理", status: "active", popularity: 88 },
    { name: "阿里云", category: "云服务", description: "云资源管理和监控", status: "active", popularity: 85 },
    { name: "OpenAI", category: "AI/ML", description: "GPT模型调用和管理", status: "active", popularity: 98 }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* 导航栏 */}
      <nav className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0 flex items-center">
                <Zap className="w-8 h-8 text-blue-600" />
                <span className="ml-2 text-xl font-bold text-gray-900">VoiceHelper</span>
                <span className="ml-2 px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full">开发者平台</span>
              </div>
              
              <div className="hidden md:ml-10 md:flex md:space-x-8">
                <button
                  onClick={() => setActiveTab('overview')}
                  className={`px-3 py-2 text-sm font-medium ${
                    activeTab === 'overview' 
                      ? 'text-blue-600 border-b-2 border-blue-600' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  概览
                </button>
                <button
                  onClick={() => setActiveTab('docs')}
                  className={`px-3 py-2 text-sm font-medium ${
                    activeTab === 'docs' 
                      ? 'text-blue-600 border-b-2 border-blue-600' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  API文档
                </button>
                <button
                  onClick={() => setActiveTab('sdks')}
                  className={`px-3 py-2 text-sm font-medium ${
                    activeTab === 'sdks' 
                      ? 'text-blue-600 border-b-2 border-blue-600' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  SDK下载
                </button>
                <button
                  onClick={() => setActiveTab('services')}
                  className={`px-3 py-2 text-sm font-medium ${
                    activeTab === 'services' 
                      ? 'text-blue-600 border-b-2 border-blue-600' 
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  服务集成
                </button>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <a href="https://github.com/voicehelper" className="text-gray-500 hover:text-gray-700">
                <Github className="w-5 h-5" />
              </a>
              <button className="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
                获取API Key
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* 主要内容 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* 概览页面 */}
        {activeTab === 'overview' && (
          <div className="space-y-12">
            {/* Hero区域 */}
            <div className="text-center">
              <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl md:text-6xl">
                构建下一代
                <span className="text-blue-600"> AI 应用</span>
              </h1>
              <p className="mt-3 max-w-md mx-auto text-base text-gray-500 sm:text-lg md:mt-5 md:text-xl md:max-w-3xl">
                VoiceHelper AI 提供企业级智能对话、语音处理、知识检索和500+服务集成能力，
                帮助开发者快速构建智能应用。
              </p>
              
              <div className="mt-8 flex justify-center space-x-4">
                <button className="bg-blue-600 text-white px-8 py-3 rounded-lg text-lg font-medium hover:bg-blue-700 flex items-center">
                  <Play className="w-5 h-5 mr-2" />
                  快速开始
                </button>
                <button className="border border-gray-300 text-gray-700 px-8 py-3 rounded-lg text-lg font-medium hover:bg-gray-50 flex items-center">
                  <Book className="w-5 h-5 mr-2" />
                  查看文档
                </button>
              </div>
            </div>

            {/* 统计数据 */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">500+</div>
                <div className="text-sm text-gray-500">服务集成</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">150ms</div>
                <div className="text-sm text-gray-500">语音延迟</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">99.9%</div>
                <div className="text-sm text-gray-500">服务可用性</div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">10K+</div>
                <div className="text-sm text-gray-500">开发者</div>
              </div>
            </div>

            {/* 快速开始代码示例 */}
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">快速开始</h2>
              
              <div className="space-y-6">
                {codeExamples.map((example, index) => (
                  <div key={index} className="border rounded-lg overflow-hidden">
                    <div className="bg-gray-50 px-4 py-2 border-b flex justify-between items-center">
                      <span className="font-medium text-gray-700">{example.title}</span>
                      <button
                        onClick={() => copyCode(example.code, `code-${index}`)}
                        className="flex items-center text-sm text-gray-500 hover:text-gray-700"
                      >
                        {copiedCode === `code-${index}` ? (
                          <>
                            <Check className="w-4 h-4 mr-1" />
                            已复制
                          </>
                        ) : (
                          <>
                            <Copy className="w-4 h-4 mr-1" />
                            复制
                          </>
                        )}
                      </button>
                    </div>
                    <pre className="p-4 text-sm bg-gray-900 text-gray-100 overflow-x-auto">
                      <code>{example.code}</code>
                    </pre>
                  </div>
                ))}
              </div>
            </div>

            {/* 核心功能 */}
            <div className="bg-white rounded-xl shadow-lg p-8">
              <h2 className="text-2xl font-bold text-gray-900 mb-6">核心功能</h2>
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {features.map((feature, index) => (
                  <div key={index} className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                    <div className="flex items-start">
                      <div className="flex-shrink-0">
                        <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center text-blue-600">
                          {feature.icon}
                        </div>
                      </div>
                      <div className="ml-4">
                        <div className="flex items-center">
                          <h3 className="text-lg font-medium text-gray-900">{feature.title}</h3>
                          <span className={`ml-2 px-2 py-1 text-xs rounded-full ${
                            feature.status === 'available' 
                              ? 'bg-green-100 text-green-800' 
                              : feature.status === 'beta'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-gray-100 text-gray-800'
                          }`}>
                            {feature.status === 'available' ? '可用' : feature.status === 'beta' ? '测试版' : '即将推出'}
                          </span>
                        </div>
                        <p className="mt-2 text-sm text-gray-500">{feature.description}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* API文档页面 */}
        {activeTab === 'docs' && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">API 文档</h2>
              <a 
                href="/docs/api/openapi_v3_complete.yaml" 
                className="flex items-center text-blue-600 hover:text-blue-700"
              >
                <ExternalLink className="w-4 h-4 mr-1" />
                OpenAPI 规范
              </a>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              <div className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                <MessageSquare className="w-8 h-8 text-blue-600 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">对话接口</h3>
                <p className="text-sm text-gray-500 mb-4">
                  创建智能对话，支持流式响应、工具调用和多模态输入
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">POST /chat/completions</span>
                    <span className="text-green-600">创建对话</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">GET /chat/conversations</span>
                    <span className="text-blue-600">获取对话列表</span>
                  </div>
                </div>
              </div>

              <div className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                <Headphones className="w-8 h-8 text-purple-600 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">语音接口</h3>
                <p className="text-sm text-gray-500 mb-4">
                  语音识别、合成和实时对话，支持多语言和情感表达
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">POST /voice/transcribe</span>
                    <span className="text-green-600">语音转文字</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">POST /voice/synthesize</span>
                    <span className="text-green-600">文字转语音</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">WS /voice/realtime</span>
                    <span className="text-purple-600">实时对话</span>
                  </div>
                </div>
              </div>

              <div className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                <Brain className="w-8 h-8 text-green-600 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">知识库接口</h3>
                <p className="text-sm text-gray-500 mb-4">
                  文档管理、知识检索和GraphRAG查询
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">POST /knowledge/datasets</span>
                    <span className="text-green-600">创建数据集</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">POST /knowledge/search</span>
                    <span className="text-blue-600">知识检索</span>
                  </div>
                </div>
              </div>

              <div className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                <Globe className="w-8 h-8 text-orange-600 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">服务集成</h3>
                <p className="text-sm text-gray-500 mb-4">
                  500+第三方服务集成和操作执行
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">GET /integrations/services</span>
                    <span className="text-blue-600">可用服务</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">POST /integrations/connections</span>
                    <span className="text-green-600">创建连接</span>
                  </div>
                </div>
              </div>

              <div className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                <Users className="w-8 h-8 text-indigo-600 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">用户管理</h3>
                <p className="text-sm text-gray-500 mb-4">
                  用户资料管理和使用统计查询
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">GET /users/profile</span>
                    <span className="text-blue-600">用户资料</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">GET /users/usage</span>
                    <span className="text-blue-600">使用统计</span>
                  </div>
                </div>
              </div>

              <div className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                <Shield className="w-8 h-8 text-red-600 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">认证授权</h3>
                <p className="text-sm text-gray-500 mb-4">
                  API Key、OAuth 2.0和JWT认证方式
                </p>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">API Key</span>
                    <span className="text-green-600">推荐</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">OAuth 2.0</span>
                    <span className="text-blue-600">企业级</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* SDK下载页面 */}
        {activeTab === 'sdks' && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">SDK 下载</h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {sdks.map((sdk, index) => (
                <div key={index} className="border rounded-lg p-6 hover:shadow-md transition-shadow">
                  <div className="flex items-start justify-between mb-4">
                    <div>
                      <h3 className="text-lg font-medium text-gray-900">{sdk.name}</h3>
                      <p className="text-sm text-gray-500">{sdk.language}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-500">v{sdk.version}</div>
                      <div className="flex items-center text-sm text-gray-500">
                        <Download className="w-4 h-4 mr-1" />
                        {sdk.downloads}
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-600 mb-4">{sdk.description}</p>
                  
                  <div className="space-y-3">
                    <div>
                      <label className="text-sm font-medium text-gray-700">安装命令</label>
                      <div className="mt-1 bg-gray-100 rounded p-2 text-sm font-mono flex justify-between items-center">
                        <span>{sdk.installCommand}</span>
                        <button
                          onClick={() => copyCode(sdk.installCommand, `install-${index}`)}
                          className="text-gray-500 hover:text-gray-700"
                        >
                          {copiedCode === `install-${index}` ? (
                            <Check className="w-4 h-4" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>
                    
                    <div>
                      <label className="text-sm font-medium text-gray-700">快速开始</label>
                      <div className="mt-1 bg-gray-100 rounded p-2 text-sm font-mono flex justify-between items-center">
                        <span>{sdk.quickStart}</span>
                        <button
                          onClick={() => copyCode(sdk.quickStart, `quick-${index}`)}
                          className="text-gray-500 hover:text-gray-700"
                        >
                          {copiedCode === `quick-${index}` ? (
                            <Check className="w-4 h-4" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                  
                  <div className="mt-4 flex items-center justify-between">
                    <div className="flex items-center text-sm text-gray-500">
                      <Star className="w-4 h-4 mr-1 text-yellow-400" />
                      {sdk.stars}
                    </div>
                    <div className="flex space-x-2">
                      <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                        文档
                      </button>
                      <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                        示例
                      </button>
                      <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                        GitHub
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 服务集成页面 */}
        {activeTab === 'services' && (
          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">服务集成</h2>
              <div className="text-sm text-gray-500">
                已集成 <span className="font-medium text-blue-600">500+</span> 个服务
              </div>
            </div>
            
            {/* 热门服务 */}
            <div className="mb-8">
              <h3 className="text-lg font-medium text-gray-900 mb-4">热门服务</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {popularServices.map((service, index) => (
                  <div key={index} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between mb-2">
                      <div>
                        <h4 className="font-medium text-gray-900">{service.name}</h4>
                        <p className="text-sm text-gray-500">{service.category}</p>
                      </div>
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        service.status === 'active' 
                          ? 'bg-green-100 text-green-800' 
                          : service.status === 'beta'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {service.status === 'active' ? '可用' : service.status === 'beta' ? '测试版' : '已弃用'}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600 mb-3">{service.description}</p>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center text-sm text-gray-500">
                        <TrendingUp className="w-4 h-4 mr-1" />
                        热度 {service.popularity}%
                      </div>
                      <button className="text-blue-600 hover:text-blue-700 text-sm font-medium">
                        查看详情
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 服务分类 */}
            <div>
              <h3 className="text-lg font-medium text-gray-900 mb-4">服务分类</h3>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
                {[
                  { name: '办公套件', count: 100, icon: '📊' },
                  { name: '开发工具', count: 120, icon: '⚙️' },
                  { name: '社交平台', count: 80, icon: '💬' },
                  { name: '电商平台', count: 60, icon: '🛒' },
                  { name: '云服务', count: 100, icon: '☁️' },
                  { name: 'AI/ML服务', count: 40, icon: '🤖' },
                  { name: '数据库', count: 30, icon: '🗄️' },
                  { name: '消息队列', count: 25, icon: '📨' },
                  { name: '监控告警', count: 35, icon: '📈' },
                  { name: '安全工具', count: 20, icon: '🔒' }
                ].map((category, index) => (
                  <div key={index} className="border rounded-lg p-4 text-center hover:shadow-md transition-shadow cursor-pointer">
                    <div className="text-2xl mb-2">{category.icon}</div>
                    <div className="font-medium text-gray-900">{category.name}</div>
                    <div className="text-sm text-gray-500">{category.count} 个服务</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </main>

      {/* 页脚 */}
      <footer className="bg-white border-t mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
            <div>
              <div className="flex items-center">
                <Zap className="w-6 h-6 text-blue-600" />
                <span className="ml-2 text-lg font-bold text-gray-900">VoiceHelper</span>
              </div>
              <p className="mt-2 text-sm text-gray-500">
                企业级智能对话和语音处理平台
              </p>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-3">产品</h3>
              <ul className="space-y-2 text-sm text-gray-500">
                <li><a href="#" className="hover:text-gray-700">API 文档</a></li>
                <li><a href="#" className="hover:text-gray-700">SDK 下载</a></li>
                <li><a href="#" className="hover:text-gray-700">服务集成</a></li>
                <li><a href="#" className="hover:text-gray-700">定价方案</a></li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-3">开发者</h3>
              <ul className="space-y-2 text-sm text-gray-500">
                <li><a href="#" className="hover:text-gray-700">快速开始</a></li>
                <li><a href="#" className="hover:text-gray-700">示例代码</a></li>
                <li><a href="#" className="hover:text-gray-700">社区论坛</a></li>
                <li><a href="#" className="hover:text-gray-700">状态页面</a></li>
              </ul>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-gray-900 mb-3">支持</h3>
              <ul className="space-y-2 text-sm text-gray-500">
                <li><a href="#" className="hover:text-gray-700">帮助中心</a></li>
                <li><a href="#" className="hover:text-gray-700">联系我们</a></li>
                <li><a href="#" className="hover:text-gray-700">服务条款</a></li>
                <li><a href="#" className="hover:text-gray-700">隐私政策</a></li>
              </ul>
            </div>
          </div>
          
          <div className="mt-8 pt-8 border-t border-gray-200">
            <div className="flex items-center justify-between">
              <p className="text-sm text-gray-500">
                © 2025 VoiceHelper AI. All rights reserved.
              </p>
              <div className="flex items-center space-x-4">
                <span className="flex items-center text-sm text-gray-500">
                  <CheckCircle className="w-4 h-4 mr-1 text-green-500" />
                  服务正常
                </span>
                <span className="text-sm text-gray-500">v1.9.0</span>
              </div>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
