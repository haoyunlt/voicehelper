# VoiceHelper 测试数据集

这个目录包含了VoiceHelper项目的各种测试数据集，用于验证系统的功能、性能和安全性。

## 📁 目录结构

```
datasets/
├── chat/                           # 对话测试数据
│   ├── comprehensive_conversation_data.json
│   ├── conversation_scenarios.json
│   ├── intent_classification.json
│   └── emotion_analysis.json
├── voice/                          # 语音测试数据
│   ├── comprehensive_voice_test_data.json
│   ├── asr_test_cases.json
│   ├── tts_test_cases.json
│   └── voice_emotion_test.json
├── rag/                           # RAG知识库数据
│   ├── comprehensive_knowledge_base.json
│   └── knowledge_base_samples.json
├── security/                      # 安全测试数据
│   ├── comprehensive_security_test_data.json
│   └── security_test_cases.json
├── performance/                   # 性能测试数据
│   ├── comprehensive_performance_test_data.json
│   └── load_test_scenarios.json
├── multimodal/                    # 多模态测试数据
│   ├── comprehensive_multimodal_test_data.json
│   └── fusion_test_cases.json
├── agent/                         # 智能代理测试数据
│   ├── tool_calling_test.json
│   └── reasoning_chain_test.json
└── integration/                   # 集成测试数据
    └── api_integration_test.json
```

## 🎯 数据集类型

### 1. 对话测试数据 (Chat)
- **综合对话数据**: 包含多轮对话、意图识别、情感分析等场景
- **意图分类**: 用户意图识别测试用例
- **情感分析**: 对话情感识别和分析数据
- **上下文理解**: 多轮对话上下文保持测试

**主要场景**:
- 产品咨询对话
- 技术支持请求
- 用户投诉处理
- 多语言对话
- 复杂业务流程

### 2. 语音测试数据 (Voice)
- **ASR测试**: 语音识别准确性和鲁棒性测试
- **TTS测试**: 语音合成质量和自然度测试
- **情感语音**: 语音情感识别和合成测试
- **多语言支持**: 不同语言的语音处理测试

**测试条件**:
- 清晰语音环境
- 噪音环境
- 不同口音和方言
- 情感语调变化
- 实时流式处理

### 3. RAG知识库数据 (RAG)
- **产品文档**: VoiceHelper功能和使用说明
- **技术文档**: 系统架构和API文档
- **故障排查**: 常见问题和解决方案
- **业务政策**: 商业条款和政策说明

**测试查询**:
- 简单事实查询
- 复杂推理查询
- 多文档综合查询
- 边界情况查询

### 4. 安全测试数据 (Security)
- **认证测试**: JWT令牌、API密钥验证
- **授权测试**: 基于角色的访问控制
- **输入验证**: SQL注入、XSS攻击防护
- **数据保护**: 加密、脱敏、合规性测试

**安全场景**:
- 恶意输入检测
- 权限绕过尝试
- 数据泄露防护
- 合规性验证

### 5. 性能测试数据 (Performance)
- **负载测试**: 不同并发用户数下的性能表现
- **压力测试**: 系统极限负载测试
- **并发测试**: 并发访问和竞态条件测试
- **内存测试**: 内存泄漏和垃圾回收测试

**性能指标**:
- 响应时间分布
- 吞吐量测试
- 资源使用率
- 可扩展性验证

### 6. 多模态测试数据 (Multimodal)
- **跨模态理解**: 文本、图像、语音融合理解
- **注意力机制**: 跨模态注意力测试
- **融合策略**: 不同融合方法的效果对比
- **性能优化**: 多模态处理效率测试

**融合场景**:
- 文本+图像理解
- 语音+文本分析
- 多模态推理
- 时序对齐测试

## 🚀 使用方法

### 1. 直接使用现有数据集

```bash
# 查看对话测试数据
cat datasets/chat/comprehensive_conversation_data.json

# 使用语音测试数据
python -c "
import json
with open('datasets/voice/comprehensive_voice_test_data.json') as f:
    data = json.load(f)
    print(f'语音测试用例数量: {data[\"metadata\"][\"total_samples\"]}')
"
```

### 2. 使用测试数据生成器

```bash
# 生成所有类型的测试数据
python ../test_data_generator.py --type all

# 生成特定类型的数据
python ../test_data_generator.py --type chat --count 100 --language zh-CN
python ../test_data_generator.py --type voice --count 200 --language en-US
python ../test_data_generator.py --type security --count 150

# 指定输出目录
python ../test_data_generator.py --type all --output-dir custom_test_data
```

### 3. 在测试中使用

```python
import json
from pathlib import Path

# 加载对话测试数据
def load_chat_test_data():
    with open('datasets/chat/comprehensive_conversation_data.json') as f:
        return json.load(f)

# 加载安全测试数据
def load_security_test_data():
    with open('datasets/security/comprehensive_security_test_data.json') as f:
        return json.load(f)

# 使用示例
chat_data = load_chat_test_data()
for scenario in chat_data['scenarios']:
    print(f"测试场景: {scenario['title']}")
    # 执行测试逻辑...
```

## 📊 数据统计

| 数据类型 | 文件数量 | 测试用例数 | 覆盖场景 |
|---------|---------|-----------|---------|
| 对话测试 | 4 | 100+ | 多轮对话、意图识别、情感分析 |
| 语音测试 | 4 | 200+ | ASR、TTS、情感语音、多语言 |
| RAG测试 | 2 | 300+ | 文档检索、知识问答、语义搜索 |
| 安全测试 | 2 | 150+ | 认证、授权、输入验证、数据保护 |
| 性能测试 | 2 | 80+ | 负载、压力、并发、内存测试 |
| 多模态测试 | 2 | 120+ | 跨模态融合、注意力机制、推理 |

## 🔧 数据格式说明

### 通用元数据格式
```json
{
  "metadata": {
    "name": "数据集名称",
    "version": "版本号",
    "created_at": "创建时间",
    "updated_at": "更新时间",
    "total_samples": "样本总数",
    "categories": ["分类列表"],
    "languages": ["支持语言"]
  }
}
```

### 测试用例格式
```json
{
  "id": "唯一标识符",
  "category": "测试类别",
  "priority": "优先级",
  "test_data": {
    "input": "输入数据",
    "context": "上下文信息"
  },
  "expected_result": {
    "output": "预期输出",
    "metrics": "评估指标"
  },
  "evaluation_criteria": {
    "accuracy_threshold": "准确率阈值",
    "performance_target": "性能目标"
  }
}
```

## 🎨 数据质量保证

### 1. 数据验证
- JSON格式验证
- 必填字段检查
- 数据类型验证
- 取值范围验证

### 2. 一致性检查
- 跨文件引用一致性
- 标识符唯一性
- 分类标准统一性
- 版本兼容性

### 3. 覆盖率分析
- 功能覆盖率统计
- 场景覆盖率分析
- 边界情况覆盖
- 异常情况覆盖

## 📈 持续更新

### 更新策略
- **定期更新**: 每月更新一次基础数据集
- **功能驱动**: 新功能发布时同步更新测试数据
- **问题驱动**: 发现测试盲点时及时补充数据
- **社区贡献**: 接受社区贡献的高质量测试数据

### 版本管理
- 使用语义化版本号 (Semantic Versioning)
- 向后兼容性保证
- 变更日志记录
- 迁移指南提供

## 🤝 贡献指南

### 添加新的测试数据
1. 遵循现有的数据格式规范
2. 确保数据质量和准确性
3. 添加必要的元数据和文档
4. 通过数据验证测试

### 改进现有数据
1. 提交Issue说明改进建议
2. Fork项目并创建特性分支
3. 实施改进并添加测试
4. 提交Pull Request

### 数据格式规范
- 使用UTF-8编码
- JSON格式，2空格缩进
- 字段命名使用snake_case
- 添加适当的注释和说明

## 📞 支持与反馈

如果您在使用测试数据集时遇到问题或有改进建议，请：

1. 查看现有的Issue和文档
2. 在GitHub上提交Issue
3. 联系开发团队
4. 参与社区讨论

---

**注意**: 这些测试数据仅用于开发和测试目的，请勿在生产环境中使用敏感或真实的用户数据。
