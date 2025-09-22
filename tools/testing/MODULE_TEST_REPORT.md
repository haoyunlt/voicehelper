# 模块测试报告
**生成时间**: 2025-09-22T16:01:39.975378

## 📊 测试总览
- **总测试数**: 9
- **通过**: 9 ✅
- **失败**: 0 ❌
- **错误**: 0 ⚠️
- **成功率**: 100.0%
- **总耗时**: 0.02秒

## 📋 详细结果
### Backend Api
- ✅ **健康检查接口** (8.6ms)
  - response: {'build_time': 'unknown', 'git_commit': 'unknown', 'status': 'ok', 'timestamp': 1758528099, 'version': 'dev'}
  - status_code: 200
- ✅ **API响应时间测试** (1.2ms)
  - target: < 200ms
  - actual: 1.2ms

### Algorithm Service
- ✅ **算法服务健康检查** (3.3ms)
  - response: {'status': 'healthy'}
  - status_code: 200

### Chat Functionality
- ✅ **对话场景: 产品咨询多轮对话** (100.0ms)
  - scenario: 产品咨询多轮对话
  - category: multi_turn
  - turns: 3
- ✅ **对话场景: 客户投诉处理** (100.0ms)
  - scenario: 客户投诉处理
  - category: emotion_analysis
  - turns: 2

### Voice Functionality
- ✅ **ASR测试: clear_speech** (100.0ms)
  - expected: 你好，我想了解一下你们的智能聊天机器人产品
  - category: clear_speech
  - language: zh-CN
  - duration: 3.2
- ✅ **ASR测试: noisy_environment** (100.0ms)
  - expected: 请问你们的技术支持服务时间是什么时候
  - category: noisy_environment
  - language: zh-CN
  - duration: 4.1
- ✅ **ASR测试: accented_speech** (100.0ms)
  - expected: 我需要预订明天下午的会议室
  - category: accented_speech
  - language: zh-CN
  - duration: 3.8

### Performance
- ✅ **并发请求测试** (4.5ms)
  - concurrent_requests: 10
  - success_count: 10
  - success_rate: 100.0%
