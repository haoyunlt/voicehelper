
# VoiceHelper 部署验证报告

## 总体评分: 7.0/100

## 文件结构验证 (权重40%)
- ❌ backend_main
- ❌ algo_main
- ❌ frontend_chat
- ❌ websocket_voice
- ❌ rag_service
- ❌ agent_service
- ❌ monitoring_config
- ❌ grafana_dashboard
- ❌ alert_rules
- ❌ docker_compose
- ❌ k8s_deployment

## 配置文件验证 (权重35%)
- ❌ environment_config
- ✅ docker_compose_syntax
- ❌ monitoring_setup
- ❌ service_endpoints
- ❌ security_config

## 部署配置验证 (权重25%)
- ❌ docker_images
- ❌ service_dependencies
- ❌ resource_limits
- ❌ health_checks
- ❌ scaling_config

## 评估结果
- 总体评分: 7.0/100
- 评估等级: D (不合格)
- 建议: 缺失太多关键组件，需要重新检查

## DoD性能阈值提醒
请确保生产环境满足以下性能要求：
- 文本首Token P95 < 800ms
- 语音首响 P95 < 700ms
- Barge-in延迟 P95 < 150ms
- RAG Recall@5 >= 85%
- 检索P95 < 200ms
- 系统可用性 >= 99.9%
- 错误率 < 1%
- WebSocket断连率 < 5%
