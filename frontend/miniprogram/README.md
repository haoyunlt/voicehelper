# 微信小程序客户端

## 概述

这是智能聊天机器人系统的微信小程序客户端，提供了完整的文本和语音交互功能。

## 功能特性

- 🎤 **语音交互**: 支持实时语音输入和TTS语音合成
- 💬 **文本聊天**: 传统的文本消息交互
- 🔄 **流式响应**: 支持SSE和WebSocket双协议
- 🔐 **微信登录**: 集成微信授权登录
- 📱 **自适应**: 根据iOS/Android自动调整音频配置
- 🔌 **断线重连**: 自动处理网络中断和重连
- 📊 **错误上报**: 自动收集和上报错误信息

## 技术架构

### 核心文件

- `app.js` - 小程序主入口，全局状态管理
- `pages/chat/chat.js` - 聊天页面，核心交互逻辑
- `app.json` - 小程序配置文件

### 音频处理

- **录音**: 使用 `wx.getRecorderManager()` 
- **播放**: 使用 `wx.createInnerAudioContext()` 和 `wx.createWebAudioContext()`
- **格式**: Android使用MP3，iOS使用AAC
- **采样率**: 16kHz，单声道

### 网络通信

- **WebSocket**: 用于实时语音流
- **HTTP SSE**: 用于文本聊天流式响应
- **断线重连**: 3秒自动重连机制

## 开发指南

### 环境配置

1. 在微信开发者工具中导入项目
2. 配置AppID和服务器域名
3. 修改 `app.js` 中的API地址

```javascript
// app.js
globalData: {
  apiUrl: 'https://your-api-domain.com/api/v1',
  wsUrl: 'wss://your-api-domain.com/api/v1'
}
```

### 调试技巧

1. **开启调试模式**
```json
// app.json
{
  "debug": true
}
```

2. **查看网络请求**
- 在开发者工具中查看Network面板
- 检查WebSocket连接状态

3. **音频调试**
- 使用真机调试测试录音功能
- 检查音频权限是否正确授予

## API接口

### WebSocket协议

**连接地址**: `/api/v1/voice/stream`

**消息类型**:
- `start` - 初始化连接
- `audio` - 发送音频数据
- `stop` - 停止录音
- `cancel` - 取消请求

### HTTP接口

- `POST /api/v1/chat/stream` - 文本聊天
- `POST /api/v1/auth/wechat/miniprogram/login` - 微信登录
- `GET /api/v1/conversations/{id}/messages` - 获取历史消息

## 部署注意事项

1. **域名配置**: 在微信公众平台配置合法域名
2. **HTTPS**: 所有接口必须使用HTTPS
3. **WSS**: WebSocket必须使用WSS协议
4. **权限申请**: 需要申请录音权限

## 性能优化

1. **音频缓冲**: 使用队列管理音频播放
2. **消息分页**: 历史消息分页加载
3. **防抖处理**: 输入和发送添加防抖
4. **资源清理**: 页面卸载时清理定时器和连接

## 已知问题

1. iOS设备可能需要用户手动触发首次音频播放
2. Android某些机型录音格式兼容性问题
3. 网络切换时WebSocket重连可能延迟

## 更新日志

### v1.0.0 (2024-03)
- 初始版本发布
- 支持文本和语音交互
- 集成微信登录

### v1.1.0 (计划中)
- 添加会话管理功能
- 支持图片消息
- 优化音频处理性能
