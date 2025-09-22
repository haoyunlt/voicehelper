// 语音增强聊天页面 - 完整版实现
const app = getApp()

Page({
  data: {
    messages: [],
    inputValue: '',
    isRecording: false,
    isPlaying: false,
    scrollToView: '',
    showVoiceInput: true,
    showAgentStatus: false,
    agentStatus: '',
    references: [],
    wsConnected: false,
    currentRequestId: null,
    audioQueue: [], // 音频播放队列
    isProcessing: false,
    currentTranscript: '', // 实时转写文本
    connectionQuality: 'good', // good, fair, poor
    latencyStats: {
      asr: 0,
      llm: 0,
      tts: 0,
      e2e: 0
    },
    voiceConfig: {
      lang: 'zh_CN',
      sampleRate: 16000,
      frameSize: 640, // 40ms at 16kHz
      vadEnabled: true,
      minSpeechMs: 200,
      maxSilenceMs: 500
    }
  },

  // WebSocket相关
  ws: null,
  audioFrameBuffer: [],
  playbackBuffer: [],
  reconnectTimer: null,
  sequenceNumber: 0,
  latencyTracker: new Map(),
  
  // 音频相关
  recorderManager: null,
  audioContext: null,
  innerAudioContext: null,
  
  onLoad(options) {
    // 检查登录状态
    if (!app.globalData.token) {
      wx.redirectTo({
        url: '/pages/login/login'
      })
      return
    }
    
    // 初始化会话
    this.initConversation(options.conversationId)
    
    // 初始化音频
    this.initAudio()
    
    // 连接WebSocket
    this.connectWebSocket()
    
    // 加载历史消息
    this.loadHistory()
  },

  onUnload() {
    // 清理资源
    this.cleanup()
  },

  onHide() {
    // 停止录音和播放
    this.stopRecording()
    this.stopPlayback()
  },

  onShow() {
    // 重新连接WebSocket（如果需要）
    if (!this.data.wsConnected) {
      this.connectWebSocket()
    }
  },

  // 初始化会话
  initConversation(conversationId) {
    if (conversationId) {
      app.globalData.currentConversation = { id: conversationId }
    } else {
      app.createConversation()
    }
  },

  // 初始化音频
  initAudio() {
    try {
      // 初始化录音管理器
      this.recorderManager = wx.getRecorderManager()
      
      // 录音事件监听
      this.recorderManager.onStart(() => {
        console.log('录音开始')
        this.setData({ isRecording: true })
      })
      
      this.recorderManager.onStop((res) => {
        console.log('录音结束', res)
        this.setData({ isRecording: false })
        
        // 处理录音结果
        if (res.tempFilePath) {
          this.processRecordedAudio(res.tempFilePath)
        }
      })
      
      this.recorderManager.onError((err) => {
        console.error('录音错误', err)
        this.setData({ isRecording: false })
        wx.showToast({
          title: '录音失败',
          icon: 'error'
        })
      })
      
      // 实时音频帧监听
      this.recorderManager.onFrameRecorded((res) => {
        if (this.data.wsConnected && res.frameBuffer) {
          this.sendAudioFrame(res.frameBuffer)
        }
      })
      
      // 初始化音频播放
      this.innerAudioContext = wx.createInnerAudioContext()
      this.innerAudioContext.onError((err) => {
        console.error('音频播放错误', err)
      })
      
      this.innerAudioContext.onEnded(() => {
        this.playNextAudio()
      })
      
    } catch (error) {
      console.error('音频初始化失败', error)
    }
  },

  // 连接WebSocket
  connectWebSocket() {
    if (this.ws && this.ws.readyState === 1) {
      return
    }

    const url = `${app.globalData.wsUrl}/voice/stream?token=${app.globalData.token}`
    
    this.ws = wx.connectSocket({
      url: url,
      header: {
        'X-Tenant-ID': app.globalData.tenantId || 'default'
      },
      success: () => {
        console.log('WebSocket连接请求已发送')
      },
      fail: (err) => {
        console.error('WebSocket连接失败', err)
        wx.showToast({
          title: '连接失败',
          icon: 'error'
        })
        this.scheduleReconnect()
      }
    })
    
    // WebSocket事件处理
    this.ws.onOpen(() => {
      console.log('WebSocket已连接')
      this.setData({ wsConnected: true })
      
      // 清除重连定时器
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer)
        this.reconnectTimer = null
      }
      
      // 发送初始化消息
      this.sendMessage({
        type: 'start',
        codec: 'pcm16',
        sample_rate: this.data.voiceConfig.sampleRate,
        conversation_id: app.globalData.currentConversation.id,
        lang: this.data.voiceConfig.lang,
        vad: {
          enable: this.data.voiceConfig.vadEnabled,
          min_speech_ms: this.data.voiceConfig.minSpeechMs,
          min_silence_ms: this.data.voiceConfig.maxSilenceMs
        }
      })
    })
    
    this.ws.onMessage((res) => {
      this.handleWebSocketMessage(res.data)
    })
    
    this.ws.onError((err) => {
      console.error('WebSocket错误', err)
      this.setData({ 
        wsConnected: false,
        connectionQuality: 'poor'
      })
      this.scheduleReconnect()
    })
    
    this.ws.onClose(() => {
      console.log('WebSocket已关闭')
      this.setData({ wsConnected: false })
      this.scheduleReconnect()
    })
  },

  // 处理WebSocket消息
  handleWebSocketMessage(data) {
    try {
      let message
      
      // 尝试解析JSON消息
      if (typeof data === 'string') {
        message = JSON.parse(data)
      } else if (data instanceof ArrayBuffer) {
        // 二进制音频数据
        this.handleAudioData(data)
        return
      } else {
        console.warn('未知消息格式', typeof data)
        return
      }
      
      const now = Date.now()
      
      switch (message.type) {
        case 'connected':
          console.log('语音会话已连接:', message.session_id)
          break
          
        case 'asr_partial':
          // 实时转写
          this.setData({
            currentTranscript: message.text || ''
          })
          break
          
        case 'asr_final':
          // 最终转写结果
          this.addMessage({
            id: Date.now(),
            type: 'user',
            content: message.text || '',
            timestamp: new Date(),
            modality: 'voice',
            confidence: message.confidence
          })
          
          this.setData({ currentTranscript: '' })
          
          // 计算ASR延迟
          if (message.trace_id && this.latencyTracker.has(message.trace_id)) {
            const startTime = this.latencyTracker.get(message.trace_id)
            this.updateLatencyStats('asr', now - startTime)
          }
          break
          
        case 'llm_delta':
          // LLM流式响应
          this.handleLLMDelta(message.delta || '')
          break
          
        case 'llm_done':
          // LLM响应完成
          this.handleLLMDone()
          break
          
        case 'tts_chunk':
          // TTS音频块
          if (message.audio) {
            this.queueAudioForPlayback(message.audio, message.format)
          }
          break
          
        case 'references':
          // 参考资料
          this.setData({
            references: message.references || []
          })
          break
          
        case 'agent_plan':
          // Agent规划
          this.setData({
            showAgentStatus: true,
            agentStatus: `规划: ${message.plan?.reasoning || '制定执行计划'}`
          })
          break
          
        case 'agent_step':
          // Agent步骤
          this.setData({
            agentStatus: `执行: ${message.description || '处理中'}`
          })
          break
          
        case 'agent_complete':
          // Agent完成
          this.setData({
            showAgentStatus: false,
            agentStatus: ''
          })
          break
          
        case 'throttle':
          // 限流警告
          console.warn('连接被限流:', message.reason)
          this.setData({ connectionQuality: 'poor' })
          break
          
        case 'heartbeat':
          // 心跳响应
          this.updateConnectionQuality(message.latency_ms || 0)
          break
          
        case 'error':
          console.error('语音服务错误:', message)
          wx.showToast({
            title: message.message || '服务错误',
            icon: 'error'
          })
          break
          
        default:
          console.log('未知消息类型:', message.type)
      }
      
    } catch (error) {
      console.error('处理WebSocket消息失败:', error)
    }
  },

  // 发送WebSocket消息
  sendMessage(message) {
    if (this.ws && this.data.wsConnected) {
      this.ws.send({
        data: JSON.stringify(message)
      })
    }
  },

  // 发送音频帧
  sendAudioFrame(frameBuffer) {
    if (!this.ws || !this.data.wsConnected) {
      return
    }
    
    try {
      // 构建二进制帧头部（20字节）
      const header = new ArrayBuffer(20)
      const headerView = new DataView(header)
      
      headerView.setUint32(0, this.sequenceNumber++, true) // sequence
      headerView.setUint32(4, this.data.voiceConfig.sampleRate, true) // sample_rate
      headerView.setUint8(8, 1, true) // channels
      headerView.setUint16(9, frameBuffer.byteLength, true) // frame_size
      headerView.setBigUint64(12, BigInt(Date.now()), true) // timestamp
      
      // 合并头部和音频数据
      const frame = new ArrayBuffer(header.byteLength + frameBuffer.byteLength)
      new Uint8Array(frame).set(new Uint8Array(header), 0)
      new Uint8Array(frame).set(new Uint8Array(frameBuffer), header.byteLength)
      
      // 发送二进制数据
      this.ws.send({
        data: frame
      })
      
      // 记录发送时间用于延迟计算
      this.latencyTracker.set(`frame_${this.sequenceNumber}`, Date.now())
      
    } catch (error) {
      console.error('发送音频帧失败:', error)
    }
  },

  // 处理接收到的音频数据
  handleAudioData(audioBuffer) {
    // 将音频数据加入播放队列
    this.data.audioQueue.push({
      buffer: audioBuffer,
      timestamp: Date.now()
    })
    
    // 如果没有在播放，开始播放
    if (!this.data.isPlaying) {
      this.playNextAudio()
    }
  },

  // 播放下一个音频
  playNextAudio() {
    if (this.data.audioQueue.length === 0) {
      this.setData({ isPlaying: false })
      return
    }
    
    this.setData({ isPlaying: true })
    
    try {
      const audioItem = this.data.audioQueue.shift()
      
      // 将ArrayBuffer转换为临时文件
      const fs = wx.getFileSystemManager()
      const tempPath = `${wx.env.USER_DATA_PATH}/temp_audio_${Date.now()}.wav`
      
      fs.writeFileSync(tempPath, audioItem.buffer)
      
      // 播放音频
      this.innerAudioContext.src = tempPath
      this.innerAudioContext.play()
      
    } catch (error) {
      console.error('播放音频失败:', error)
      this.playNextAudio() // 尝试播放下一个
    }
  },

  // 开始录音
  startRecording() {
    if (!this.data.wsConnected) {
      wx.showToast({
        title: '请等待连接',
        icon: 'none'
      })
      return
    }
    
    try {
      this.recorderManager.start({
        duration: 60000, // 最长60秒
        sampleRate: this.data.voiceConfig.sampleRate,
        numberOfChannels: 1,
        encodeBitRate: 48000,
        format: 'PCM',
        frameSize: this.data.voiceConfig.frameSize
      })
      
      // 发送开始录音消息
      this.sendMessage({
        type: 'start_recording',
        timestamp: Date.now()
      })
      
    } catch (error) {
      console.error('开始录音失败:', error)
      wx.showToast({
        title: '录音失败',
        icon: 'error'
      })
    }
  },

  // 停止录音
  stopRecording() {
    if (this.data.isRecording) {
      this.recorderManager.stop()
      
      // 发送停止录音消息
      this.sendMessage({
        type: 'stop_recording',
        timestamp: Date.now()
      })
    }
  },

  // 切换录音状态
  toggleRecording() {
    if (this.data.isRecording) {
      this.stopRecording()
    } else {
      this.startRecording()
    }
  },

  // 停止播放
  stopPlayback() {
    if (this.innerAudioContext) {
      this.innerAudioContext.stop()
    }
    
    // 清空播放队列
    this.setData({ 
      audioQueue: [],
      isPlaying: false 
    })
  },

  // 处理LLM流式响应
  handleLLMDelta(delta) {
    const messages = this.data.messages
    let lastMessage = messages[messages.length - 1]
    
    // 如果最后一条消息不是助手消息，创建新的
    if (!lastMessage || lastMessage.type !== 'assistant' || lastMessage.isComplete) {
      lastMessage = {
        id: Date.now(),
        type: 'assistant',
        content: '',
        timestamp: new Date(),
        modality: 'voice',
        isStreaming: true
      }
      messages.push(lastMessage)
    }
    
    // 追加内容
    lastMessage.content += delta
    lastMessage.timestamp = new Date()
    
    this.setData({
      messages: messages,
      scrollToView: `msg-${lastMessage.id}`
    })
  },

  // 处理LLM响应完成
  handleLLMDone() {
    const messages = this.data.messages
    const lastMessage = messages[messages.length - 1]
    
    if (lastMessage && lastMessage.type === 'assistant') {
      lastMessage.isStreaming = false
      lastMessage.isComplete = true
      
      this.setData({
        messages: messages,
        isProcessing: false
      })
    }
  },

  // 添加消息
  addMessage(message) {
    const messages = this.data.messages
    messages.push(message)
    
    this.setData({
      messages: messages,
      scrollToView: `msg-${message.id}`
    })
  },

  // 更新延迟统计
  updateLatencyStats(type, latency) {
    const stats = { ...this.data.latencyStats }
    stats[type] = latency
    
    this.setData({ latencyStats: stats })
  },

  // 更新连接质量
  updateConnectionQuality(latency) {
    let quality = 'good'
    
    if (latency > 300) {
      quality = 'poor'
    } else if (latency > 100) {
      quality = 'fair'
    }
    
    this.setData({ connectionQuality: quality })
  },

  // 计划重连
  scheduleReconnect() {
    if (this.reconnectTimer) return
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null
      this.connectWebSocket()
    }, 3000)
  },

  // 加载历史消息
  loadHistory() {
    // TODO: 从本地存储或服务器加载历史消息
    console.log('加载历史消息')
  },

  // 清理资源
  cleanup() {
    // 清理定时器
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
    
    // 关闭WebSocket
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    
    // 停止录音和播放
    this.stopRecording()
    this.stopPlayback()
    
    // 清理音频资源
    if (this.innerAudioContext) {
      this.innerAudioContext.destroy()
      this.innerAudioContext = null
    }
  },

  // 文本输入处理
  onInputChange(e) {
    this.setData({
      inputValue: e.detail.value
    })
  },

  // 发送文本消息
  sendTextMessage() {
    const content = this.data.inputValue.trim()
    if (!content) return
    
    // 添加用户消息
    this.addMessage({
      id: Date.now(),
      type: 'user',
      content: content,
      timestamp: new Date(),
      modality: 'text'
    })
    
    // 清空输入
    this.setData({ inputValue: '' })
    
    // 发送到服务器
    this.sendMessage({
      type: 'text_message',
      content: content,
      conversation_id: app.globalData.currentConversation.id,
      timestamp: Date.now()
    })
  },

  // 复制消息
  copyMessage(e) {
    const content = e.currentTarget.dataset.content
    wx.setClipboardData({
      data: content,
      success: () => {
        wx.showToast({
          title: '已复制',
          icon: 'success'
        })
      }
    })
  },

  // 查看引用详情
  viewReference(e) {
    const reference = e.currentTarget.dataset.reference
    
    wx.showModal({
      title: reference.title || '参考资料',
      content: reference.content || reference.source,
      showCancel: false,
      confirmText: '确定'
    })
  }
})