// 聊天页面 - 最新版本（整合了WebSocket和SSE流式处理）
const app = getApp();

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
    currentTranscript: '' // 实时转写文本
  },

  // WebSocket相关
  ws: null,
  audioFrameBuffer: [],
  playbackBuffer: [],
  reconnectTimer: null,
  
  onLoad(options) {
    // 检查登录状态
    if (!app.globalData.token) {
      wx.redirectTo({
        url: '/pages/login/login'
      });
      return;
    }
    
    // 初始化会话
    this.initConversation(options.conversationId);
    
    // 连接WebSocket
    this.connectWebSocket();
    
    // 设置音频回调
    this.setupAudioCallbacks();
    
    // 加载历史消息
    this.loadHistory();
  },

  onUnload() {
    // 清理定时器
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }
    
    // 断开WebSocket
    if (this.ws) {
      this.ws.close();
    }
    
    // 停止录音
    if (this.data.isRecording) {
      app.stopRecording();
    }
    
    // 停止播放
    if (this.data.isPlaying) {
      app.stopAudio();
    }
  },

  // 初始化会话
  initConversation(conversationId) {
    if (conversationId) {
      app.globalData.currentConversation = {
        id: conversationId,
        messages: []
      };
    } else {
      app.createConversation();
    }
  },

  // 连接WebSocket
  connectWebSocket() {
    const url = `${app.globalData.wsUrl}/voice/stream?token=${app.globalData.token}`;
    
    this.ws = wx.connectSocket({
      url: url,
      header: {
        'X-Tenant-ID': app.globalData.tenantId
      },
      success: () => {
        console.log('WebSocket连接请求已发送');
      },
      fail: (err) => {
        console.error('WebSocket连接失败', err);
        wx.showToast({
          title: '连接失败',
          icon: 'error'
        });
        this.scheduleReconnect();
      }
    });
    
    // WebSocket事件处理
    this.ws.onOpen(() => {
      console.log('WebSocket已连接');
      this.setData({ wsConnected: true });
      
      // 清除重连定时器
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
      
      // 发送初始化消息
      this.ws.send({
        data: JSON.stringify({
          type: 'start',
          codec: 'pcm16',
          sample_rate: 16000,
          conversation_id: app.globalData.currentConversation.id,
          lang: 'zh-CN',
          vad: {
            enable: true,
            min_speech_ms: 200,
            min_silence_ms: 250
          }
        })
      });
    });
    
    this.ws.onMessage((res) => {
      this.handleWebSocketMessage(res.data);
    });
    
    this.ws.onError((err) => {
      console.error('WebSocket错误', err);
      this.setData({ wsConnected: false });
      this.scheduleReconnect();
    });
    
    this.ws.onClose(() => {
      console.log('WebSocket已关闭');
      this.setData({ wsConnected: false });
      this.scheduleReconnect();
    });
  },

  // 计划重连
  scheduleReconnect() {
    if (this.reconnectTimer) return;
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      if (app.globalData.isConnected) {
        console.log('尝试重新连接WebSocket...');
        this.connectWebSocket();
      }
    }, 3000);
  },

  // 处理WebSocket消息
  handleWebSocketMessage(data) {
    try {
      const message = JSON.parse(data);
      
      switch (message.type) {
        case 'asr_partial':
          this.handleASRPartial(message);
          break;
        case 'asr_final':
          this.handleASRFinal(message);
          break;
        case 'llm_delta':
          this.handleLLMDelta(message);
          break;
        case 'tts_chunk':
          this.handleTTSChunk(message);
          break;
        case 'refs':
          this.handleReferences(message);
          break;
        case 'agent_plan':
          this.handleAgentPlan(message);
          break;
        case 'done':
          this.handleDone(message);
          break;
        case 'error':
          this.handleError(message);
          break;
        case 'interrupted':
          this.handleInterrupted(message);
          break;
      }
    } catch (err) {
      console.error('处理WebSocket消息失败', err);
    }
  },

  // 处理ASR部分结果
  handleASRPartial(message) {
    // 更新实时字幕
    this.setData({
      currentTranscript: message.text
    });
  },

  // 处理ASR最终结果
  handleASRFinal(message) {
    // 添加用户消息
    this.addMessage({
      role: 'user',
      content: message.text,
      modality: 'voice',
      timestamp: new Date().toISOString()
    });
    
    this.setData({
      currentTranscript: ''
    });
  },

  // 处理LLM增量输出
  handleLLMDelta(message) {
    const messages = this.data.messages;
    const lastMessage = messages[messages.length - 1];
    
    if (lastMessage && lastMessage.role === 'assistant') {
      // 更新最后一条助手消息
      lastMessage.content += message.text;
      this.setData({
        messages: messages,
        scrollToView: `msg-${messages.length - 1}`
      });
    } else {
      // 创建新的助手消息
      this.addMessage({
        role: 'assistant',
        content: message.text,
        modality: 'voice',
        timestamp: new Date().toISOString()
      });
    }
  },

  // 处理TTS音频块
  handleTTSChunk(message) {
    // 将base64音频解码并加入播放队列
    const audioData = wx.base64ToArrayBuffer(message.chunk);
    this.audioQueue.push(audioData);
    
    // 开始播放
    if (!this.data.isPlaying) {
      this.playNextAudio();
    }
  },

  // 处理引用
  handleReferences(message) {
    this.setData({
      references: message.items || message.references || []
    });
  },

  // 处理Agent计划
  handleAgentPlan(message) {
    this.setData({
      showAgentStatus: true,
      agentStatus: `计划: ${message.items.join(' → ')}`
    });
  },

  // 处理完成
  handleDone(message) {
    this.setData({
      isProcessing: false,
      showAgentStatus: false,
      currentRequestId: null
    });
    
    // 显示统计信息
    if (message.usage) {
      console.log('使用统计', message.usage);
    }
  },

  // 处理错误
  handleError(message) {
    const errorMsg = message.error?.message || message.message || '处理失败';
    wx.showToast({
      title: errorMsg,
      icon: 'error'
    });
    
    this.setData({
      isProcessing: false,
      showAgentStatus: false
    });
  },

  // 处理中断
  handleInterrupted(message) {
    console.log('已中断', message.request_id);
    
    // 清空音频队列
    this.audioQueue = [];
    
    // 停止播放
    if (this.data.isPlaying) {
      app.stopAudio();
      this.setData({ isPlaying: false });
    }
  },

  // 设置音频回调
  setupAudioCallbacks() {
    // 录音帧回调
    app.audioFrameCallback = (frameBuffer) => {
      if (this.data.isRecording && this.ws && this.data.wsConnected) {
        // 发送音频帧到服务器
        const base64 = wx.arrayBufferToBase64(frameBuffer);
        this.ws.send({
          data: JSON.stringify({
            type: 'audio',
            seq: this.audioFrameBuffer.length,
            chunk: base64
          })
        });
        
        this.audioFrameBuffer.push(frameBuffer);
      }
    };
    
    // 录音停止回调
    app.audioStopCallback = (res) => {
      console.log('录音完成', res);
      
      // 发送停止信号
      if (this.ws && this.data.wsConnected) {
        this.ws.send({
          data: JSON.stringify({
            type: 'stop'
          })
        });
      }
      
      this.setData({
        isRecording: false,
        isProcessing: true
      });
      
      // 清空缓冲区
      this.audioFrameBuffer = [];
    };
  },

  // 播放下一个音频
  playNextAudio() {
    if (this.audioQueue.length === 0) {
      this.setData({ isPlaying: false });
      return;
    }
    
    const audioData = this.audioQueue.shift();
    
    // 使用WebAudioContext播放PCM数据
    const audioContext = app.globalData.audioContext;
    const sampleRate = 16000;
    const audioBuffer = audioContext.createBuffer(1, audioData.byteLength / 2, sampleRate);
    
    // 转换PCM16到Float32
    const channelData = audioBuffer.getChannelData(0);
    const int16Array = new Int16Array(audioData);
    for (let i = 0; i < int16Array.length; i++) {
      channelData[i] = int16Array[i] / 32768.0;
    }
    
    // 创建音频源并播放
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioContext.destination);
    
    source.onended = () => {
      // 播放下一个音频块
      this.playNextAudio();
    };
    
    source.start();
    this.setData({ isPlaying: true });
  },

  // 加载历史消息
  loadHistory() {
    const conversationId = app.globalData.currentConversation?.id;
    if (!conversationId) return;
    
    wx.request({
      url: `${app.globalData.apiUrl}/conversations/${conversationId}/messages`,
      method: 'GET',
      header: {
        'Authorization': `Bearer ${app.globalData.token}`,
        'X-Tenant-ID': app.globalData.tenantId
      },
      success: (res) => {
        if (res.statusCode === 200) {
          const messages = res.data.messages || [];
          this.setData({
            messages: messages,
            scrollToView: messages.length > 0 ? `msg-${messages.length - 1}` : ''
          });
        }
      }
    });
  },

  // 添加消息
  addMessage(message) {
    const messages = this.data.messages;
    messages.push({
      ...message,
      id: `msg-${messages.length}`
    });
    
    this.setData({
      messages: messages,
      scrollToView: `msg-${messages.length - 1}`
    });
    
    // 保存到全局
    if (app.globalData.currentConversation) {
      app.globalData.currentConversation.messages = messages;
    }
  },

  // 文本输入
  onInputChange(e) {
    this.setData({
      inputValue: e.detail.value
    });
  },

  // 发送文本消息
  sendTextMessage() {
    const content = this.data.inputValue.trim();
    if (!content) return;
    
    // 添加用户消息
    this.addMessage({
      role: 'user',
      content: content,
      modality: 'text',
      timestamp: new Date().toISOString()
    });
    
    // 清空输入
    this.setData({
      inputValue: '',
      isProcessing: true
    });
    
    // 发送请求
    this.sendChatRequest(content, 'text');
  },

  // 发送聊天请求（SSE流式）
  sendChatRequest(content, modality) {
    const requestId = `req_${Date.now()}`;
    this.setData({ currentRequestId: requestId });
    
    // 创建SSE连接
    const requestTask = wx.request({
      url: `${app.globalData.apiUrl}/chat/stream`,
      method: 'POST',
      header: {
        'Authorization': `Bearer ${app.globalData.token}`,
        'X-Tenant-ID': app.globalData.tenantId,
        'X-Request-ID': requestId,
        'Accept': 'text/event-stream'
      },
      data: {
        conversation_id: app.globalData.currentConversation.id,
        messages: [
          {
            role: 'user',
            content: content
          }
        ],
        modality: modality,
        top_k: 5,
        temperature: 0.3
      },
      enableChunked: true,
      success: (res) => {
        console.log('请求成功', res);
      },
      fail: (err) => {
        console.error('请求失败', err);
        wx.showToast({
          title: '发送失败',
          icon: 'error'
        });
        this.setData({ isProcessing: false });
      }
    });
    
    // 处理流式响应
    requestTask.onChunkReceived((res) => {
      const decoder = new TextDecoder('utf-8');
      const chunk = decoder.decode(res.data);
      
      // 解析SSE数据
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            this.handleSSEData(data);
          } catch (e) {
            console.error('解析SSE数据失败', e);
          }
        }
      }
    });
  },

  // 处理SSE数据
  handleSSEData(data) {
    switch (data.type) {
      case 'delta':
        this.handleLLMDelta({ text: data.content });
        break;
      case 'refs':
        this.handleReferences(data);
        break;
      case 'done':
        this.handleDone(data);
        break;
      case 'error':
        this.handleError(data);
        break;
    }
  },

  // 切换到语音输入
  toggleVoiceInput() {
    this.setData({
      showVoiceInput: !this.data.showVoiceInput
    });
  },

  // 开始录音
  startRecording() {
    // 检查WebSocket连接
    if (!this.data.wsConnected) {
      wx.showToast({
        title: '连接未就绪',
        icon: 'none'
      });
      return;
    }
    
    wx.vibrateShort(); // 震动反馈
    
    this.setData({
      isRecording: true
    });
    
    // 清空之前的音频缓冲
    this.audioFrameBuffer = [];
    
    // 开始录音
    app.startRecording();
  },

  // 停止录音
  stopRecording() {
    wx.vibrateShort();
    
    this.setData({
      isRecording: false
    });
    
    app.stopRecording();
  },

  // 长按开始录音
  onTouchStart() {
    this.startRecording();
  },

  // 松开停止录音
  onTouchEnd() {
    if (this.data.isRecording) {
      this.stopRecording();
    }
  },

  // 取消当前请求（用于打断）
  cancelCurrentRequest() {
    if (this.data.currentRequestId) {
      // 通过WebSocket发送取消请求
      if (this.ws && this.data.wsConnected) {
        this.ws.send({
          data: JSON.stringify({
            type: 'cancel',
            request_id: this.data.currentRequestId
          })
        });
      }
      
      // 通过HTTP发送取消请求
      wx.request({
        url: `${app.globalData.apiUrl}/chat/cancel`,
        method: 'POST',
        header: {
          'Authorization': `Bearer ${app.globalData.token}`,
          'X-Tenant-ID': app.globalData.tenantId
        },
        data: {
          request_id: this.data.currentRequestId
        }
      });
      
      // 清空音频队列
      this.audioQueue = [];
      
      // 停止播放
      if (this.data.isPlaying) {
        app.stopAudio();
        this.setData({ isPlaying: false });
      }
      
      this.setData({
        isProcessing: false,
        currentRequestId: null
      });
    }
  },

  // 查看引用详情
  viewReference(e) {
    const index = e.currentTarget.dataset.index;
    const reference = this.data.references[index];
    
    wx.navigateTo({
      url: `/pages/reference/reference?id=${reference.chunk_id}&source=${reference.source}`
    });
  },

  // 清空会话
  clearConversation() {
    wx.showModal({
      title: '确认清空',
      content: '是否清空当前会话？',
      success: (res) => {
        if (res.confirm) {
          app.createConversation();
          this.setData({
            messages: [],
            references: [],
            currentTranscript: ''
          });
        }
      }
    });
  },

  // 分享会话
  onShareAppMessage() {
    return {
      title: '智能助手对话',
      path: `/pages/chat/chat?conversationId=${app.globalData.currentConversation?.id}`
    };
  }
});
