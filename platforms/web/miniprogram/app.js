// 微信小程序主入口
App({
  globalData: {
    userInfo: null,
    token: null,
    tenantId: null,
    apiUrl: process.env.NEXT_PUBLIC_API_URL || 'https://api.chatbot.example.com/api/v1',
    wsUrl: process.env.NEXT_PUBLIC_WS_URL || 'wss://api.chatbot.example.com/api/v1',
    currentConversation: null,
    audioContext: null,
    recorderManager: null,
    innerAudioContext: null,
    systemInfo: null,
    audioConfig: null,
    isConnected: true,
    networkType: 'wifi',
    wasDisconnected: false
  },

  onLaunch() {
    // 初始化音频管理器
    this.initAudioManagers();
    
    // 检查登录状态
    this.checkLoginStatus();
    
    // 获取系统信息
    this.getSystemInfo();
    
    // 监听网络状态
    this.monitorNetwork();
    
    // 初始化错误上报
    this.initErrorReporting();
  },

  // 初始化音频管理器
  initAudioManagers() {
    // 录音管理器
    const recorderManager = wx.getRecorderManager();
    this.globalData.recorderManager = recorderManager;
    
    // 音频播放器
    const innerAudioContext = wx.createInnerAudioContext();
    this.globalData.innerAudioContext = innerAudioContext;
    
    // WebAudio上下文（用于高级音频处理）
    const audioContext = wx.createWebAudioContext();
    this.globalData.audioContext = audioContext;
    
    // 配置录音参数
    recorderManager.onFrameRecorded((res) => {
      // 实时音频帧回调
      if (this.audioFrameCallback) {
        this.audioFrameCallback(res.frameBuffer);
      }
    });
    
    recorderManager.onStop((res) => {
      console.log('录音停止', res);
      if (this.audioStopCallback) {
        this.audioStopCallback(res);
      }
    });
    
    recorderManager.onError((err) => {
      console.error('录音错误', err);
      wx.showToast({
        title: '录音失败',
        icon: 'error'
      });
    });
  },

  // 检查登录状态
  checkLoginStatus() {
    const token = wx.getStorageSync('token');
    const tenantId = wx.getStorageSync('tenantId');
    
    if (token) {
      this.globalData.token = token;
      this.globalData.tenantId = tenantId;
      
      // 验证token有效性
      this.validateToken(token).catch(() => {
        // Token无效，清除并重新登录
        this.clearLoginInfo();
      });
    }
  },

  // 验证Token
  validateToken(token) {
    return new Promise((resolve, reject) => {
      wx.request({
        url: `${this.globalData.apiUrl}/auth/validate`,
        method: 'GET',
        header: {
          'Authorization': `Bearer ${token}`,
          'X-Tenant-ID': this.globalData.tenantId
        },
        success: (res) => {
          if (res.statusCode === 200) {
            resolve(res.data);
          } else {
            reject(new Error('Token invalid'));
          }
        },
        fail: reject
      });
    });
  },

  // 微信登录
  wxLogin() {
    return new Promise((resolve, reject) => {
      wx.login({
        success: (res) => {
          if (res.code) {
            // 发送code到后端换取token
            wx.request({
              url: `${this.globalData.apiUrl}/auth/wechat/miniprogram/login`,
              method: 'POST',
              data: {
                code: res.code
              },
              success: (response) => {
                if (response.statusCode === 200) {
                  const { token, tenant_id, user_info } = response.data;
                  
                  // 保存登录信息
                  this.globalData.token = token;
                  this.globalData.tenantId = tenant_id;
                  this.globalData.userInfo = user_info;
                  
                  // 持久化存储
                  wx.setStorageSync('token', token);
                  wx.setStorageSync('tenantId', tenant_id);
                  wx.setStorageSync('userInfo', user_info);
                  
                  resolve(response.data);
                } else {
                  reject(new Error('Login failed'));
                }
              },
              fail: reject
            });
          } else {
            reject(new Error('微信登录失败'));
          }
        },
        fail: reject
      });
    });
  },

  // 清除登录信息
  clearLoginInfo() {
    this.globalData.token = null;
    this.globalData.tenantId = null;
    this.globalData.userInfo = null;
    wx.removeStorageSync('token');
    wx.removeStorageSync('tenantId');
    wx.removeStorageSync('userInfo');
  },

  // 获取系统信息
  getSystemInfo() {
    wx.getSystemInfo({
      success: (res) => {
        this.globalData.systemInfo = res;
        console.log('系统信息', res);
        
        // 根据系统调整配置
        if (res.platform === 'android') {
          // Android特殊配置
          this.globalData.audioConfig = {
            sampleRate: 16000,
            encodeBitRate: 48000,
            format: 'mp3'
          };
        } else {
          // iOS配置
          this.globalData.audioConfig = {
            sampleRate: 16000,
            encodeBitRate: 48000,
            format: 'aac'
          };
        }
      }
    });
  },

  // 监听网络状态
  monitorNetwork() {
    wx.onNetworkStatusChange((res) => {
      this.globalData.isConnected = res.isConnected;
      this.globalData.networkType = res.networkType;
      
      if (!res.isConnected) {
        wx.showToast({
          title: '网络已断开',
          icon: 'none'
        });
      } else if (this.globalData.wasDisconnected) {
        wx.showToast({
          title: '网络已恢复',
          icon: 'success'
        });
        // 重连WebSocket等
        if (this.reconnectCallback) {
          this.reconnectCallback();
        }
      }
      
      this.globalData.wasDisconnected = !res.isConnected;
    });
  },

  // 初始化错误上报
  initErrorReporting() {
    // 捕获未处理的Promise错误
    wx.onUnhandledRejection((res) => {
      console.error('未处理的Promise错误', res);
      this.reportError({
        type: 'unhandled_rejection',
        error: res.reason,
        promise: res.promise
      });
    });
    
    // 捕获小程序错误
    wx.onError((error) => {
      console.error('小程序错误', error);
      this.reportError({
        type: 'app_error',
        error: error
      });
    });
  },

  // 错误上报
  reportError(errorInfo) {
    if (!this.globalData.token) return;
    
    wx.request({
      url: `${this.globalData.apiUrl}/analytics/error`,
      method: 'POST',
      header: {
        'Authorization': `Bearer ${this.globalData.token}`,
        'X-Tenant-ID': this.globalData.tenantId
      },
      data: {
        ...errorInfo,
        timestamp: new Date().toISOString(),
        system_info: this.globalData.systemInfo,
        network_type: this.globalData.networkType
      }
    });
  },

  // 全局方法：创建新会话
  createConversation() {
    const conversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.globalData.currentConversation = {
      id: conversationId,
      messages: [],
      created_at: new Date().toISOString()
    };
    return conversationId;
  },

  // 全局方法：发送消息
  sendMessage(content, modality = 'text') {
    return new Promise((resolve, reject) => {
      if (!this.globalData.token) {
        wx.showModal({
          title: '未登录',
          content: '请先登录',
          success: (res) => {
            if (res.confirm) {
              wx.navigateTo({
                url: '/pages/login/login'
              });
            }
          }
        });
        reject(new Error('Not logged in'));
        return;
      }
      
      const conversationId = this.globalData.currentConversation?.id || this.createConversation();
      
      wx.request({
        url: `${this.globalData.apiUrl}/chat/stream`,
        method: 'POST',
        header: {
          'Authorization': `Bearer ${this.globalData.token}`,
          'X-Tenant-ID': this.globalData.tenantId,
          'Content-Type': 'application/json'
        },
        data: {
          conversation_id: conversationId,
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
        enableChunked: true, // 启用流式响应
        success: (res) => {
          resolve(res);
        },
        fail: (err) => {
          console.error('发送消息失败', err);
          wx.showToast({
            title: '发送失败',
            icon: 'error'
          });
          reject(err);
        }
      });
    });
  },

  // 全局方法：开始录音
  startRecording(options = {}) {
    const recorderManager = this.globalData.recorderManager;
    
    const defaultOptions = {
      duration: 60000, // 最长60秒
      sampleRate: 16000,
      numberOfChannels: 1,
      encodeBitRate: 48000,
      format: this.globalData.audioConfig?.format || 'mp3',
      frameSize: 50, // 50KB一帧，用于实时传输
      ...options
    };
    
    recorderManager.start(defaultOptions);
  },

  // 全局方法：停止录音
  stopRecording() {
    this.globalData.recorderManager.stop();
  },

  // 全局方法：播放音频
  playAudio(url) {
    const innerAudioContext = this.globalData.innerAudioContext;
    innerAudioContext.src = url;
    innerAudioContext.play();
    
    return new Promise((resolve, reject) => {
      innerAudioContext.onEnded(() => {
        resolve();
      });
      
      innerAudioContext.onError((err) => {
        reject(err);
      });
    });
  },

  // 全局方法：停止播放
  stopAudio() {
    this.globalData.innerAudioContext.stop();
  }
});
