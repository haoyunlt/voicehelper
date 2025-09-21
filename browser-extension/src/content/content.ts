/**
 * VoiceHelper Browser Extension - Content Script
 * 网页内容分析和智能交互功能
 */

import { VoiceHelperSDK } from '@voicehelper/sdk';

// 类型定义
interface PageAnalysis {
  title: string;
  url: string;
  content: string;
  images: string[];
  links: string[];
  forms: FormData[];
  metadata: Record<string, string>;
}

interface FormData {
  action: string;
  method: string;
  fields: Array<{
    name: string;
    type: string;
    label?: string;
    required: boolean;
  }>;
}

interface VoiceHelperWidget {
  container: HTMLElement;
  isVisible: boolean;
  isRecording: boolean;
  sdk: VoiceHelperSDK | null;
}

class VoiceHelperContentScript {
  private widget: VoiceHelperWidget | null = null;
  private config: any = {};
  private isEnabled = false;
  private observer: MutationObserver | null = null;

  constructor() {
    this.init();
  }

  private async init(): Promise<void> {
    // 获取配置
    this.config = await this.getStorageData('voicehelper_config') || {};
    this.isEnabled = this.config.enabled !== false;

    if (!this.isEnabled) return;

    // 等待页面加载完成
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.setup());
    } else {
      this.setup();
    }

    // 监听来自popup的消息
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      this.handleMessage(message, sendResponse);
      return true; // 保持消息通道开放
    });
  }

  private setup(): void {
    this.createWidget();
    this.setupPageObserver();
    this.setupKeyboardShortcuts();
    this.injectStyles();
  }

  private createWidget(): void {
    // 创建悬浮widget
    const container = document.createElement('div');
    container.id = 'voicehelper-widget';
    container.className = 'voicehelper-widget';
    
    container.innerHTML = `
      <div class="voicehelper-widget-content">
        <div class="voicehelper-header">
          <div class="voicehelper-logo">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4z"/>
            </svg>
            <span>VoiceHelper</span>
          </div>
          <div class="voicehelper-controls">
            <button id="voicehelper-minimize" title="最小化">−</button>
            <button id="voicehelper-close" title="关闭">×</button>
          </div>
        </div>
        
        <div class="voicehelper-body">
          <div class="voicehelper-tabs">
            <button class="voicehelper-tab active" data-tab="chat">对话</button>
            <button class="voicehelper-tab" data-tab="analyze">分析</button>
            <button class="voicehelper-tab" data-tab="tools">工具</button>
          </div>
          
          <div class="voicehelper-tab-content">
            <!-- 对话标签页 -->
            <div id="voicehelper-chat-tab" class="voicehelper-tab-panel active">
              <div class="voicehelper-messages" id="voicehelper-messages"></div>
              <div class="voicehelper-input-area">
                <div class="voicehelper-input-container">
                  <textarea 
                    id="voicehelper-input" 
                    placeholder="输入消息或点击语音按钮..."
                    rows="2"
                  ></textarea>
                  <div class="voicehelper-input-buttons">
                    <button id="voicehelper-voice-btn" class="voicehelper-voice-btn" title="语音输入">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
                        <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                      </svg>
                    </button>
                    <button id="voicehelper-send-btn" class="voicehelper-send-btn" title="发送">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                      </svg>
                    </button>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- 分析标签页 -->
            <div id="voicehelper-analyze-tab" class="voicehelper-tab-panel">
              <div class="voicehelper-analysis-content">
                <div class="voicehelper-analysis-item">
                  <h4>页面摘要</h4>
                  <p id="voicehelper-page-summary">点击"分析页面"按钮开始分析...</p>
                </div>
                <div class="voicehelper-analysis-item">
                  <h4>关键信息</h4>
                  <ul id="voicehelper-key-info"></ul>
                </div>
                <button id="voicehelper-analyze-btn" class="voicehelper-btn-primary">
                  分析页面
                </button>
              </div>
            </div>
            
            <!-- 工具标签页 -->
            <div id="voicehelper-tools-tab" class="voicehelper-tab-panel">
              <div class="voicehelper-tools-grid">
                <button class="voicehelper-tool-btn" data-tool="translate">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12.87 15.07l-2.54-2.51.03-.03c1.74-1.94 2.98-4.17 3.71-6.53H17V4h-7V2H8v2H1v1.99h11.17C11.5 7.92 10.44 9.75 9 11.35 8.07 10.32 7.3 9.19 6.69 8h-2c.73 1.63 1.73 3.17 2.98 4.56l-5.09 5.02L4 19l5-5 3.11 3.11.76-2.04zM18.5 10h-2L12 22h2l1.12-3h4.75L21 22h2l-4.5-12zm-2.62 7l1.62-4.33L19.12 17h-3.24z"/>
                  </svg>
                  翻译
                </button>
                <button class="voicehelper-tool-btn" data-tool="summarize">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M3 13h2v-2H3v2zm0 4h2v-2H3v2zm0-8h2V7H3v2zm4 4h14v-2H7v2zm0 4h14v-2H7v2zM7 7v2h14V7H7z"/>
                  </svg>
                  摘要
                </button>
                <button class="voicehelper-tool-btn" data-tool="extract">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 2c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 2 2h10c1.1 0 2-.9 2-2V8l-6-6H6zm7 7V3.5L18.5 9H13z"/>
                  </svg>
                  提取
                </button>
                <button class="voicehelper-tool-btn" data-tool="fill-form">
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M14 2H6c-1.1 0-1.99.9-1.99 2L4 20c0 1.1.89 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm2 16H8v-2h8v2zm0-4H8v-2h8v2zm-3-5V3.5L18.5 9H13z"/>
                  </svg>
                  填表
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;

    document.body.appendChild(container);

    this.widget = {
      container,
      isVisible: false,
      isRecording: false,
      sdk: null
    };

    this.setupWidgetEvents();
    this.initializeSDK();
  }

  private setupWidgetEvents(): void {
    if (!this.widget) return;

    const container = this.widget.container;

    // 标签页切换
    container.querySelectorAll('.voicehelper-tab').forEach(tab => {
      tab.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        const tabName = target.dataset.tab;
        this.switchTab(tabName!);
      });
    });

    // 最小化/关闭按钮
    container.querySelector('#voicehelper-minimize')?.addEventListener('click', () => {
      this.toggleWidget();
    });

    container.querySelector('#voicehelper-close')?.addEventListener('click', () => {
      this.hideWidget();
    });

    // 对话功能
    const input = container.querySelector('#voicehelper-input') as HTMLTextAreaElement;
    const sendBtn = container.querySelector('#voicehelper-send-btn');
    const voiceBtn = container.querySelector('#voicehelper-voice-btn');

    sendBtn?.addEventListener('click', () => {
      this.sendMessage(input.value);
      input.value = '';
    });

    input?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage(input.value);
        input.value = '';
      }
    });

    voiceBtn?.addEventListener('click', () => {
      this.toggleVoiceRecording();
    });

    // 分析功能
    container.querySelector('#voicehelper-analyze-btn')?.addEventListener('click', () => {
      this.analyzePage();
    });

    // 工具功能
    container.querySelectorAll('.voicehelper-tool-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const target = e.target as HTMLElement;
        const tool = target.closest('[data-tool]')?.getAttribute('data-tool');
        if (tool) {
          this.executeTool(tool);
        }
      });
    });

    // 拖拽功能
    this.setupDragging();
  }

  private setupDragging(): void {
    if (!this.widget) return;

    const header = this.widget.container.querySelector('.voicehelper-header') as HTMLElement;
    let isDragging = false;
    let startX = 0;
    let startY = 0;
    let startLeft = 0;
    let startTop = 0;

    header.addEventListener('mousedown', (e) => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      const rect = this.widget!.container.getBoundingClientRect();
      startLeft = rect.left;
      startTop = rect.top;
      
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    });

    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      
      this.widget!.container.style.left = `${startLeft + deltaX}px`;
      this.widget!.container.style.top = `${startTop + deltaY}px`;
    };

    const onMouseUp = () => {
      isDragging = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }

  private async initializeSDK(): Promise<void> {
    if (!this.widget) return;

    const apiKey = await this.getStorageData('voicehelper_api_key');
    if (apiKey) {
      this.widget.sdk = new VoiceHelperSDK({ apiKey });
    }
  }

  private switchTab(tabName: string): void {
    if (!this.widget) return;

    // 更新标签按钮状态
    this.widget.container.querySelectorAll('.voicehelper-tab').forEach(tab => {
      tab.classList.remove('active');
    });
    this.widget.container.querySelector(`[data-tab="${tabName}"]`)?.classList.add('active');

    // 更新标签页内容
    this.widget.container.querySelectorAll('.voicehelper-tab-panel').forEach(panel => {
      panel.classList.remove('active');
    });
    this.widget.container.querySelector(`#voicehelper-${tabName}-tab`)?.classList.add('active');
  }

  private async sendMessage(message: string): Promise<void> {
    if (!message.trim() || !this.widget?.sdk) return;

    const messagesContainer = this.widget.container.querySelector('#voicehelper-messages');
    if (!messagesContainer) return;

    // 添加用户消息
    this.addMessage(message, 'user');

    try {
      // 获取页面上下文
      const pageContext = this.getPageContext();
      
      // 发送到API
      const response = await this.widget.sdk.createChatCompletion({
        messages: [
          {
            role: 'system',
            content: `你是VoiceHelper AI助手。当前页面信息：标题"${pageContext.title}"，URL"${pageContext.url}"。请基于页面内容提供帮助。`
          },
          {
            role: 'user',
            content: message
          }
        ],
        model: 'gpt-4'
      });

      // 添加AI回复
      const reply = response.choices[0]?.message?.content || '抱歉，我无法处理您的请求。';
      this.addMessage(reply, 'assistant');

    } catch (error) {
      this.addMessage('抱歉，发生了错误。请检查您的API配置。', 'assistant');
    }
  }

  private addMessage(content: string, role: 'user' | 'assistant'): void {
    if (!this.widget) return;

    const messagesContainer = this.widget.container.querySelector('#voicehelper-messages');
    if (!messagesContainer) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `voicehelper-message voicehelper-message-${role}`;
    
    const time = new Date().toLocaleTimeString('zh-CN', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });

    messageDiv.innerHTML = `
      <div class="voicehelper-message-content">
        <div class="voicehelper-message-text">${content}</div>
        <div class="voicehelper-message-time">${time}</div>
      </div>
    `;

    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  private toggleVoiceRecording(): void {
    if (!this.widget) return;

    const voiceBtn = this.widget.container.querySelector('#voicehelper-voice-btn');
    
    if (this.widget.isRecording) {
      // 停止录音
      this.widget.isRecording = false;
      voiceBtn?.classList.remove('recording');
      // 这里应该调用语音识别API
    } else {
      // 开始录音
      this.widget.isRecording = true;
      voiceBtn?.classList.add('recording');
      // 这里应该开始录音
    }
  }

  private async analyzePage(): Promise<void> {
    if (!this.widget?.sdk) return;

    const analyzeBtn = this.widget.container.querySelector('#voicehelper-analyze-btn') as HTMLButtonElement;
    const summaryEl = this.widget.container.querySelector('#voicehelper-page-summary');
    const keyInfoEl = this.widget.container.querySelector('#voicehelper-key-info');

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '分析中...';

    try {
      const pageAnalysis = this.analyzePageContent();
      
      // 使用AI分析页面内容
      const response = await this.widget.sdk.createChatCompletion({
        messages: [
          {
            role: 'system',
            content: '你是一个网页内容分析专家。请分析以下网页内容，提供简洁的摘要和关键信息。'
          },
          {
            role: 'user',
            content: `请分析这个网页：\n标题：${pageAnalysis.title}\nURL：${pageAnalysis.url}\n内容：${pageAnalysis.content.substring(0, 2000)}...`
          }
        ],
        model: 'gpt-4'
      });

      const analysis = response.choices[0]?.message?.content || '分析失败';
      
      if (summaryEl) {
        summaryEl.textContent = analysis;
      }

      if (keyInfoEl) {
        keyInfoEl.innerHTML = '';
        pageAnalysis.links.slice(0, 5).forEach(link => {
          const li = document.createElement('li');
          li.textContent = link;
          keyInfoEl.appendChild(li);
        });
      }

    } catch (error) {
      if (summaryEl) {
        summaryEl.textContent = '分析失败，请检查API配置。';
      }
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = '分析页面';
    }
  }

  private analyzePageContent(): PageAnalysis {
    return {
      title: document.title,
      url: window.location.href,
      content: document.body.innerText || '',
      images: Array.from(document.images).map(img => img.src),
      links: Array.from(document.links).map(link => link.href),
      forms: Array.from(document.forms).map(form => ({
        action: form.action,
        method: form.method,
        fields: Array.from(form.elements).map(element => {
          const input = element as HTMLInputElement;
          return {
            name: input.name,
            type: input.type,
            label: input.labels?.[0]?.textContent || '',
            required: input.required
          };
        })
      })),
      metadata: this.extractMetadata()
    };
  }

  private extractMetadata(): Record<string, string> {
    const metadata: Record<string, string> = {};
    
    document.querySelectorAll('meta').forEach(meta => {
      const name = meta.getAttribute('name') || meta.getAttribute('property');
      const content = meta.getAttribute('content');
      if (name && content) {
        metadata[name] = content;
      }
    });

    return metadata;
  }

  private async executeTool(tool: string): Promise<void> {
    if (!this.widget?.sdk) return;

    switch (tool) {
      case 'translate':
        await this.translatePage();
        break;
      case 'summarize':
        await this.summarizePage();
        break;
      case 'extract':
        await this.extractKeyInfo();
        break;
      case 'fill-form':
        await this.fillForm();
        break;
    }
  }

  private async translatePage(): Promise<void> {
    const selectedText = window.getSelection()?.toString();
    const textToTranslate = selectedText || document.body.innerText.substring(0, 1000);

    if (!textToTranslate) return;

    try {
      const response = await this.widget!.sdk!.createChatCompletion({
        messages: [
          {
            role: 'system',
            content: '请将以下文本翻译成中文，保持原意和格式。'
          },
          {
            role: 'user',
            content: textToTranslate
          }
        ],
        model: 'gpt-4'
      });

      const translation = response.choices[0]?.message?.content || '翻译失败';
      this.showTooltip(translation);

    } catch (error) {
      this.showTooltip('翻译失败，请检查API配置。');
    }
  }

  private async summarizePage(): Promise<void> {
    const pageContent = document.body.innerText.substring(0, 3000);

    try {
      const response = await this.widget!.sdk!.createChatCompletion({
        messages: [
          {
            role: 'system',
            content: '请为以下网页内容生成简洁的摘要，突出关键信息。'
          },
          {
            role: 'user',
            content: `网页标题：${document.title}\n内容：${pageContent}`
          }
        ],
        model: 'gpt-4'
      });

      const summary = response.choices[0]?.message?.content || '摘要生成失败';
      this.showTooltip(summary);

    } catch (error) {
      this.showTooltip('摘要生成失败，请检查API配置。');
    }
  }

  private async extractKeyInfo(): Promise<void> {
    const pageAnalysis = this.analyzePageContent();

    try {
      const response = await this.widget!.sdk!.createChatCompletion({
        messages: [
          {
            role: 'system',
            content: '请从以下网页内容中提取关键信息，如联系方式、价格、日期等重要数据。'
          },
          {
            role: 'user',
            content: `网页：${pageAnalysis.title}\n内容：${pageAnalysis.content.substring(0, 2000)}`
          }
        ],
        model: 'gpt-4'
      });

      const keyInfo = response.choices[0]?.message?.content || '信息提取失败';
      this.showTooltip(keyInfo);

    } catch (error) {
      this.showTooltip('信息提取失败，请检查API配置。');
    }
  }

  private async fillForm(): Promise<void> {
    const forms = document.forms;
    if (forms.length === 0) {
      this.showTooltip('页面中没有找到表单。');
      return;
    }

    // 这里可以实现智能表单填写功能
    this.showTooltip('智能表单填写功能开发中...');
  }

  private showTooltip(content: string): void {
    const tooltip = document.createElement('div');
    tooltip.className = 'voicehelper-tooltip';
    tooltip.innerHTML = `
      <div class="voicehelper-tooltip-content">
        <div class="voicehelper-tooltip-text">${content}</div>
        <button class="voicehelper-tooltip-close">×</button>
      </div>
    `;

    document.body.appendChild(tooltip);

    // 自动关闭
    setTimeout(() => {
      tooltip.remove();
    }, 10000);

    // 手动关闭
    tooltip.querySelector('.voicehelper-tooltip-close')?.addEventListener('click', () => {
      tooltip.remove();
    });
  }

  private getPageContext(): { title: string; url: string } {
    return {
      title: document.title,
      url: window.location.href
    };
  }

  private setupPageObserver(): void {
    this.observer = new MutationObserver((mutations) => {
      // 监听页面变化，可以用于动态分析
    });

    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: false
    });
  }

  private setupKeyboardShortcuts(): void {
    document.addEventListener('keydown', (e) => {
      // Ctrl/Cmd + Shift + V - 切换widget显示
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'V') {
        e.preventDefault();
        this.toggleWidget();
      }

      // Ctrl/Cmd + Shift + S - 开始语音
      if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'S') {
        e.preventDefault();
        this.toggleVoiceRecording();
      }
    });
  }

  private injectStyles(): void {
    const style = document.createElement('style');
    style.textContent = `
      .voicehelper-widget {
        position: fixed;
        top: 20px;
        right: 20px;
        width: 350px;
        height: 500px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        z-index: 10000;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 14px;
        display: none;
        flex-direction: column;
        overflow: hidden;
      }

      .voicehelper-widget.visible {
        display: flex;
      }

      .voicehelper-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        background: #2196F3;
        color: white;
        cursor: move;
      }

      .voicehelper-logo {
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 600;
      }

      .voicehelper-controls button {
        background: none;
        border: none;
        color: white;
        cursor: pointer;
        padding: 4px 8px;
        margin-left: 4px;
        border-radius: 4px;
        font-size: 16px;
      }

      .voicehelper-controls button:hover {
        background: rgba(255, 255, 255, 0.2);
      }

      .voicehelper-body {
        flex: 1;
        display: flex;
        flex-direction: column;
      }

      .voicehelper-tabs {
        display: flex;
        border-bottom: 1px solid #e0e0e0;
      }

      .voicehelper-tab {
        flex: 1;
        padding: 12px;
        background: none;
        border: none;
        cursor: pointer;
        font-size: 14px;
        color: #666;
        border-bottom: 2px solid transparent;
      }

      .voicehelper-tab.active {
        color: #2196F3;
        border-bottom-color: #2196F3;
      }

      .voicehelper-tab-content {
        flex: 1;
        position: relative;
      }

      .voicehelper-tab-panel {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        display: none;
        flex-direction: column;
      }

      .voicehelper-tab-panel.active {
        display: flex;
      }

      .voicehelper-messages {
        flex: 1;
        padding: 16px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 12px;
      }

      .voicehelper-message {
        display: flex;
      }

      .voicehelper-message-user {
        justify-content: flex-end;
      }

      .voicehelper-message-content {
        max-width: 80%;
        padding: 8px 12px;
        border-radius: 12px;
        background: #f5f5f5;
      }

      .voicehelper-message-user .voicehelper-message-content {
        background: #2196F3;
        color: white;
      }

      .voicehelper-message-time {
        font-size: 11px;
        color: #999;
        margin-top: 4px;
      }

      .voicehelper-input-area {
        padding: 16px;
        border-top: 1px solid #e0e0e0;
      }

      .voicehelper-input-container {
        display: flex;
        gap: 8px;
        align-items: flex-end;
      }

      .voicehelper-input-container textarea {
        flex: 1;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 8px 12px;
        resize: none;
        font-family: inherit;
        font-size: 14px;
      }

      .voicehelper-input-buttons {
        display: flex;
        gap: 4px;
      }

      .voicehelper-voice-btn,
      .voicehelper-send-btn {
        width: 36px;
        height: 36px;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #f5f5f5;
        color: #666;
      }

      .voicehelper-voice-btn:hover,
      .voicehelper-send-btn:hover {
        background: #e0e0e0;
      }

      .voicehelper-voice-btn.recording {
        background: #f44336;
        color: white;
      }

      .voicehelper-analysis-content {
        padding: 16px;
        height: 100%;
        overflow-y: auto;
      }

      .voicehelper-analysis-item {
        margin-bottom: 20px;
      }

      .voicehelper-analysis-item h4 {
        margin: 0 0 8px 0;
        color: #333;
        font-size: 14px;
        font-weight: 600;
      }

      .voicehelper-analysis-item p {
        margin: 0;
        color: #666;
        line-height: 1.5;
      }

      .voicehelper-btn-primary {
        width: 100%;
        padding: 12px;
        background: #2196F3;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
      }

      .voicehelper-btn-primary:hover {
        background: #1976D2;
      }

      .voicehelper-tools-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        padding: 16px;
      }

      .voicehelper-tool-btn {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
        padding: 20px;
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        cursor: pointer;
        font-size: 12px;
        color: #495057;
      }

      .voicehelper-tool-btn:hover {
        background: #e9ecef;
      }

      .voicehelper-tooltip {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: white;
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        z-index: 10001;
        max-width: 400px;
        max-height: 300px;
      }

      .voicehelper-tooltip-content {
        padding: 20px;
        position: relative;
      }

      .voicehelper-tooltip-text {
        line-height: 1.5;
        color: #333;
        margin-bottom: 12px;
      }

      .voicehelper-tooltip-close {
        position: absolute;
        top: 8px;
        right: 8px;
        background: none;
        border: none;
        font-size: 18px;
        cursor: pointer;
        color: #999;
      }
    `;

    document.head.appendChild(style);
  }

  private toggleWidget(): void {
    if (!this.widget) return;

    if (this.widget.isVisible) {
      this.hideWidget();
    } else {
      this.showWidget();
    }
  }

  private showWidget(): void {
    if (!this.widget) return;

    this.widget.container.classList.add('visible');
    this.widget.isVisible = true;
  }

  private hideWidget(): void {
    if (!this.widget) return;

    this.widget.container.classList.remove('visible');
    this.widget.isVisible = false;
  }

  private async handleMessage(message: any, sendResponse: (response: any) => void): Promise<void> {
    switch (message.action) {
      case 'toggle-widget':
        this.toggleWidget();
        sendResponse({ success: true });
        break;

      case 'analyze-page':
        const analysis = this.analyzePageContent();
        sendResponse({ success: true, data: analysis });
        break;

      case 'get-selected-text':
        const selectedText = window.getSelection()?.toString() || '';
        sendResponse({ success: true, data: selectedText });
        break;

      case 'update-config':
        this.config = { ...this.config, ...message.config };
        await this.setStorageData('voicehelper_config', this.config);
        if (message.config.apiKey) {
          await this.setStorageData('voicehelper_api_key', message.config.apiKey);
          await this.initializeSDK();
        }
        sendResponse({ success: true });
        break;

      default:
        sendResponse({ success: false, error: 'Unknown action' });
    }
  }

  private async getStorageData(key: string): Promise<any> {
    return new Promise((resolve) => {
      chrome.storage.sync.get([key], (result) => {
        resolve(result[key]);
      });
    });
  }

  private async setStorageData(key: string, value: any): Promise<void> {
    return new Promise((resolve) => {
      chrome.storage.sync.set({ [key]: value }, () => {
        resolve();
      });
    });
  }
}

// 初始化内容脚本
new VoiceHelperContentScript();
