// Content script for AI Chatbot browser extension
class ChatbotWidget {
  constructor() {
    this.isVisible = false;
    this.widget = null;
    this.messages = [];
    this.isLoading = false;
    this.apiEndpoint = 'http://localhost:8080/api/v1';
    
    this.init();
  }

  async init() {
    // Check if widget should be enabled on this site
    const settings = await this.getSettings();
    if (settings.disabledSites && settings.disabledSites.includes(window.location.hostname)) {
      return;
    }

    this.createWidget();
    this.setupEventListeners();
    this.setupContextMenu();
    
    // Auto-show widget if enabled
    if (settings.autoShow) {
      setTimeout(() => this.showWidget(), 2000);
    }
  }

  async getSettings() {
    return new Promise((resolve) => {
      chrome.storage.sync.get({
        autoShow: false,
        disabledSites: [],
        apiEndpoint: 'http://localhost:8080/api/v1',
        theme: 'light',
        position: 'bottom-right'
      }, resolve);
    });
  }

  createWidget() {
    // Create widget container
    this.widget = document.createElement('div');
    this.widget.id = 'chatbot-widget';
    this.widget.className = 'chatbot-widget chatbot-hidden';
    
    this.widget.innerHTML = `
      <div class="chatbot-header">
        <div class="chatbot-title">
          <img src="${chrome.runtime.getURL('icons/icon-32.png')}" alt="AI" class="chatbot-icon">
          <span>AI Assistant</span>
        </div>
        <div class="chatbot-controls">
          <button class="chatbot-minimize" title="Minimize">−</button>
          <button class="chatbot-close" title="Close">×</button>
        </div>
      </div>
      
      <div class="chatbot-messages" id="chatbot-messages">
        <div class="chatbot-message chatbot-assistant">
          <div class="chatbot-avatar">🤖</div>
          <div class="chatbot-content">
            <p>Hello! I'm your AI assistant. How can I help you today?</p>
          </div>
        </div>
      </div>
      
      <div class="chatbot-input-container">
        <div class="chatbot-input-wrapper">
          <textarea 
            id="chatbot-input" 
            placeholder="Type your message..." 
            rows="1"
            maxlength="1000"
          ></textarea>
          <button id="chatbot-send" class="chatbot-send-btn" disabled>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
        <div class="chatbot-actions">
          <button class="chatbot-action-btn" id="chatbot-voice" title="Voice input">
            🎤
          </button>
          <button class="chatbot-action-btn" id="chatbot-attach" title="Attach file">
            📎
          </button>
          <button class="chatbot-action-btn" id="chatbot-clear" title="Clear chat">
            🗑️
          </button>
        </div>
      </div>
      
      <div class="chatbot-status" id="chatbot-status"></div>
    `;

    // Add CSS
    this.addStyles();
    
    // Append to body
    document.body.appendChild(this.widget);
    
    // Setup widget interactions
    this.setupWidgetEvents();
  }

  addStyles() {
    if (document.getElementById('chatbot-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'chatbot-styles';
    style.textContent = `
      .chatbot-widget {
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 350px;
        height: 500px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
        z-index: 10000;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        display: flex;
        flex-direction: column;
        transition: all 0.3s ease;
        border: 1px solid #e0e0e0;
      }
      
      .chatbot-hidden {
        transform: translateY(100%) scale(0.8);
        opacity: 0;
        pointer-events: none;
      }
      
      .chatbot-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        background: #007AFF;
        color: white;
        border-radius: 12px 12px 0 0;
        cursor: move;
      }
      
      .chatbot-title {
        display: flex;
        align-items: center;
        font-weight: 600;
        font-size: 14px;
      }
      
      .chatbot-icon {
        width: 20px;
        height: 20px;
        margin-right: 8px;
      }
      
      .chatbot-controls {
        display: flex;
        gap: 8px;
      }
      
      .chatbot-minimize,
      .chatbot-close {
        background: none;
        border: none;
        color: white;
        font-size: 16px;
        cursor: pointer;
        padding: 4px 8px;
        border-radius: 4px;
        transition: background 0.2s;
      }
      
      .chatbot-minimize:hover,
      .chatbot-close:hover {
        background: rgba(255, 255, 255, 0.2);
      }
      
      .chatbot-messages {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      
      .chatbot-message {
        display: flex;
        gap: 8px;
        max-width: 85%;
      }
      
      .chatbot-user {
        align-self: flex-end;
        flex-direction: row-reverse;
      }
      
      .chatbot-assistant {
        align-self: flex-start;
      }
      
      .chatbot-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        flex-shrink: 0;
      }
      
      .chatbot-user .chatbot-avatar {
        background: #007AFF;
        color: white;
      }
      
      .chatbot-assistant .chatbot-avatar {
        background: #f0f0f0;
      }
      
      .chatbot-content {
        background: #f8f9fa;
        padding: 8px 12px;
        border-radius: 12px;
        font-size: 14px;
        line-height: 1.4;
      }
      
      .chatbot-user .chatbot-content {
        background: #007AFF;
        color: white;
      }
      
      .chatbot-input-container {
        padding: 12px 16px;
        border-top: 1px solid #e0e0e0;
      }
      
      .chatbot-input-wrapper {
        display: flex;
        gap: 8px;
        align-items: flex-end;
        margin-bottom: 8px;
      }
      
      #chatbot-input {
        flex: 1;
        border: 1px solid #ddd;
        border-radius: 20px;
        padding: 8px 12px;
        font-size: 14px;
        resize: none;
        outline: none;
        transition: border-color 0.2s;
        max-height: 80px;
      }
      
      #chatbot-input:focus {
        border-color: #007AFF;
      }
      
      .chatbot-send-btn {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border: none;
        background: #007AFF;
        color: white;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.2s;
      }
      
      .chatbot-send-btn:disabled {
        background: #ccc;
        cursor: not-allowed;
      }
      
      .chatbot-send-btn:not(:disabled):hover {
        background: #0056b3;
        transform: scale(1.05);
      }
      
      .chatbot-actions {
        display: flex;
        gap: 8px;
        justify-content: center;
      }
      
      .chatbot-action-btn {
        background: none;
        border: 1px solid #ddd;
        border-radius: 6px;
        padding: 6px 10px;
        cursor: pointer;
        font-size: 12px;
        transition: all 0.2s;
      }
      
      .chatbot-action-btn:hover {
        background: #f0f0f0;
        border-color: #007AFF;
      }
      
      .chatbot-status {
        padding: 8px 16px;
        font-size: 12px;
        color: #666;
        text-align: center;
        border-top: 1px solid #e0e0e0;
        background: #f8f9fa;
        border-radius: 0 0 12px 12px;
      }
      
      .chatbot-typing {
        display: flex;
        align-items: center;
        gap: 4px;
        font-style: italic;
        color: #666;
      }
      
      .chatbot-typing::after {
        content: '...';
        animation: chatbot-dots 1.5s infinite;
      }
      
      @keyframes chatbot-dots {
        0%, 20% { content: '.'; }
        40% { content: '..'; }
        60%, 100% { content: '...'; }
      }
      
      .chatbot-minimized {
        height: 60px;
      }
      
      .chatbot-minimized .chatbot-messages,
      .chatbot-minimized .chatbot-input-container,
      .chatbot-minimized .chatbot-status {
        display: none;
      }
      
      @media (max-width: 480px) {
        .chatbot-widget {
          width: calc(100vw - 20px);
          height: calc(100vh - 40px);
          bottom: 10px;
          right: 10px;
        }
      }
    `;
    
    document.head.appendChild(style);
  }

  setupWidgetEvents() {
    const input = this.widget.querySelector('#chatbot-input');
    const sendBtn = this.widget.querySelector('#chatbot-send');
    const closeBtn = this.widget.querySelector('.chatbot-close');
    const minimizeBtn = this.widget.querySelector('.chatbot-minimize');
    const clearBtn = this.widget.querySelector('#chatbot-clear');
    const voiceBtn = this.widget.querySelector('#chatbot-voice');
    
    // Input events
    input.addEventListener('input', () => {
      sendBtn.disabled = !input.value.trim();
      this.autoResize(input);
    });
    
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        if (input.value.trim()) {
          this.sendMessage(input.value.trim());
        }
      }
    });
    
    // Send button
    sendBtn.addEventListener('click', () => {
      if (input.value.trim()) {
        this.sendMessage(input.value.trim());
      }
    });
    
    // Control buttons
    closeBtn.addEventListener('click', () => this.hideWidget());
    minimizeBtn.addEventListener('click', () => this.toggleMinimize());
    clearBtn.addEventListener('click', () => this.clearChat());
    voiceBtn.addEventListener('click', () => this.toggleVoiceInput());
    
    // Make widget draggable
    this.makeDraggable();
  }

  setupEventListeners() {
    // Listen for messages from background script
    chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
      switch (message.action) {
        case 'toggle-widget':
          this.toggleWidget();
          break;
        case 'quick-ask':
          this.quickAsk(message.text);
          break;
        case 'show-widget':
          this.showWidget();
          break;
        case 'hide-widget':
          this.hideWidget();
          break;
      }
    });
    
    // Listen for keyboard shortcuts
    document.addEventListener('keydown', (e) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        e.preventDefault();
        this.toggleWidget();
      }
    });
  }

  setupContextMenu() {
    document.addEventListener('contextmenu', (e) => {
      const selectedText = window.getSelection().toString().trim();
      if (selectedText) {
        // Store selected text for potential quick ask
        this.selectedText = selectedText;
      }
    });
  }

  makeDraggable() {
    const header = this.widget.querySelector('.chatbot-header');
    let isDragging = false;
    let startX, startY, startLeft, startTop;
    
    header.addEventListener('mousedown', (e) => {
      isDragging = true;
      startX = e.clientX;
      startY = e.clientY;
      startLeft = this.widget.offsetLeft;
      startTop = this.widget.offsetTop;
      
      document.addEventListener('mousemove', onMouseMove);
      document.addEventListener('mouseup', onMouseUp);
    });
    
    const onMouseMove = (e) => {
      if (!isDragging) return;
      
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      
      let newLeft = startLeft + deltaX;
      let newTop = startTop + deltaY;
      
      // Keep widget within viewport
      newLeft = Math.max(0, Math.min(newLeft, window.innerWidth - this.widget.offsetWidth));
      newTop = Math.max(0, Math.min(newTop, window.innerHeight - this.widget.offsetHeight));
      
      this.widget.style.left = newLeft + 'px';
      this.widget.style.top = newTop + 'px';
      this.widget.style.right = 'auto';
      this.widget.style.bottom = 'auto';
    };
    
    const onMouseUp = () => {
      isDragging = false;
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
  }

  autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 80) + 'px';
  }

  showWidget() {
    this.isVisible = true;
    this.widget.classList.remove('chatbot-hidden');
    this.widget.querySelector('#chatbot-input').focus();
  }

  hideWidget() {
    this.isVisible = false;
    this.widget.classList.add('chatbot-hidden');
  }

  toggleWidget() {
    if (this.isVisible) {
      this.hideWidget();
    } else {
      this.showWidget();
    }
  }

  toggleMinimize() {
    this.widget.classList.toggle('chatbot-minimized');
  }

  async sendMessage(text) {
    if (this.isLoading) return;
    
    const input = this.widget.querySelector('#chatbot-input');
    const messagesContainer = this.widget.querySelector('#chatbot-messages');
    
    // Add user message
    this.addMessage(text, 'user');
    
    // Clear input
    input.value = '';
    input.style.height = 'auto';
    this.widget.querySelector('#chatbot-send').disabled = true;
    
    // Show typing indicator
    this.showTyping();
    this.isLoading = true;
    
    try {
      const settings = await this.getSettings();
      const response = await fetch(`${settings.apiEndpoint}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: text,
          conversationId: 'browser-extension',
          stream: false,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      // Remove typing indicator
      this.hideTyping();
      
      // Add assistant response
      this.addMessage(data.message || 'Sorry, I could not process your request.', 'assistant');
      
    } catch (error) {
      console.error('Chat error:', error);
      this.hideTyping();
      this.addMessage('Sorry, I encountered an error. Please try again later.', 'assistant');
      
      // Show error in status
      this.showStatus('Connection error. Check your internet connection.', 'error');
    } finally {
      this.isLoading = false;
    }
  }

  addMessage(text, sender) {
    const messagesContainer = this.widget.querySelector('#chatbot-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chatbot-message chatbot-${sender}`;
    
    const avatar = sender === 'user' ? '👤' : '🤖';
    
    messageDiv.innerHTML = `
      <div class="chatbot-avatar">${avatar}</div>
      <div class="chatbot-content">
        <p>${this.formatMessage(text)}</p>
      </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Store message
    this.messages.push({ text, sender, timestamp: Date.now() });
  }

  formatMessage(text) {
    // Basic markdown-like formatting
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  }

  showTyping() {
    const messagesContainer = this.widget.querySelector('#chatbot-messages');
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chatbot-message chatbot-assistant';
    typingDiv.id = 'chatbot-typing';
    
    typingDiv.innerHTML = `
      <div class="chatbot-avatar">🤖</div>
      <div class="chatbot-content">
        <div class="chatbot-typing">Typing</div>
      </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  }

  hideTyping() {
    const typingDiv = this.widget.querySelector('#chatbot-typing');
    if (typingDiv) {
      typingDiv.remove();
    }
  }

  clearChat() {
    if (confirm('Are you sure you want to clear the chat history?')) {
      const messagesContainer = this.widget.querySelector('#chatbot-messages');
      messagesContainer.innerHTML = `
        <div class="chatbot-message chatbot-assistant">
          <div class="chatbot-avatar">🤖</div>
          <div class="chatbot-content">
            <p>Hello! I'm your AI assistant. How can I help you today?</p>
          </div>
        </div>
      `;
      this.messages = [];
    }
  }

  async quickAsk(text) {
    this.showWidget();
    await new Promise(resolve => setTimeout(resolve, 300)); // Wait for animation
    this.sendMessage(`Please help me with: "${text}"`);
  }

  toggleVoiceInput() {
    // Voice input functionality would be implemented here
    this.showStatus('Voice input not yet implemented', 'info');
  }

  showStatus(message, type = 'info') {
    const statusDiv = this.widget.querySelector('#chatbot-status');
    statusDiv.textContent = message;
    statusDiv.className = `chatbot-status chatbot-status-${type}`;
    
    setTimeout(() => {
      statusDiv.textContent = '';
      statusDiv.className = 'chatbot-status';
    }, 3000);
  }
}

// Initialize widget when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new ChatbotWidget();
  });
} else {
  new ChatbotWidget();
}
