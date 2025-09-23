/**
 * V2架构聊天SSE客户端
 * 基于BaseStreamClient实现流式聊天功能
 */

import { BaseStreamClient, EventSink, StreamOptions } from './base';
import { 
  ChatRequest as ChatRequestType, 
  ChatStreamEvent, 
  QueryResponse,
  APIError,
  CHAT_EVENTS 
} from '@/types';

// 重新导出类型以保持向后兼容
export type ChatRequest = ChatRequestType;

export interface ChatResponse {
  event: string;
  data: any;
  meta?: {
    session_id: string;
    timestamp: number;
    trace_id: string;
    tenant_id: string;
  };
}

export class ChatSSEClient extends BaseStreamClient {
  private activeConnections = new Map<string, AbortController>();
  
  async streamChat(
    request: ChatRequest, 
    sink: EventSink, 
    options: StreamOptions = {}
  ): Promise<void> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), options.timeout || 30000);
    
    // 跟踪活跃连接
    this.activeConnections.set(request.session_id, controller);
    
    try {
      await this.retryWithBackoff(async () => {
        const response = await fetch(`${this.baseURL}/api/v2/chat/stream`, {
          method: 'POST',
          headers: {
            ...this.addTraceHeaders(),
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(request),
          signal: controller.signal,
        });
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        if (!response.body) {
          throw new Error('Response body is null');
        }
        
        await this.processSSEStream(response.body, sink, controller.signal);
      }, options.retries || 3);
      
    } catch (error) {
      if (error instanceof Error && error.name === 'AbortError') {
        sink.on('cancelled', { session_id: request.session_id });
      } else {
        sink.on('error', { error: error instanceof Error ? error.message : String(error) });
      }
    } finally {
      clearTimeout(timeoutId);
      this.activeConnections.delete(request.session_id);
    }
  }
  
  private async processSSEStream(
    body: ReadableStream<Uint8Array>, 
    sink: EventSink, 
    signal: AbortSignal
  ): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    try {
      while (true) {
        if (signal.aborted) {
          break;
        }
        
        const { done, value } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        // 处理完整的SSE事件
        const events = this.parseSSEEvents(buffer);
        buffer = events.remainder;
        
        for (const event of events.parsed) {
          this.handleSSEEvent(event, sink);
        }
      }
    } finally {
      reader.releaseLock();
    }
  }
  
  private parseSSEEvents(buffer: string): { parsed: any[], remainder: string } {
    const events: any[] = [];
    const lines = buffer.split('\n');
    let currentEvent: any = {};
    let i = 0;
    
    while (i < lines.length) {
      const line = lines[i].trim();
      
      if (line === '') {
        // 空行表示事件结束
        if (currentEvent.data) {
          events.push(currentEvent);
          currentEvent = {};
        }
      } else if (line.startsWith('event: ')) {
        currentEvent.event = line.substring(7);
      } else if (line.startsWith('data: ')) {
        const data = line.substring(6);
        try {
          currentEvent.data = JSON.parse(data);
        } catch (e) {
          currentEvent.data = data;
        }
      } else if (line.startsWith('id: ')) {
        currentEvent.id = line.substring(4);
      } else if (line.startsWith('retry: ')) {
        currentEvent.retry = parseInt(line.substring(7));
      }
      
      i++;
    }
    
    // 如果最后一个事件不完整，保留在buffer中
    let remainder = '';
    if (Object.keys(currentEvent).length > 0) {
      // 重构未完成的事件
      const lastEventStart = buffer.lastIndexOf('\n\n');
      if (lastEventStart !== -1) {
        remainder = buffer.substring(lastEventStart + 2);
      } else {
        remainder = buffer;
      }
    }
    
    return { parsed: events, remainder };
  }
  
  private handleSSEEvent(event: any, sink: EventSink): void {
    const eventType = event.event || 'message';
    const eventData = event.data;
    
    // 处理不同类型的事件
    switch (eventType) {
      case 'intent':
        sink.on('intent', eventData);
        break;
      case 'retrieve':
        sink.on('retrieve', eventData);
        break;
      case 'plan':
        sink.on('plan', eventData);
        break;
      case 'tool_result':
        sink.on('tool_result', eventData);
        break;
      case 'answer':
        sink.on('answer', eventData);
        break;
      case 'audio':
        sink.on('audio', eventData);
        break;
      case 'error':
        sink.on('error', eventData);
        break;
      case 'done':
        sink.on('completed', eventData);
        break;
      default:
        sink.on(eventType, eventData);
    }
  }
  
  async cancelChat(sessionId: string): Promise<void> {
    // 取消活跃连接
    const controller = this.activeConnections.get(sessionId);
    if (controller) {
      controller.abort();
    }
    
    // 发送取消请求到服务器
    try {
      await fetch(`${this.baseURL}/api/v2/chat/cancel`, {
        method: 'POST',
        headers: this.addTraceHeaders(),
        body: JSON.stringify({ session_id: sessionId }),
      });
    } catch (error) {
      console.warn('Failed to send cancel request:', error);
    }
  }
  
  // 获取活跃连接数
  getActiveConnectionsCount(): number {
    return this.activeConnections.size;
  }
  
  // 取消所有活跃连接
  cancelAllConnections(): void {
    for (const [sessionId, controller] of this.activeConnections) {
      controller.abort();
    }
    this.activeConnections.clear();
  }
}
