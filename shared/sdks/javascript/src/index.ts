/**
 * Chatbot JavaScript/TypeScript SDK
 * 
 * @example
 * ```typescript
 * import { ChatbotClient } from '@chatbot/sdk';
 * 
 * const client = new ChatbotClient({
 *   apiKey: 'your-api-key',
 *   baseURL: 'https://api.chatbot.ai/v1'
 * });
 * 
 * // 发送消息
 * const response = await client.chat.sendMessage('conversation-id', {
 *   content: 'Hello, world!',
 *   stream: true
 * });
 * ```
 */

export * from './client';
export * from './types';
export * from './errors';

// 默认导出
export { ChatbotClient as default } from './client';
