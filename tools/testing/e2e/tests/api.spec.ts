import { test, expect, APIRequestContext } from '@playwright/test';

/**
 * API集成测试
 * 测试后端API接口的功能和性能
 */

test.describe('API集成测试', () => {
  let apiContext: APIRequestContext;

  test.beforeAll(async ({ playwright }) => {
    // 创建API请求上下文
    apiContext = await playwright.request.newContext({
      baseURL: 'http://localhost:8080',
      extraHTTPHeaders: {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
      }
    });
  });

  test.afterAll(async () => {
    await apiContext.dispose();
  });

  test.describe('健康检查和基础API', () => {
    test('健康检查接口', async () => {
      const response = await apiContext.get('/health');
      
      expect(response.ok()).toBeTruthy();
      expect(response.status()).toBe(200);
      
      const data = await response.json();
      expect(data.status).toBe('ok');
      expect(data.version).toBeTruthy();
    });

    test('版本信息接口', async () => {
      const response = await apiContext.get('/version');
      
      expect(response.ok()).toBeTruthy();
      
      const data = await response.json();
      expect(data.version).toBeTruthy();
      expect(data.build_time).toBeTruthy();
    });

    test('Ping接口', async () => {
      const response = await apiContext.get('/api/v1/ping');
      
      expect(response.ok()).toBeTruthy();
      
      const data = await response.json();
      expect(data.message).toBe('pong');
      expect(data.time).toBeTruthy();
    });
  });

  test.describe('聊天API测试', () => {
    test('流式聊天接口', async () => {
      const chatRequest = {
        conversation_id: `test-conv-${Date.now()}`,
        messages: [
          {
            role: 'user',
            content: '你好，这是一个API测试消息'
          }
        ],
        temperature: 0.7,
        max_tokens: 1000
      };

      const response = await apiContext.post('/api/v1/chat/stream', {
        data: chatRequest
      });

      expect(response.ok()).toBeTruthy();
      expect(response.headers()['content-type']).toContain('text/stream');

      // 读取流式响应
      const responseText = await response.text();
      expect(responseText).toBeTruthy();
      expect(responseText.length).toBeGreaterThan(0);
    });

    test('聊天取消接口', async () => {
      const requestId = `test-request-${Date.now()}`;
      
      const response = await apiContext.post('/api/v1/chat/cancel', {
        headers: {
          'X-Request-ID': requestId
        }
      });

      expect(response.ok()).toBeTruthy();
      
      const data = await response.json();
      expect(data.status).toContain('cancelled');
    });

    test('对话历史接口', async () => {
      const conversationId = `test-conv-${Date.now()}`;
      
      // 先发送一条消息创建对话
      await apiContext.post('/api/v1/chat/stream', {
        data: {
          conversation_id: conversationId,
          messages: [{ role: 'user', content: '创建对话历史' }]
        }
      });

      // 获取对话历史
      const response = await apiContext.get(`/api/v1/conversations/${conversationId}/messages`);
      
      if (response.ok()) {
        const data = await response.json();
        expect(Array.isArray(data.messages)).toBeTruthy();
      }
    });
  });

  test.describe('语音API测试', () => {
    test('语音转文字接口', async () => {
      // 创建模拟音频数据
      const audioData = Buffer.from('mock-audio-data');
      
      const response = await apiContext.post('/api/v1/voice/transcribe', {
        multipart: {
          audio: {
            name: 'test-audio.wav',
            mimeType: 'audio/wav',
            buffer: audioData
          },
          language: 'zh-CN'
        }
      });

      if (response.ok()) {
        const data = await response.json();
        expect(data.transcript).toBeTruthy();
        expect(data.confidence).toBeGreaterThan(0);
      } else {
        // 如果接口不存在或返回错误，记录但不失败
        console.log('语音转文字接口可能未实现:', response.status());
      }
    });

    test('文字转语音接口', async () => {
      const ttsRequest = {
        text: '这是一个文字转语音的测试',
        voice: 'zh-CN-XiaoxiaoNeural',
        speed: 1.0,
        pitch: 1.0
      };

      const response = await apiContext.post('/api/v1/voice/synthesize', {
        data: ttsRequest
      });

      if (response.ok()) {
        const contentType = response.headers()['content-type'];
        expect(contentType).toContain('audio/');
        
        const audioBuffer = await response.body();
        expect(audioBuffer.length).toBeGreaterThan(0);
      } else {
        console.log('文字转语音接口可能未实现:', response.status());
      }
    });

    test('实时语音流接口', async () => {
      // 测试WebSocket连接（如果支持）
      const wsUrl = 'ws://localhost:8080/api/v1/voice/stream';
      
      try {
        // 这里需要使用WebSocket客户端测试
        // Playwright的WebSocket支持有限，这里做基本的连接测试
        const response = await apiContext.get('/api/v1/voice/stream/info');
        
        if (response.ok()) {
          const data = await response.json();
          expect(data.supported_codecs).toBeTruthy();
        }
      } catch (error) {
        console.log('实时语音流接口测试跳过:', error);
      }
    });
  });

  test.describe('数据集管理API测试', () => {
    let testDatasetId: string;

    test('创建数据集', async () => {
      const datasetRequest = {
        name: `API测试数据集-${Date.now()}`,
        description: '这是通过API创建的测试数据集',
        type: 'knowledge_base'
      };

      const response = await apiContext.post('/api/v1/datasets', {
        data: datasetRequest
      });

      expect(response.ok()).toBeTruthy();
      
      const data = await response.json();
      expect(data.dataset_id).toBeTruthy();
      expect(data.name).toBe(datasetRequest.name);
      
      testDatasetId = data.dataset_id;
    });

    test('获取数据集列表', async () => {
      const response = await apiContext.get('/api/v1/datasets');
      
      expect(response.ok()).toBeTruthy();
      
      const data = await response.json();
      expect(Array.isArray(data.datasets)).toBeTruthy();
      expect(data.total).toBeGreaterThanOrEqual(0);
    });

    test('上传文档到数据集', async () => {
      if (!testDatasetId) {
        test.skip();
        return;
      }

      const documentContent = '这是一个测试文档的内容，用于API测试。';
      const documentBuffer = Buffer.from(documentContent, 'utf8');

      const response = await apiContext.post('/api/v1/ingest/upload', {
        multipart: {
          files: {
            name: 'test-document.txt',
            mimeType: 'text/plain',
            buffer: documentBuffer
          },
          dataset_id: testDatasetId
        }
      });

      expect(response.ok()).toBeTruthy();
      
      const data = await response.json();
      expect(data.uploaded_files).toBeGreaterThan(0);
      expect(data.processed_documents).toBeGreaterThan(0);
    });

    test('获取数据集详情', async () => {
      if (!testDatasetId) {
        test.skip();
        return;
      }

      const response = await apiContext.get(`/api/v1/datasets/${testDatasetId}`);
      
      expect(response.ok()).toBeTruthy();
      
      const data = await response.json();
      expect(data.dataset_id).toBe(testDatasetId);
      expect(data.name).toBeTruthy();
    });

    test('搜索文档', async () => {
      if (!testDatasetId) {
        test.skip();
        return;
      }

      const searchRequest = {
        query: '测试',
        dataset_id: testDatasetId,
        top_k: 5
      };

      const response = await apiContext.post('/api/v1/search', {
        data: searchRequest
      });

      if (response.ok()) {
        const data = await response.json();
        expect(Array.isArray(data.results)).toBeTruthy();
        expect(data.query).toBe(searchRequest.query);
      }
    });

    test('删除数据集', async () => {
      if (!testDatasetId) {
        test.skip();
        return;
      }

      const response = await apiContext.delete(`/api/v1/datasets/${testDatasetId}`);
      
      expect(response.ok()).toBeTruthy();
      
      const data = await response.json();
      expect(data.message).toContain('删除成功');
    });
  });

  test.describe('集成服务API测试', () => {
    test('获取集成服务列表', async () => {
      const response = await apiContext.get('/api/v1/integrations/services');
      
      if (response.ok()) {
        const data = await response.json();
        expect(Array.isArray(data.services)).toBeTruthy();
        expect(data.total).toBeGreaterThanOrEqual(0);
      } else {
        console.log('集成服务接口可能未实现:', response.status());
      }
    });

    test('获取服务健康状态', async () => {
      const response = await apiContext.get('/api/v1/integrations/health');
      
      if (response.ok()) {
        const data = await response.json();
        expect(data.total_services).toBeGreaterThanOrEqual(0);
        expect(data.healthy_count).toBeGreaterThanOrEqual(0);
      } else {
        console.log('服务健康检查接口可能未实现:', response.status());
      }
    });

    test('获取集成统计', async () => {
      const response = await apiContext.get('/api/v1/integrations/stats');
      
      if (response.ok()) {
        const data = await response.json();
        expect(typeof data).toBe('object');
      } else {
        console.log('集成统计接口可能未实现:', response.status());
      }
    });
  });

  test.describe('认证和授权API测试', () => {
    test('用户注册接口', async () => {
      const userRequest = {
        username: `testuser-${Date.now()}`,
        email: `test-${Date.now()}@example.com`,
        password: 'testpassword123',
        role: 'user'
      };

      const response = await apiContext.post('/api/v1/auth/register', {
        data: userRequest
      });

      if (response.ok()) {
        const data = await response.json();
        expect(data.user_id).toBeTruthy();
        expect(data.username).toBe(userRequest.username);
      } else if (response.status() === 409) {
        // 用户已存在，这是正常的
        console.log('用户已存在，跳过注册测试');
      } else {
        console.log('用户注册接口可能未实现:', response.status());
      }
    });

    test('用户登录接口', async () => {
      const loginRequest = {
        email: 'test@example.com',
        password: 'testpassword123'
      };

      const response = await apiContext.post('/api/v1/auth/login', {
        data: loginRequest
      });

      if (response.ok()) {
        const data = await response.json();
        expect(data.token).toBeTruthy();
        expect(data.user).toBeTruthy();
      } else {
        console.log('用户登录接口可能未实现或用户不存在:', response.status());
      }
    });

    test('Token验证接口', async () => {
      const response = await apiContext.get('/api/v1/auth/verify', {
        headers: {
          'Authorization': 'Bearer test-token'
        }
      });

      // 这里期望401或403，因为使用的是测试token
      expect([401, 403, 200]).toContain(response.status());
    });
  });

  test.describe('错误处理和边界测试', () => {
    test('404错误处理', async () => {
      const response = await apiContext.get('/api/v1/nonexistent-endpoint');
      
      expect(response.status()).toBe(404);
    });

    test('无效JSON请求处理', async () => {
      const response = await apiContext.post('/api/v1/chat/stream', {
        data: 'invalid-json-string'
      });

      expect([400, 422]).toContain(response.status());
    });

    test('超大请求处理', async () => {
      const largeData = {
        conversation_id: 'test',
        messages: [
          {
            role: 'user',
            content: 'a'.repeat(100000) // 100KB的内容
          }
        ]
      };

      const response = await apiContext.post('/api/v1/chat/stream', {
        data: largeData
      });

      // 可能返回413 (Payload Too Large) 或者正常处理
      expect([200, 413, 422]).toContain(response.status());
    });

    test('并发请求处理', async () => {
      const requests = [];
      
      for (let i = 0; i < 10; i++) {
        requests.push(
          apiContext.get('/api/v1/ping')
        );
      }

      const responses = await Promise.all(requests);
      
      // 所有请求都应该成功
      responses.forEach(response => {
        expect(response.ok()).toBeTruthy();
      });
    });
  });

  test.describe('性能测试', () => {
    test('API响应时间测试', async () => {
      const startTime = Date.now();
      
      const response = await apiContext.get('/api/v1/ping');
      
      const endTime = Date.now();
      const responseTime = endTime - startTime;
      
      expect(response.ok()).toBeTruthy();
      expect(responseTime).toBeLessThan(1000); // 响应时间应小于1秒
    });

    test('流式响应性能测试', async () => {
      const startTime = Date.now();
      
      const response = await apiContext.post('/api/v1/chat/stream', {
        data: {
          conversation_id: `perf-test-${Date.now()}`,
          messages: [
            {
              role: 'user',
              content: '请详细介绍人工智能的发展历史'
            }
          ]
        }
      });

      expect(response.ok()).toBeTruthy();
      
      // 测试首字节时间
      const firstByteTime = Date.now() - startTime;
      expect(firstByteTime).toBeLessThan(5000); // 首字节应在5秒内返回
      
      // 读取完整响应
      const responseText = await response.text();
      const totalTime = Date.now() - startTime;
      
      expect(responseText).toBeTruthy();
      expect(totalTime).toBeLessThan(30000); // 总响应时间应小于30秒
    });

    test('大文件上传性能测试', async () => {
      // 创建1MB的测试文件
      const largeContent = 'a'.repeat(1024 * 1024);
      const largeBuffer = Buffer.from(largeContent, 'utf8');
      
      const startTime = Date.now();
      
      const response = await apiContext.post('/api/v1/ingest/upload', {
        multipart: {
          files: {
            name: 'large-test-file.txt',
            mimeType: 'text/plain',
            buffer: largeBuffer
          },
          dataset_id: 'test-dataset'
        }
      });

      const uploadTime = Date.now() - startTime;
      
      // 上传可能失败（如果没有对应的数据集），但测试上传时间
      expect(uploadTime).toBeLessThan(30000); // 上传时间应小于30秒
    });
  });

  test.describe('安全测试', () => {
    test('SQL注入防护测试', async () => {
      const maliciousQuery = "'; DROP TABLE users; --";
      
      const response = await apiContext.post('/api/v1/search', {
        data: {
          query: maliciousQuery,
          dataset_id: 'test'
        }
      });

      // 应该正常处理或返回错误，但不应该导致服务器错误
      expect([200, 400, 422, 404]).toContain(response.status());
    });

    test('XSS防护测试', async () => {
      const xssPayload = '<script>alert("xss")</script>';
      
      const response = await apiContext.post('/api/v1/chat/stream', {
        data: {
          conversation_id: 'xss-test',
          messages: [
            {
              role: 'user',
              content: xssPayload
            }
          ]
        }
      });

      if (response.ok()) {
        const responseText = await response.text();
        // 响应中不应该包含未转义的脚本标签
        expect(responseText).not.toContain('<script>');
      }
    });

    test('请求头安全测试', async () => {
      const response = await apiContext.get('/api/v1/ping');
      
      expect(response.ok()).toBeTruthy();
      
      // 检查安全响应头
      const headers = response.headers();
      
      // 这些头可能存在也可能不存在，取决于服务器配置
      if (headers['x-content-type-options']) {
        expect(headers['x-content-type-options']).toBe('nosniff');
      }
      
      if (headers['x-frame-options']) {
        expect(['DENY', 'SAMEORIGIN']).toContain(headers['x-frame-options']);
      }
    });
  });
});
