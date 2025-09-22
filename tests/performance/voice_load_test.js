/**
 * VoiceHelper 语音链路压测脚本
 * 功能: 文本70% + 语音30% 混合场景 + 乱序/丢包注入 + 性能指标收集
 */

import ws from 'k6/ws';
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Trend, Rate, Gauge } from 'k6/metrics';
import { randomBytes } from 'k6/crypto';

// 自定义指标
const wsConnections = new Counter('ws_connections_total');
const wsConnectionDuration = new Trend('ws_connection_duration');
const e2eLatency = new Trend('e2e_latency_ms');
const audioFramesSent = new Counter('audio_frames_sent_total');
const audioFramesReceived = new Counter('audio_frames_received_total');
const audioDropRate = new Rate('audio_drop_rate');
const audioOutOfOrder = new Counter('audio_out_of_order_total');
const bargeInSuccess = new Rate('barge_in_success_rate');
const ttsLatency = new Trend('tts_first_chunk_latency_ms');
const asrLatency = new Trend('asr_final_latency_ms');
const concurrentUsers = new Gauge('concurrent_users');

// 测试配置
export let options = {
    scenarios: {
        // 文本聊天场景 (70%)
        text_chat: {
            executor: 'constant-vus',
            vus: 70,
            duration: '5m',
            exec: 'textChatTest',
            tags: { scenario: 'text_chat' },
        },
        
        // 语音对话场景 (30%)
        voice_chat: {
            executor: 'constant-vus',
            vus: 30,
            duration: '5m',
            exec: 'voiceChatTest',
            tags: { scenario: 'voice_chat' },
        },
        
        // 打断测试场景
        barge_in_test: {
            executor: 'constant-vus',
            vus: 5,
            duration: '5m',
            exec: 'bargeInTest',
            tags: { scenario: 'barge_in' },
        },
        
        // 网络异常测试
        network_chaos: {
            executor: 'constant-vus',
            vus: 10,
            duration: '5m',
            exec: 'networkChaosTest',
            tags: { scenario: 'chaos' },
        }
    },
    
    thresholds: {
        // 性能阈值
        'e2e_latency_ms': ['p(95)<500', 'p(99)<1000'],
        'tts_first_chunk_latency_ms': ['p(95)<200'],
        'asr_final_latency_ms': ['p(95)<300'],
        'audio_drop_rate': ['rate<0.01'], // 丢包率 < 1%
        'barge_in_success_rate': ['rate>0.95'], // 打断成功率 > 95%
        'ws_connection_duration': ['p(95)<30000'], // 连接持续时间
        'http_req_duration': ['p(95)<200'], // HTTP请求延迟
        'http_req_failed': ['rate<0.01'], // HTTP错误率 < 1%
    }
};

// 测试数据
const testMessages = [
    "你好，请帮我查询今天的天气情况。",
    "我想了解一下人工智能的发展历史。",
    "请帮我制定一个健康的饮食计划。",
    "能否介绍一下量子计算的基本原理？",
    "我需要一些关于投资理财的建议。"
];

const testAudioFiles = [
    'hello_16k.pcm',
    'question_16k.pcm', 
    'command_16k.pcm',
    'conversation_16k.pcm'
];

// 环境配置
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8080';
const WS_URL = __ENV.WS_URL || 'ws://localhost:8080';

/**
 * 文本聊天测试
 */
export function textChatTest() {
    const sessionId = `text_${__VU}_${__ITER}`;
    concurrentUsers.add(1);
    
    // 建立SSE连接
    const sseUrl = `${BASE_URL}/api/v1/chat/stream`;
    const connectStart = Date.now();
    
    // 首先建立SSE流
    const streamResponse = http.get(sseUrl, {
        headers: {
            'Accept': 'text/event-stream',
            'Cache-Control': 'no-cache',
        },
        timeout: '30s',
    });
    
    check(streamResponse, {
        'SSE connection established': (r) => r.status === 200,
    });
    
    if (streamResponse.status !== 200) {
        concurrentUsers.add(-1);
        return;
    }
    
    const streamId = extractStreamId(streamResponse.body);
    
    // 发送聊天请求
    const message = testMessages[Math.floor(Math.random() * testMessages.length)];
    const chatPayload = {
        message: message,
        session_id: sessionId,
        stream_id: streamId,
        temperature: 0.7,
        max_tokens: 500
    };
    
    const requestStart = Date.now();
    const chatResponse = http.post(`${BASE_URL}/api/v1/chat/request`, JSON.stringify(chatPayload), {
        headers: {
            'Content-Type': 'application/json',
        },
        timeout: '10s',
    });
    
    check(chatResponse, {
        'Chat request accepted': (r) => r.status === 200,
        'Response has trace_id': (r) => r.json('trace_id') !== undefined,
    });
    
    // 模拟等待响应完成
    sleep(Math.random() * 3 + 1); // 1-4秒随机等待
    
    const e2eTime = Date.now() - requestStart;
    e2eLatency.add(e2eTime);
    
    concurrentUsers.add(-1);
}

/**
 * 语音对话测试
 */
export function voiceChatTest() {
    const sessionId = `voice_${__VU}_${__ITER}`;
    concurrentUsers.add(1);
    
    const wsUrl = `${WS_URL}/api/v1/voice/stream`;
    const connectStart = Date.now();
    
    const response = ws.connect(wsUrl, {}, function (socket) {
        wsConnections.add(1);
        
        let framesSent = 0;
        let framesReceived = 0;
        let sequenceNum = 0;
        let expectedSequence = 0;
        let outOfOrderCount = 0;
        let firstTTSChunk = null;
        let asrFinalTime = null;
        let conversationStart = Date.now();
        
        socket.on('open', () => {
            console.log(`Voice WebSocket connected: ${sessionId}`);
            
            // 发送测试音频数据
            sendTestAudio(socket, sessionId);
        });
        
        socket.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                handleVoiceMessage(message, {
                    sessionId,
                    framesReceived: () => ++framesReceived,
                    expectedSequence: () => expectedSequence++,
                    outOfOrderCount: () => ++outOfOrderCount,
                    setFirstTTSChunk: (time) => firstTTSChunk = time,
                    setASRFinalTime: (time) => asrFinalTime = time,
                });
            } catch (e) {
                // 可能是二进制音频数据
                framesReceived++;
                audioFramesReceived.add(1);
            }
        });
        
        socket.on('close', () => {
            const duration = Date.now() - connectStart;
            wsConnectionDuration.add(duration);
            
            // 计算指标
            if (framesSent > 0) {
                const dropRate = Math.max(0, (framesSent - framesReceived) / framesSent);
                audioDropRate.add(dropRate);
            }
            
            if (framesReceived > 0 && outOfOrderCount > 0) {
                audioOutOfOrder.add(outOfOrderCount);
            }
            
            if (firstTTSChunk) {
                ttsLatency.add(firstTTSChunk - conversationStart);
            }
            
            if (asrFinalTime) {
                asrLatency.add(asrFinalTime - conversationStart);
            }
            
            console.log(`Voice session ${sessionId} completed: sent=${framesSent}, received=${framesReceived}`);
        });
        
        socket.on('error', (e) => {
            console.error(`WebSocket error for ${sessionId}:`, e);
        });
        
        // 发送音频帧的函数
        function sendTestAudio(socket, sessionId) {
            const audioFile = testAudioFiles[Math.floor(Math.random() * testAudioFiles.length)];
            const audioData = generateTestAudioData(1600); // 100ms of 16kHz audio
            
            // 分帧发送 (20ms per frame)
            const frameSize = 320; // 20ms @ 16kHz
            const totalFrames = Math.floor(audioData.length / frameSize);
            
            for (let i = 0; i < totalFrames; i++) {
                const frameStart = i * frameSize;
                const frameEnd = Math.min(frameStart + frameSize, audioData.length);
                const frame = audioData.slice(frameStart, frameEnd);
                
                // 构造音频帧头部 (20字节)
                const header = new ArrayBuffer(20);
                const headerView = new DataView(header);
                
                headerView.setUint32(0, sequenceNum++, true); // sequence_num
                headerView.setUint32(4, 16000, true);         // sample_rate
                headerView.setUint8(8, 1);                    // channels
                headerView.setUint16(9, frame.length, true);  // frame_size
                headerView.setBigUint64(12, BigInt(Date.now()), true); // timestamp
                
                // 组合头部和音频数据
                const frameData = new Uint8Array(20 + frame.length);
                frameData.set(new Uint8Array(header), 0);
                frameData.set(frame, 20);
                
                // 注入网络异常 (3% 概率)
                if (Math.random() < 0.03) {
                    // 3% 概率乱序发送
                    setTimeout(() => {
                        if (socket.readyState === WebSocket.OPEN) {
                            socket.sendBinary(frameData);
                            framesSent++;
                            audioFramesSent.add(1);
                        }
                    }, Math.random() * 100); // 0-100ms延迟
                } else if (Math.random() < 0.01) {
                    // 1% 概率丢包 (不发送)
                    continue;
                } else {
                    // 正常发送
                    if (socket.readyState === WebSocket.OPEN) {
                        socket.sendBinary(frameData);
                        framesSent++;
                        audioFramesSent.add(1);
                    }
                }
                
                // 模拟实时发送间隔 (20ms)
                sleep(0.02);
            }
        }
        
        // 保持连接一段时间
        setTimeout(() => {
            socket.close();
        }, 10000 + Math.random() * 5000); // 10-15秒
    });
    
    check(response, {
        'WebSocket connection successful': (r) => r && r.status === 101,
    });
    
    concurrentUsers.add(-1);
}

/**
 * 打断测试
 */
export function bargeInTest() {
    const sessionId = `barge_${__VU}_${__ITER}`;
    concurrentUsers.add(1);
    
    const wsUrl = `${WS_URL}/api/v1/voice/stream`;
    
    const response = ws.connect(wsUrl, {}, function (socket) {
        let ttsStarted = false;
        let interruptSent = false;
        let interruptSuccess = false;
        
        socket.on('open', () => {
            // 发送一个长文本触发TTS
            const longMessage = "这是一个很长的测试消息，用来触发TTS生成。我们将在TTS播放过程中测试打断功能。这个消息需要足够长，以便我们有时间发送打断信号。";
            
            socket.send(JSON.stringify({
                type: 'text_message',
                data: { message: longMessage },
                meta: { session_id: sessionId }
            }));
        });
        
        socket.on('message', (data) => {
            try {
                const message = JSON.parse(data);
                
                if (message.type === 'tts_start') {
                    ttsStarted = true;
                    
                    // 在TTS开始后随机时间发送打断
                    setTimeout(() => {
                        if (!interruptSent) {
                            socket.send(JSON.stringify({
                                type: 'cancel',
                                data: { reason: 'barge_in' },
                                meta: { session_id: sessionId }
                            }));
                            interruptSent = true;
                        }
                    }, Math.random() * 2000 + 500); // 0.5-2.5秒后打断
                }
                
                if (message.type === 'tts_cancelled' && interruptSent) {
                    interruptSuccess = true;
                }
                
            } catch (e) {
                // 忽略二进制数据
            }
        });
        
        socket.on('close', () => {
            if (interruptSent) {
                bargeInSuccess.add(interruptSuccess);
            }
        });
        
        // 10秒后关闭连接
        setTimeout(() => {
            socket.close();
        }, 10000);
    });
    
    check(response, {
        'Barge-in WebSocket connection successful': (r) => r && r.status === 101,
    });
    
    concurrentUsers.add(-1);
}

/**
 * 网络异常测试
 */
export function networkChaosTest() {
    const sessionId = `chaos_${__VU}_${__ITER}`;
    concurrentUsers.add(1);
    
    const wsUrl = `${WS_URL}/api/v1/voice/stream`;
    
    const response = ws.connect(wsUrl, {}, function (socket) {
        let reconnectCount = 0;
        const maxReconnects = 3;
        
        socket.on('open', () => {
            console.log(`Chaos test connection ${reconnectCount + 1} for ${sessionId}`);
            
            // 发送一些数据
            sendChaosTestData(socket, sessionId);
            
            // 随机断开连接模拟网络问题
            setTimeout(() => {
                if (Math.random() < 0.7) { // 70% 概率断开
                    socket.close();
                }
            }, Math.random() * 5000 + 1000); // 1-6秒后断开
        });
        
        socket.on('close', () => {
            // 尝试重连
            if (reconnectCount < maxReconnects) {
                reconnectCount++;
                setTimeout(() => {
                    // 重新连接逻辑 (简化版)
                    console.log(`Attempting reconnect ${reconnectCount} for ${sessionId}`);
                }, Math.random() * 2000 + 500);
            }
        });
        
        function sendChaosTestData(socket, sessionId) {
            // 发送一些测试消息
            for (let i = 0; i < 5; i++) {
                setTimeout(() => {
                    if (socket.readyState === WebSocket.OPEN) {
                        socket.send(JSON.stringify({
                            type: 'test_message',
                            data: { index: i, message: `Chaos test message ${i}` },
                            meta: { session_id: sessionId }
                        }));
                    }
                }, i * 500);
            }
        }
    });
    
    check(response, {
        'Chaos test connection successful': (r) => r && r.status === 101,
    });
    
    concurrentUsers.add(-1);
}

// 辅助函数

function extractStreamId(sseBody) {
    // 从SSE响应中提取stream_id (简化实现)
    const match = sseBody.match(/"stream_id":"([^"]+)"/);
    return match ? match[1] : `stream_${Date.now()}`;
}

function handleVoiceMessage(message, context) {
    switch (message.type) {
        case 'asr_final':
            context.setASRFinalTime(Date.now());
            break;
            
        case 'tts_chunk':
            if (!context.firstTTSChunk) {
                context.setFirstTTSChunk(Date.now());
            }
            break;
            
        case 'audio_frame':
            const receivedSeq = message.meta?.sequence_num || 0;
            const expectedSeq = context.expectedSequence();
            
            if (receivedSeq < expectedSeq - 1) {
                context.outOfOrderCount();
            }
            
            context.framesReceived();
            break;
    }
}

function generateTestAudioData(length) {
    // 生成模拟音频数据 (16位PCM)
    const audioData = new Int16Array(length);
    for (let i = 0; i < length; i++) {
        // 生成简单的正弦波 + 噪声
        const sine = Math.sin(2 * Math.PI * 440 * i / 16000) * 0.3; // 440Hz
        const noise = (Math.random() - 0.5) * 0.1;
        audioData[i] = Math.floor((sine + noise) * 32767);
    }
    return new Uint8Array(audioData.buffer);
}

// 测试生命周期钩子

export function setup() {
    console.log('Starting VoiceHelper load test...');
    
    // 健康检查
    const healthResponse = http.get(`${BASE_URL}/health`);
    check(healthResponse, {
        'Service is healthy': (r) => r.status === 200,
    });
    
    return {
        startTime: Date.now(),
    };
}

export function teardown(data) {
    const duration = (Date.now() - data.startTime) / 1000;
    console.log(`Load test completed in ${duration} seconds`);
    
    // 生成测试报告
    console.log('=== Test Summary ===');
    console.log(`Total WebSocket connections: ${wsConnections.count}`);
    console.log(`Audio frames sent: ${audioFramesSent.count}`);
    console.log(`Audio frames received: ${audioFramesReceived.count}`);
    console.log(`Out of order frames: ${audioOutOfOrder.count}`);
}
