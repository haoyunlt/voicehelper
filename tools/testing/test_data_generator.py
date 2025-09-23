#!/usr/bin/env python3
"""
VoiceHelper测试数据生成器

这个脚本用于生成各种类型的测试数据，包括：
- 对话测试数据
- 语音测试数据
- RAG知识库数据
- 安全测试数据
- 性能测试数据
- 多模态测试数据

使用方法：
    python test_data_generator.py --type all
    python test_data_generator.py --type chat --count 100
    python test_data_generator.py --type voice --language zh-CN
"""

import json
import random
import string
import uuid
import argparse
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TestDataGenerator:
    """测试数据生成器主类"""
    
    def __init__(self, output_dir: str = "generated_test_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化各种模板和配置
        self._init_templates()
        self._init_configs()
    
    def _init_templates(self):
        """初始化数据模板"""
        self.chat_templates = {
            "greetings": [
                "你好", "您好", "Hi", "Hello", "早上好", "下午好", "晚上好"
            ],
            "product_inquiries": [
                "我想了解{product}的功能",
                "能介绍一下{product}吗",
                "{product}有什么特色",
                "如何使用{product}",
                "{product}的价格是多少"
            ],
            "technical_support": [
                "我遇到了{problem}",
                "系统出现{error}",
                "{feature}不工作了",
                "如何解决{issue}",
                "为什么{function}失效了"
            ],
            "products": [
                "VoiceHelper", "语音助手", "智能客服", "语音识别系统", "TTS引擎"
            ],
            "problems": [
                "连接问题", "识别错误", "响应延迟", "音质问题", "功能异常"
            ],
            "errors": [
                "网络错误", "认证失败", "服务超时", "数据异常", "系统崩溃"
            ]
        }
        
        self.voice_templates = {
            "languages": ["zh-CN", "en-US", "ja-JP", "ko-KR", "fr-FR", "de-DE"],
            "emotions": ["neutral", "happy", "sad", "angry", "surprised", "calm"],
            "speaking_rates": ["slow", "normal", "fast"],
            "voice_qualities": ["studio", "broadcast", "telephone", "mobile", "noisy"]
        }
        
        self.security_templates = {
            "sql_injections": [
                "'; DROP TABLE users; --",
                "1' OR '1'='1",
                "admin'/**/OR/**/1=1#",
                "1; UPDATE users SET password='hacked' WHERE id=1; --"
            ],
            "xss_attacks": [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "javascript:alert('XSS')",
                "<svg onload=alert('XSS')>"
            ],
            "invalid_tokens": [
                "invalid_token_123",
                "expired_token_456",
                "malformed.token.789",
                ""
            ]
        }
    
    def _init_configs(self):
        """初始化配置参数"""
        self.default_configs = {
            "chat": {
                "scenarios_per_category": 20,
                "turns_per_conversation": 5,
                "languages": ["zh-CN", "en-US"],
                "complexity_levels": ["simple", "medium", "complex"]
            },
            "voice": {
                "samples_per_language": 50,
                "duration_range": [3, 30],
                "quality_levels": ["high", "medium", "low"],
                "noise_levels": [0, 5, 10, 15, 20]
            },
            "rag": {
                "documents_per_domain": 50,
                "chunk_sizes": [256, 512, 1024],
                "domains": ["product", "technical", "support", "business"]
            },
            "security": {
                "tests_per_category": 25,
                "severity_levels": ["low", "medium", "high", "critical"],
                "attack_types": ["injection", "xss", "auth", "csrf"]
            },
            "performance": {
                "load_scenarios": 20,
                "concurrent_users": [10, 50, 100, 500, 1000],
                "test_durations": [300, 600, 1800, 3600]
            },
            "multimodal": {
                "test_cases_per_type": 30,
                "modality_combinations": [
                    ["text", "image"],
                    ["text", "audio"],
                    ["image", "audio"],
                    ["text", "image", "audio"]
                ]
            }
        }
    
    def generate_chat_data(self, count: int = 50, language: str = "zh-CN") -> Dict[str, Any]:
        """生成对话测试数据"""
        logger.info(f"生成{count}个对话测试数据，语言：{language}")
        
        scenarios = []
        categories = ["product_inquiry", "technical_support", "casual_chat", "complaint"]
        
        for i in range(count):
            category = random.choice(categories)
            scenario = self._generate_conversation_scenario(i + 1, category, language)
            scenarios.append(scenario)
        
        data = {
            "metadata": {
                "name": "生成的对话测试数据",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "total_scenarios": count,
                "language": language,
                "categories": categories
            },
            "scenarios": scenarios
        }
        
        return data
    
    def _generate_conversation_scenario(self, scenario_id: int, category: str, language: str) -> Dict[str, Any]:
        """生成单个对话场景"""
        turns = random.randint(3, 8)
        conversation = []
        
        # 生成对话轮次
        for turn in range(1, turns + 1):
            if turn == 1:
                # 第一轮：用户开场
                user_message = self._generate_opening_message(category, language)
                intent = self._get_intent_for_category(category)
                emotion = "neutral"
            else:
                # 后续轮次
                user_message = self._generate_follow_up_message(category, turn, language)
                intent = self._get_follow_up_intent(category, turn)
                emotion = random.choice(["neutral", "curious", "satisfied", "frustrated"])
            
            # 用户消息
            user_turn = {
                "turn": turn,
                "speaker": "user",
                "message": user_message,
                "intent": intent,
                "emotion": emotion,
                "confidence": round(random.uniform(0.8, 0.98), 2),
                "entities": self._extract_entities(user_message, category),
                "context": {
                    "conversation_stage": self._get_conversation_stage(turn, turns),
                    "user_type": random.choice(["new_user", "existing_user", "premium_user"])
                }
            }
            conversation.append(user_turn)
            
            # 助手回复（除了最后一轮）
            if turn < turns:
                assistant_message = self._generate_assistant_response(user_message, category, language)
                assistant_turn = {
                    "turn": turn,
                    "speaker": "assistant",
                    "message": assistant_message,
                    "intent": "provide_assistance",
                    "emotion": "helpful",
                    "confidence": round(random.uniform(0.9, 0.99), 2),
                    "context": {
                        "response_type": self._get_response_type(category),
                        "information_provided": True
                    }
                }
                conversation.append(assistant_turn)
        
        return {
            "id": f"generated_conv_{scenario_id:03d}",
            "category": category,
            "title": self._generate_scenario_title(category, language),
            "description": self._generate_scenario_description(category, language),
            "complexity": random.choice(["simple", "medium", "complex"]),
            "language": language,
            "conversation": conversation,
            "expected_outcomes": self._generate_expected_outcomes(category),
            "test_assertions": self._generate_test_assertions(category)
        }
    
    def generate_voice_data(self, count: int = 100, language: str = "zh-CN") -> Dict[str, Any]:
        """生成语音测试数据"""
        logger.info(f"生成{count}个语音测试数据，语言：{language}")
        
        test_cases = []
        categories = ["clear_speech", "noisy_environment", "emotional_speech", "fast_speech"]
        
        for i in range(count):
            category = random.choice(categories)
            test_case = self._generate_voice_test_case(i + 1, category, language)
            test_cases.append(test_case)
        
        data = {
            "metadata": {
                "name": "生成的语音测试数据",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "total_samples": count,
                "language": language,
                "categories": categories
            },
            "test_cases": test_cases
        }
        
        return data
    
    def _generate_voice_test_case(self, case_id: int, category: str, language: str) -> Dict[str, Any]:
        """生成单个语音测试用例"""
        duration = round(random.uniform(3.0, 30.0), 1)
        
        return {
            "id": f"voice_test_{case_id:03d}",
            "category": category,
            "language": language,
            "speaker_profile": {
                "gender": random.choice(["male", "female"]),
                "age_group": random.choice(["20-30", "30-40", "40-50", "50-60"]),
                "accent": self._get_accent_for_language(language),
                "speaking_rate": random.choice(self.voice_templates["speaking_rates"])
            },
            "audio_properties": {
                "duration": duration,
                "sample_rate": random.choice([8000, 16000, 22050, 44100]),
                "bit_depth": random.choice([16, 24]),
                "channels": 1,
                "format": random.choice(["wav", "mp3", "webm"]),
                "quality": random.choice(self.voice_templates["voice_qualities"])
            },
            "transcript": self._generate_voice_transcript(category, language, duration),
            "expected_confidence": round(random.uniform(0.8, 0.98), 2),
            "emotion_profile": {
                "primary_emotion": random.choice(self.voice_templates["emotions"]),
                "intensity": round(random.uniform(0.3, 0.9), 2)
            },
            "test_conditions": self._generate_test_conditions(category),
            "evaluation_criteria": self._generate_voice_evaluation_criteria(category)
        }
    
    def generate_rag_data(self, count: int = 200) -> Dict[str, Any]:
        """生成RAG知识库测试数据"""
        logger.info(f"生成{count}个RAG知识库文档")
        
        documents = []
        domains = self.default_configs["rag"]["domains"]
        
        for i in range(count):
            domain = random.choice(domains)
            document = self._generate_rag_document(i + 1, domain)
            documents.append(document)
        
        # 生成查询测试用例
        queries = self._generate_rag_queries(50)
        
        data = {
            "metadata": {
                "name": "生成的RAG知识库数据",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "total_documents": count,
                "total_queries": len(queries),
                "domains": domains
            },
            "documents": documents,
            "query_test_cases": queries
        }
        
        return data
    
    def generate_security_data(self, count: int = 100) -> Dict[str, Any]:
        """生成安全测试数据"""
        logger.info(f"生成{count}个安全测试用例")
        
        test_cases = []
        categories = ["authentication", "authorization", "input_validation", "xss_protection", "sql_injection"]
        
        for i in range(count):
            category = random.choice(categories)
            test_case = self._generate_security_test_case(i + 1, category)
            test_cases.append(test_case)
        
        data = {
            "metadata": {
                "name": "生成的安全测试数据",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "total_test_cases": count,
                "categories": categories
            },
            "test_cases": test_cases
        }
        
        return data
    
    def generate_performance_data(self, count: int = 50) -> Dict[str, Any]:
        """生成性能测试数据"""
        logger.info(f"生成{count}个性能测试场景")
        
        scenarios = []
        test_types = ["load_testing", "stress_testing", "concurrency_testing", "memory_testing"]
        
        for i in range(count):
            test_type = random.choice(test_types)
            scenario = self._generate_performance_scenario(i + 1, test_type)
            scenarios.append(scenario)
        
        data = {
            "metadata": {
                "name": "生成的性能测试数据",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "total_scenarios": count,
                "test_types": test_types
            },
            "scenarios": scenarios
        }
        
        return data
    
    def generate_multimodal_data(self, count: int = 80) -> Dict[str, Any]:
        """生成多模态测试数据"""
        logger.info(f"生成{count}个多模态测试用例")
        
        test_cases = []
        categories = ["text_image_fusion", "text_audio_fusion", "image_audio_fusion", "multimodal_reasoning"]
        
        for i in range(count):
            category = random.choice(categories)
            test_case = self._generate_multimodal_test_case(i + 1, category)
            test_cases.append(test_case)
        
        data = {
            "metadata": {
                "name": "生成的多模态测试数据",
                "version": "1.0.0",
                "created_at": datetime.now().isoformat(),
                "total_test_cases": count,
                "categories": categories
            },
            "test_cases": test_cases
        }
        
        return data
    
    def generate_all_data(self, counts: Optional[Dict[str, int]] = None) -> Dict[str, str]:
        """生成所有类型的测试数据"""
        if counts is None:
            counts = {
                "chat": 50,
                "voice": 100,
                "rag": 200,
                "security": 100,
                "performance": 50,
                "multimodal": 80
            }
        
        generated_files = {}
        
        # 生成各类数据
        data_generators = {
            "chat": self.generate_chat_data,
            "voice": self.generate_voice_data,
            "rag": self.generate_rag_data,
            "security": self.generate_security_data,
            "performance": self.generate_performance_data,
            "multimodal": self.generate_multimodal_data
        }
        
        for data_type, generator in data_generators.items():
            try:
                count = counts.get(data_type, 50)
                data = generator(count)
                filename = f"generated_{data_type}_test_data.json"
                filepath = self.output_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                generated_files[data_type] = str(filepath)
                logger.info(f"已生成{data_type}测试数据：{filepath}")
                
            except Exception as e:
                logger.error(f"生成{data_type}数据时出错：{e}")
        
        return generated_files
    
    # 辅助方法
    def _generate_opening_message(self, category: str, language: str) -> str:
        """生成开场消息"""
        if language == "zh-CN":
            templates = {
                "product_inquiry": ["你好，我想了解VoiceHelper的功能", "能介绍一下你们的语音助手吗"],
                "technical_support": ["我遇到了一个技术问题", "系统出现了异常"],
                "casual_chat": ["你好", "Hi，在吗"],
                "complaint": ["我对你们的服务不满意", "有个问题需要投诉"]
            }
        else:
            templates = {
                "product_inquiry": ["Hello, I'd like to know about VoiceHelper", "Can you tell me about your voice assistant?"],
                "technical_support": ["I'm having a technical issue", "The system is not working properly"],
                "casual_chat": ["Hello", "Hi there"],
                "complaint": ["I'm not satisfied with your service", "I have a complaint"]
            }
        
        return random.choice(templates.get(category, ["Hello"]))
    
    def _generate_assistant_response(self, user_message: str, category: str, language: str) -> str:
        """生成助手回复"""
        if language == "zh-CN":
            responses = {
                "product_inquiry": "很高兴为您介绍VoiceHelper！我们提供企业级的智能语音解决方案...",
                "technical_support": "我来帮您解决这个技术问题。请告诉我具体遇到了什么情况？",
                "casual_chat": "您好！我是VoiceHelper智能助手，有什么可以帮助您的吗？",
                "complaint": "非常抱歉给您带来不便，我会认真处理您的问题..."
            }
        else:
            responses = {
                "product_inquiry": "I'd be happy to introduce VoiceHelper! We provide enterprise-level intelligent voice solutions...",
                "technical_support": "I'll help you solve this technical issue. Could you tell me what specific problem you're encountering?",
                "casual_chat": "Hello! I'm VoiceHelper AI assistant. How can I help you today?",
                "complaint": "I sincerely apologize for the inconvenience. I will handle your issue carefully..."
            }
        
        return responses.get(category, "How can I help you?")
    
    def _get_intent_for_category(self, category: str) -> str:
        """根据类别获取意图"""
        intent_mapping = {
            "product_inquiry": "product_inquiry",
            "technical_support": "support_request",
            "casual_chat": "greeting",
            "complaint": "complaint"
        }
        return intent_mapping.get(category, "unknown")
    
    def _extract_entities(self, message: str, category: str) -> List[Dict[str, Any]]:
        """提取实体"""
        entities = []
        
        # 简单的实体提取逻辑
        if "VoiceHelper" in message:
            entities.append({
                "type": "product",
                "value": "VoiceHelper",
                "start": message.find("VoiceHelper"),
                "end": message.find("VoiceHelper") + len("VoiceHelper")
            })
        
        return entities
    
    def _generate_voice_transcript(self, category: str, language: str, duration: float) -> str:
        """生成语音转录文本"""
        if language == "zh-CN":
            templates = {
                "clear_speech": "欢迎使用VoiceHelper智能语音助手，我们提供专业的语音识别和合成服务",
                "noisy_environment": "请帮我查询一下今天的会议安排",
                "emotional_speech": "太好了！我们的项目终于成功了！",
                "fast_speech": "我需要快速处理这个紧急任务，请立即帮我联系相关人员"
            }
        else:
            templates = {
                "clear_speech": "Welcome to VoiceHelper intelligent voice assistant",
                "noisy_environment": "Please help me check today's meeting schedule",
                "emotional_speech": "Great! Our project finally succeeded!",
                "fast_speech": "I need to handle this urgent task quickly"
            }
        
        base_text = templates.get(category, "Test audio content")
        
        # 根据时长调整文本长度
        if duration > 15:
            base_text = base_text + " " + base_text
        
        return base_text
    
    def _generate_rag_document(self, doc_id: int, domain: str) -> Dict[str, Any]:
        """生成RAG文档"""
        titles = {
            "product": f"VoiceHelper产品功能介绍 {doc_id}",
            "technical": f"技术架构文档 {doc_id}",
            "support": f"常见问题解答 {doc_id}",
            "business": f"商业政策说明 {doc_id}"
        }
        
        content_templates = {
            "product": "VoiceHelper是一个企业级智能语音助手平台，提供语音识别、语音合成、对话管理等核心功能...",
            "technical": "系统采用微服务架构，包括网关服务、语音处理服务、对话管理服务等核心组件...",
            "support": "Q: 如何提高语音识别准确率？A: 可以通过优化音频质量、调整语言模型等方式...",
            "business": "我们提供灵活的定价方案，包括基础版、专业版和企业版，满足不同规模企业的需求..."
        }
        
        return {
            "id": f"{domain}_doc_{doc_id:03d}",
            "title": titles[domain],
            "category": domain,
            "content": content_templates[domain],
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "word_count": len(content_templates[domain].split()),
                "language": "zh-CN"
            }
        }
    
    def _generate_rag_queries(self, count: int) -> List[Dict[str, Any]]:
        """生成RAG查询测试用例"""
        queries = []
        query_templates = [
            "VoiceHelper支持哪些语言？",
            "如何提高语音识别准确率？",
            "系统架构是怎样的？",
            "价格方案有哪些？"
        ]
        
        for i in range(count):
            query = {
                "id": f"query_{i+1:03d}",
                "query": random.choice(query_templates),
                "expected_documents": [f"product_doc_001"],
                "relevance_threshold": round(random.uniform(0.7, 0.9), 2)
            }
            queries.append(query)
        
        return queries
    
    def _generate_security_test_case(self, case_id: int, category: str) -> Dict[str, Any]:
        """生成安全测试用例"""
        return {
            "id": f"security_test_{case_id:03d}",
            "category": category,
            "severity": random.choice(["low", "medium", "high", "critical"]),
            "test_data": self._get_security_test_data(category),
            "expected_result": self._get_security_expected_result(category)
        }
    
    def _get_security_test_data(self, category: str) -> Dict[str, Any]:
        """获取安全测试数据"""
        if category == "sql_injection":
            return {
                "malicious_input": random.choice(self.security_templates["sql_injections"]),
                "target_parameter": random.choice(["username", "search_query", "user_id"])
            }
        elif category == "xss_protection":
            return {
                "malicious_script": random.choice(self.security_templates["xss_attacks"]),
                "target_field": random.choice(["message", "username", "comment"])
            }
        else:
            return {"test_input": "generic_security_test"}
    
    def _get_security_expected_result(self, category: str) -> Dict[str, Any]:
        """获取安全测试预期结果"""
        return {
            "attack_blocked": True,
            "error_code": "SECURITY_VIOLATION",
            "http_status": 400,
            "security_alert": True
        }
    
    def _generate_performance_scenario(self, scenario_id: int, test_type: str) -> Dict[str, Any]:
        """生成性能测试场景"""
        return {
            "id": f"perf_test_{scenario_id:03d}",
            "type": test_type,
            "config": {
                "concurrent_users": random.choice([10, 50, 100, 200, 500]),
                "test_duration": random.choice([300, 600, 1800]),
                "ramp_up_time": random.choice([60, 120, 300])
            },
            "targets": {
                "response_time_p95": f"<{random.randint(500, 2000)}ms",
                "throughput": f">{random.randint(50, 500)}_rps",
                "error_rate": "<1%"
            }
        }
    
    def _generate_multimodal_test_case(self, case_id: int, category: str) -> Dict[str, Any]:
        """生成多模态测试用例"""
        modalities = {
            "text_image_fusion": ["text", "image"],
            "text_audio_fusion": ["text", "audio"],
            "image_audio_fusion": ["image", "audio"],
            "multimodal_reasoning": ["text", "image", "audio"]
        }
        
        return {
            "id": f"multimodal_test_{case_id:03d}",
            "category": category,
            "modalities": modalities[category],
            "fusion_strategy": random.choice(["early_fusion", "late_fusion", "attention_fusion"]),
            "test_data": {
                "text": "请分析这个内容",
                "image_path": f"test_image_{case_id}.jpg",
                "audio_path": f"test_audio_{case_id}.wav"
            },
            "expected_output": {
                "fusion_quality": ">0.8",
                "response_relevance": ">0.85"
            }
        }
    
    # 更多辅助方法...
    def _get_accent_for_language(self, language: str) -> str:
        accents = {
            "zh-CN": "standard_mandarin",
            "en-US": "general_american",
            "ja-JP": "standard_japanese",
            "ko-KR": "seoul_standard"
        }
        return accents.get(language, "standard")
    
    def _generate_scenario_title(self, category: str, language: str) -> str:
        if language == "zh-CN":
            titles = {
                "product_inquiry": "产品功能咨询",
                "technical_support": "技术支持请求",
                "casual_chat": "日常对话",
                "complaint": "用户投诉处理"
            }
        else:
            titles = {
                "product_inquiry": "Product Feature Inquiry",
                "technical_support": "Technical Support Request",
                "casual_chat": "Casual Conversation",
                "complaint": "Customer Complaint"
            }
        return titles.get(category, "General Conversation")
    
    def _generate_scenario_description(self, category: str, language: str) -> str:
        if language == "zh-CN":
            descriptions = {
                "product_inquiry": "用户询问产品功能和特性",
                "technical_support": "用户寻求技术问题解决方案",
                "casual_chat": "用户进行日常闲聊",
                "complaint": "用户表达不满并寻求解决"
            }
        else:
            descriptions = {
                "product_inquiry": "User inquires about product features",
                "technical_support": "User seeks technical problem resolution",
                "casual_chat": "User engages in casual conversation",
                "complaint": "User expresses dissatisfaction and seeks resolution"
            }
        return descriptions.get(category, "General conversation scenario")
    
    def _get_conversation_stage(self, turn: int, total_turns: int) -> str:
        if turn == 1:
            return "opening"
        elif turn < total_turns // 2:
            return "information_gathering"
        elif turn < total_turns - 1:
            return "problem_solving"
        else:
            return "closing"
    
    def _get_follow_up_intent(self, category: str, turn: int) -> str:
        intents = {
            "product_inquiry": ["feature_inquiry", "pricing_inquiry", "comparison_request"],
            "technical_support": ["problem_clarification", "solution_request", "status_check"],
            "casual_chat": ["continue_conversation", "topic_change", "farewell"],
            "complaint": ["issue_escalation", "solution_demand", "satisfaction_check"]
        }
        return random.choice(intents.get(category, ["general_inquiry"]))
    
    def _generate_follow_up_message(self, category: str, turn: int, language: str) -> str:
        if language == "zh-CN":
            messages = {
                "product_inquiry": ["价格怎么样？", "支持哪些功能？", "如何开始使用？"],
                "technical_support": ["具体怎么操作？", "还有其他解决方案吗？", "问题解决了吗？"],
                "casual_chat": ["今天天气不错", "你觉得怎么样？", "谢谢你的帮助"],
                "complaint": ["这个问题很严重", "我需要立即解决", "你们的处理方案是什么？"]
            }
        else:
            messages = {
                "product_inquiry": ["What about pricing?", "What features are supported?", "How do I get started?"],
                "technical_support": ["How exactly do I do this?", "Are there other solutions?", "Is the problem resolved?"],
                "casual_chat": ["Nice weather today", "What do you think?", "Thanks for your help"],
                "complaint": ["This is a serious issue", "I need immediate resolution", "What's your solution?"]
            }
        return random.choice(messages.get(category, ["Could you help me?"]))
    
    def _get_response_type(self, category: str) -> str:
        response_types = {
            "product_inquiry": "informational",
            "technical_support": "instructional",
            "casual_chat": "conversational",
            "complaint": "empathetic"
        }
        return response_types.get(category, "general")
    
    def _generate_expected_outcomes(self, category: str) -> Dict[str, str]:
        return {
            "user_satisfaction": random.choice(["low", "medium", "high"]),
            "issue_resolution": random.choice(["unresolved", "partially_resolved", "fully_resolved"]),
            "conversation_quality": random.choice(["poor", "good", "excellent"])
        }
    
    def _generate_test_assertions(self, category: str) -> List[Dict[str, Any]]:
        return [
            {
                "type": "intent_recognition",
                "expected": self._get_intent_for_category(category),
                "confidence_threshold": 0.8
            },
            {
                "type": "response_relevance",
                "expected": "high_relevance",
                "threshold": 0.85
            }
        ]
    
    def _generate_test_conditions(self, category: str) -> Dict[str, Any]:
        conditions = {
            "clear_speech": {"noise_level": 0, "echo": False},
            "noisy_environment": {"noise_level": random.randint(10, 25), "echo": True},
            "emotional_speech": {"emotion_intensity": random.uniform(0.6, 0.9)},
            "fast_speech": {"speaking_rate": random.uniform(1.5, 2.0)}
        }
        return conditions.get(category, {})
    
    def _generate_voice_evaluation_criteria(self, category: str) -> Dict[str, str]:
        return {
            "word_accuracy": f">{random.randint(85, 98)}%",
            "processing_time": f"<{random.randint(200, 1000)}ms",
            "confidence_score": f">{random.uniform(0.8, 0.95):.2f}"
        }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="VoiceHelper测试数据生成器")
    parser.add_argument("--type", choices=["all", "chat", "voice", "rag", "security", "performance", "multimodal"],
                       default="all", help="要生成的数据类型")
    parser.add_argument("--count", type=int, default=50, help="生成数据的数量")
    parser.add_argument("--language", default="zh-CN", help="语言设置")
    parser.add_argument("--output-dir", default="generated_test_data", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = TestDataGenerator(args.output_dir)
    
    try:
        if args.type == "all":
            # 生成所有类型的数据
            files = generator.generate_all_data()
            logger.info("所有测试数据生成完成：")
            for data_type, filepath in files.items():
                logger.info(f"  {data_type}: {filepath}")
        else:
            # 生成指定类型的数据
            data_generators = {
                "chat": generator.generate_chat_data,
                "voice": generator.generate_voice_data,
                "rag": generator.generate_rag_data,
                "security": generator.generate_security_data,
                "performance": generator.generate_performance_data,
                "multimodal": generator.generate_multimodal_data
            }
            
            if args.type in data_generators:
                if args.type in ["chat", "voice"]:
                    data = data_generators[args.type](args.count, args.language)
                else:
                    data = data_generators[args.type](args.count)
                
                filename = f"generated_{args.type}_test_data.json"
                filepath = Path(args.output_dir) / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"已生成{args.type}测试数据：{filepath}")
            else:
                logger.error(f"不支持的数据类型：{args.type}")
    
    except Exception as e:
        logger.error(f"生成测试数据时出错：{e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
