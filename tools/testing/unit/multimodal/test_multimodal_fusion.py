"""
多模态融合测试用例
测试覆盖：文本+图像+语音融合、跨模态注意力机制、模态权重分配、融合策略优化
"""

import pytest
import asyncio
import numpy as np
import json
import time
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
import base64
import io


@dataclass
class ModalityData:
    """模态数据类"""
    modality_type: str  # "text", "image", "audio"
    data: Any
    features: Optional[np.ndarray] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FusionResult:
    """融合结果类"""
    unified_representation: np.ndarray
    modality_weights: Dict[str, float]
    confidence: float
    fusion_strategy: str
    processing_time: float
    metadata: Dict[str, Any] = None


class TestMultimodalDataPreprocessing:
    """多模态数据预处理测试"""
    
    @pytest.fixture
    def mock_text_processor(self):
        """模拟文本处理器"""
        class MockTextProcessor:
            def __init__(self):
                self.vocab_size = 10000
                self.embedding_dim = 768
            
            def tokenize(self, text):
                """文本分词"""
                # 简单的分词模拟
                tokens = text.lower().split()
                token_ids = [hash(token) % self.vocab_size for token in tokens]
                return {
                    "tokens": tokens,
                    "token_ids": token_ids,
                    "attention_mask": [1] * len(tokens)
                }
            
            def encode(self, text):
                """文本编码"""
                tokens = self.tokenize(text)
                # 模拟BERT类编码器输出
                features = np.random.rand(len(tokens["tokens"]), self.embedding_dim)
                
                # 添加[CLS]标记的全局表示
                global_features = np.mean(features, axis=0)
                
                return {
                    "token_features": features,
                    "global_features": global_features,
                    "attention_mask": tokens["attention_mask"]
                }
        
        return MockTextProcessor()
    
    @pytest.fixture
    def mock_image_processor(self):
        """模拟图像处理器"""
        class MockImageProcessor:
            def __init__(self):
                self.image_size = (224, 224)
                self.feature_dim = 2048
            
            def preprocess_image(self, image_data):
                """图像预处理"""
                # 模拟图像预处理步骤
                if isinstance(image_data, bytes):
                    # 假设是图像字节数据
                    image_array = np.frombuffer(image_data[:1000], dtype=np.uint8)
                    if len(image_array) < 1000:
                        image_array = np.pad(image_array, (0, 1000 - len(image_array)))
                else:
                    image_array = np.array(image_data)
                
                # 模拟调整大小和归一化
                processed = image_array.reshape(-1)[:self.image_size[0] * self.image_size[1]]
                if len(processed) < self.image_size[0] * self.image_size[1]:
                    processed = np.pad(processed, (0, self.image_size[0] * self.image_size[1] - len(processed)))
                
                normalized = processed / 255.0
                
                return {
                    "processed_image": normalized.reshape(self.image_size),
                    "original_shape": image_array.shape,
                    "preprocessing_info": {
                        "resized": True,
                        "normalized": True,
                        "target_size": self.image_size
                    }
                }
            
            def extract_features(self, processed_image):
                """提取图像特征"""
                # 模拟CNN特征提取
                features = np.random.rand(self.feature_dim)
                
                # 模拟不同层级的特征
                low_level_features = features[:512]    # 低级特征（边缘、纹理）
                mid_level_features = features[512:1024]  # 中级特征（形状、模式）
                high_level_features = features[1024:]   # 高级特征（对象、语义）
                
                return {
                    "global_features": features,
                    "low_level_features": low_level_features,
                    "mid_level_features": mid_level_features,
                    "high_level_features": high_level_features,
                    "spatial_features": features.reshape(32, -1)  # 空间特征图
                }
        
        return MockImageProcessor()
    
    @pytest.fixture
    def mock_audio_processor(self):
        """模拟音频处理器"""
        class MockAudioProcessor:
            def __init__(self):
                self.sample_rate = 16000
                self.feature_dim = 512
            
            def preprocess_audio(self, audio_data):
                """音频预处理"""
                if isinstance(audio_data, bytes):
                    audio_array = np.frombuffer(audio_data, dtype=np.int16)
                else:
                    audio_array = np.array(audio_data)
                
                # 模拟音频预处理
                normalized_audio = audio_array.astype(np.float32) / 32768.0
                
                # 模拟重采样到目标采样率
                target_length = len(normalized_audio)
                if target_length > self.sample_rate * 10:  # 限制最大10秒
                    normalized_audio = normalized_audio[:self.sample_rate * 10]
                
                return {
                    "processed_audio": normalized_audio,
                    "sample_rate": self.sample_rate,
                    "duration": len(normalized_audio) / self.sample_rate,
                    "preprocessing_info": {
                        "normalized": True,
                        "resampled": True
                    }
                }
            
            def extract_features(self, processed_audio):
                """提取音频特征"""
                audio_length = len(processed_audio)
                
                # 模拟不同类型的音频特征
                # 1. 频谱特征
                spectral_features = np.random.rand(128)
                
                # 2. 韵律特征
                prosodic_features = np.random.rand(64)
                
                # 3. 语音特征
                linguistic_features = np.random.rand(256)
                
                # 4. 情感特征
                emotional_features = np.random.rand(64)
                
                # 合并所有特征
                global_features = np.concatenate([
                    spectral_features, prosodic_features, 
                    linguistic_features, emotional_features
                ])
                
                return {
                    "global_features": global_features,
                    "spectral_features": spectral_features,
                    "prosodic_features": prosodic_features,
                    "linguistic_features": linguistic_features,
                    "emotional_features": emotional_features,
                    "temporal_features": np.random.rand(100, 64)  # 时序特征
                }
        
        return MockAudioProcessor()
    
    def test_text_preprocessing(self, mock_text_processor):
        """测试文本预处理"""
        test_texts = [
            "这是一个简单的测试文本",
            "Hello, this is a test in English!",
            "混合语言 mixed language test 测试",
            "",  # 空文本
            "a" * 1000,  # 长文本
        ]
        
        for text in test_texts:
            if not text:  # 跳过空文本的编码测试
                continue
                
            # 测试分词
            tokens = mock_text_processor.tokenize(text)
            assert "tokens" in tokens
            assert "token_ids" in tokens
            assert "attention_mask" in tokens
            assert len(tokens["tokens"]) == len(tokens["token_ids"])
            assert len(tokens["tokens"]) == len(tokens["attention_mask"])
            
            # 测试编码
            encoded = mock_text_processor.encode(text)
            assert "global_features" in encoded
            assert "token_features" in encoded
            assert encoded["global_features"].shape == (mock_text_processor.embedding_dim,)
            assert encoded["token_features"].shape[1] == mock_text_processor.embedding_dim
            
            print(f"文本 '{text[:50]}...' 处理结果:")
            print(f"  Token数量: {len(tokens['tokens'])}")
            print(f"  特征维度: {encoded['global_features'].shape}")
    
    def test_image_preprocessing(self, mock_image_processor):
        """测试图像预处理"""
        # 创建测试图像数据
        test_images = [
            np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),  # RGB图像
            np.random.randint(0, 255, (50, 50), dtype=np.uint8),       # 灰度图像
            b"fake_image_bytes" * 100,                                   # 字节数据
        ]
        
        for i, image_data in enumerate(test_images):
            # 测试预处理
            processed = mock_image_processor.preprocess_image(image_data)
            assert "processed_image" in processed
            assert processed["processed_image"].shape == mock_image_processor.image_size
            
            # 测试特征提取
            features = mock_image_processor.extract_features(processed["processed_image"])
            assert "global_features" in features
            assert features["global_features"].shape == (mock_image_processor.feature_dim,)
            assert "low_level_features" in features
            assert "high_level_features" in features
            
            print(f"图像 {i+1} 处理结果:")
            print(f"  原始形状: {processed.get('original_shape', 'Unknown')}")
            print(f"  处理后形状: {processed['processed_image'].shape}")
            print(f"  特征维度: {features['global_features'].shape}")
    
    def test_audio_preprocessing(self, mock_audio_processor):
        """测试音频预处理"""
        # 创建测试音频数据
        test_audios = [
            np.random.randint(-32768, 32767, 16000, dtype=np.int16),  # 1秒音频
            np.random.randint(-32768, 32767, 48000, dtype=np.int16),  # 3秒音频
            b"fake_audio_bytes" * 1000,                                # 字节数据
        ]
        
        for i, audio_data in enumerate(test_audios):
            # 测试预处理
            processed = mock_audio_processor.preprocess_audio(audio_data)
            assert "processed_audio" in processed
            assert "duration" in processed
            assert processed["sample_rate"] == mock_audio_processor.sample_rate
            
            # 测试特征提取
            features = mock_audio_processor.extract_features(processed["processed_audio"])
            assert "global_features" in features
            assert features["global_features"].shape == (mock_audio_processor.feature_dim,)
            assert "spectral_features" in features
            assert "emotional_features" in features
            
            print(f"音频 {i+1} 处理结果:")
            print(f"  时长: {processed['duration']:.2f}s")
            print(f"  采样率: {processed['sample_rate']}Hz")
            print(f"  特征维度: {features['global_features'].shape}")


class TestCrossModalAttention:
    """跨模态注意力机制测试"""
    
    @pytest.fixture
    def mock_attention_mechanism(self):
        """模拟注意力机制"""
        class MockCrossModalAttention:
            def __init__(self, feature_dim=512):
                self.feature_dim = feature_dim
                self.attention_heads = 8
                self.head_dim = feature_dim // self.attention_heads
            
            def compute_attention_weights(self, query_features, key_features, value_features):
                """计算注意力权重"""
                batch_size = 1
                seq_len_q = query_features.shape[0] if query_features.ndim > 1 else 1
                seq_len_k = key_features.shape[0] if key_features.ndim > 1 else 1
                
                # 确保特征是2D的
                if query_features.ndim == 1:
                    query_features = query_features.reshape(1, -1)
                if key_features.ndim == 1:
                    key_features = key_features.reshape(1, -1)
                if value_features.ndim == 1:
                    value_features = value_features.reshape(1, -1)
                
                # 模拟多头注意力计算
                attention_scores = np.random.rand(self.attention_heads, seq_len_q, seq_len_k)
                
                # 应用softmax归一化
                attention_weights = []
                for head in range(self.attention_heads):
                    head_scores = attention_scores[head]
                    exp_scores = np.exp(head_scores - np.max(head_scores, axis=-1, keepdims=True))
                    head_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
                    attention_weights.append(head_weights)
                
                attention_weights = np.array(attention_weights)
                
                return {
                    "attention_weights": attention_weights,
                    "attention_scores": attention_scores,
                    "num_heads": self.attention_heads
                }
            
            def apply_attention(self, query_modality, key_modality, value_modality, 
                              query_features, key_features, value_features):
                """应用跨模态注意力"""
                attention_result = self.compute_attention_weights(
                    query_features, key_features, value_features
                )
                
                # 模拟注意力应用
                attended_features = np.random.rand(*query_features.shape)
                
                # 计算模态间的相关性
                correlation = np.corrcoef(
                    query_features.flatten(), 
                    key_features.flatten()
                )[0, 1]
                
                return {
                    "attended_features": attended_features,
                    "attention_weights": attention_result["attention_weights"],
                    "cross_modal_correlation": correlation,
                    "query_modality": query_modality,
                    "key_modality": key_modality,
                    "value_modality": value_modality
                }
        
        return MockCrossModalAttention()
    
    def test_text_to_image_attention(self, mock_attention_mechanism):
        """测试文本到图像的注意力"""
        # 创建模拟特征
        text_features = np.random.rand(10, 512)  # 10个token，每个512维
        image_features = np.random.rand(49, 512)  # 7x7空间位置，每个512维
        
        # 计算文本到图像的注意力
        attention_result = mock_attention_mechanism.apply_attention(
            query_modality="text",
            key_modality="image", 
            value_modality="image",
            query_features=text_features,
            key_features=image_features,
            value_features=image_features
        )
        
        # 验证注意力结果
        assert "attended_features" in attention_result
        assert "attention_weights" in attention_result
        assert "cross_modal_correlation" in attention_result
        
        # 验证注意力权重的形状和属性
        attention_weights = attention_result["attention_weights"]
        assert attention_weights.shape[0] == mock_attention_mechanism.attention_heads
        assert attention_weights.shape[1] == text_features.shape[0]  # query序列长度
        assert attention_weights.shape[2] == image_features.shape[0]  # key序列长度
        
        # 验证注意力权重归一化
        for head in range(attention_weights.shape[0]):
            for query_pos in range(attention_weights.shape[1]):
                weights_sum = np.sum(attention_weights[head, query_pos, :])
                assert abs(weights_sum - 1.0) < 1e-6, f"注意力权重未正确归一化: {weights_sum}"
        
        print(f"文本到图像注意力:")
        print(f"  文本特征形状: {text_features.shape}")
        print(f"  图像特征形状: {image_features.shape}")
        print(f"  注意力权重形状: {attention_weights.shape}")
        print(f"  跨模态相关性: {attention_result['cross_modal_correlation']:.3f}")
    
    def test_audio_to_text_attention(self, mock_attention_mechanism):
        """测试音频到文本的注意力"""
        # 创建模拟特征
        audio_features = np.random.rand(100, 512)  # 100个时间步，每个512维
        text_features = np.random.rand(15, 512)    # 15个token，每个512维
        
        # 计算音频到文本的注意力
        attention_result = mock_attention_mechanism.apply_attention(
            query_modality="audio",
            key_modality="text",
            value_modality="text", 
            query_features=audio_features,
            key_features=text_features,
            value_features=text_features
        )
        
        # 验证结果
        assert attention_result["query_modality"] == "audio"
        assert attention_result["key_modality"] == "text"
        
        attention_weights = attention_result["attention_weights"]
        assert attention_weights.shape[1] == audio_features.shape[0]
        assert attention_weights.shape[2] == text_features.shape[0]
        
        print(f"音频到文本注意力:")
        print(f"  音频特征形状: {audio_features.shape}")
        print(f"  文本特征形状: {text_features.shape}")
        print(f"  跨模态相关性: {attention_result['cross_modal_correlation']:.3f}")
    
    def test_multimodal_self_attention(self, mock_attention_mechanism):
        """测试多模态自注意力"""
        # 创建多模态特征序列
        text_features = np.random.rand(5, 512)
        image_features = np.random.rand(3, 512)
        audio_features = np.random.rand(7, 512)
        
        # 拼接所有模态特征
        multimodal_features = np.concatenate([text_features, image_features, audio_features], axis=0)
        
        # 创建模态标记
        modality_labels = (["text"] * 5 + ["image"] * 3 + ["audio"] * 7)
        
        # 计算自注意力
        self_attention_result = mock_attention_mechanism.apply_attention(
            query_modality="multimodal",
            key_modality="multimodal",
            value_modality="multimodal",
            query_features=multimodal_features,
            key_features=multimodal_features,
            value_features=multimodal_features
        )
        
        # 分析注意力模式
        attention_weights = self_attention_result["attention_weights"]
        avg_attention = np.mean(attention_weights, axis=0)  # 平均所有头的注意力
        
        # 分析模态内和模态间的注意力
        text_indices = list(range(5))
        image_indices = list(range(5, 8))
        audio_indices = list(range(8, 15))
        
        # 计算模态内注意力强度
        text_self_attention = np.mean(avg_attention[np.ix_(text_indices, text_indices)])
        image_self_attention = np.mean(avg_attention[np.ix_(image_indices, image_indices)])
        audio_self_attention = np.mean(avg_attention[np.ix_(audio_indices, audio_indices)])
        
        # 计算模态间注意力强度
        text_image_attention = np.mean(avg_attention[np.ix_(text_indices, image_indices)])
        text_audio_attention = np.mean(avg_attention[np.ix_(text_indices, audio_indices)])
        image_audio_attention = np.mean(avg_attention[np.ix_(image_indices, audio_indices)])
        
        print(f"多模态自注意力分析:")
        print(f"  总特征序列长度: {multimodal_features.shape[0]}")
        print(f"  模态内注意力:")
        print(f"    文本自注意力: {text_self_attention:.3f}")
        print(f"    图像自注意力: {image_self_attention:.3f}")
        print(f"    音频自注意力: {audio_self_attention:.3f}")
        print(f"  模态间注意力:")
        print(f"    文本-图像: {text_image_attention:.3f}")
        print(f"    文本-音频: {text_audio_attention:.3f}")
        print(f"    图像-音频: {image_audio_attention:.3f}")
        
        # 验证注意力权重的合理性
        assert multimodal_features.shape[0] == len(modality_labels)
        assert attention_weights.shape[1] == attention_weights.shape[2] == multimodal_features.shape[0]


class TestModalityFusionStrategies:
    """模态融合策略测试"""
    
    @pytest.fixture
    def mock_fusion_strategies(self):
        """模拟融合策略"""
        class MockFusionStrategies:
            def __init__(self):
                self.strategies = {
                    "early_fusion": self.early_fusion,
                    "late_fusion": self.late_fusion,
                    "attention_fusion": self.attention_fusion,
                    "gated_fusion": self.gated_fusion,
                    "hierarchical_fusion": self.hierarchical_fusion
                }
            
            def early_fusion(self, modality_features):
                """早期融合：特征级拼接"""
                all_features = []
                modality_info = {}
                
                for modality, features in modality_features.items():
                    if features.ndim > 1:
                        # 如果是序列特征，取平均
                        global_features = np.mean(features, axis=0)
                    else:
                        global_features = features
                    
                    all_features.append(global_features)
                    modality_info[modality] = {
                        "feature_dim": len(global_features),
                        "start_idx": sum(len(f) for f in all_features[:-1]),
                        "end_idx": sum(len(f) for f in all_features)
                    }
                
                fused_features = np.concatenate(all_features)
                
                # 计算模态权重（基于特征维度）
                total_dim = len(fused_features)
                modality_weights = {
                    modality: info["feature_dim"] / total_dim
                    for modality, info in modality_info.items()
                }
                
                return FusionResult(
                    unified_representation=fused_features,
                    modality_weights=modality_weights,
                    confidence=0.8,
                    fusion_strategy="early_fusion",
                    processing_time=0.001,
                    metadata={"modality_info": modality_info}
                )
            
            def late_fusion(self, modality_features):
                """晚期融合：决策级融合"""
                modality_decisions = {}
                modality_confidences = {}
                
                # 为每个模态生成决策
                for modality, features in modality_features.items():
                    if features.ndim > 1:
                        features = np.mean(features, axis=0)
                    
                    # 模拟模态特定的决策
                    decision_score = np.tanh(np.mean(features))  # [-1, 1]
                    confidence = 1.0 / (1.0 + np.exp(-np.std(features)))  # sigmoid
                    
                    modality_decisions[modality] = decision_score
                    modality_confidences[modality] = confidence
                
                # 基于置信度的加权融合
                total_confidence = sum(modality_confidences.values())
                modality_weights = {
                    modality: conf / total_confidence
                    for modality, conf in modality_confidences.items()
                }
                
                # 计算最终决策
                final_decision = sum(
                    modality_weights[modality] * decision
                    for modality, decision in modality_decisions.items()
                )
                
                # 创建统一表示
                unified_representation = np.array([final_decision] + list(modality_decisions.values()))
                
                return FusionResult(
                    unified_representation=unified_representation,
                    modality_weights=modality_weights,
                    confidence=np.mean(list(modality_confidences.values())),
                    fusion_strategy="late_fusion",
                    processing_time=0.002,
                    metadata={
                        "modality_decisions": modality_decisions,
                        "modality_confidences": modality_confidences
                    }
                )
            
            def attention_fusion(self, modality_features):
                """注意力融合：基于注意力机制的融合"""
                modality_list = list(modality_features.keys())
                feature_list = []
                
                # 标准化所有特征到相同维度
                target_dim = 512
                for modality, features in modality_features.items():
                    if features.ndim > 1:
                        features = np.mean(features, axis=0)
                    
                    # 调整到目标维度
                    if len(features) > target_dim:
                        features = features[:target_dim]
                    elif len(features) < target_dim:
                        features = np.pad(features, (0, target_dim - len(features)))
                    
                    feature_list.append(features)
                
                feature_matrix = np.stack(feature_list)  # [num_modalities, feature_dim]
                
                # 计算注意力权重
                # 使用特征间的相似度作为注意力分数
                attention_scores = np.dot(feature_matrix, feature_matrix.T)  # [num_modalities, num_modalities]
                
                # 对每个模态计算其对所有模态的注意力
                attention_weights_matrix = []
                for i in range(len(modality_list)):
                    scores = attention_scores[i]
                    weights = np.exp(scores) / np.sum(np.exp(scores))
                    attention_weights_matrix.append(weights)
                
                attention_weights_matrix = np.array(attention_weights_matrix)
                
                # 应用注意力进行融合
                attended_features = np.dot(attention_weights_matrix, feature_matrix)
                
                # 最终融合：平均所有注意力加权的特征
                unified_representation = np.mean(attended_features, axis=0)
                
                # 计算模态权重（基于平均注意力权重）
                modality_weights = {
                    modality: np.mean(attention_weights_matrix[:, i])
                    for i, modality in enumerate(modality_list)
                }
                
                return FusionResult(
                    unified_representation=unified_representation,
                    modality_weights=modality_weights,
                    confidence=0.9,
                    fusion_strategy="attention_fusion",
                    processing_time=0.005,
                    metadata={
                        "attention_weights_matrix": attention_weights_matrix,
                        "modality_order": modality_list
                    }
                )
            
            def gated_fusion(self, modality_features):
                """门控融合：使用门控机制控制模态贡献"""
                modality_list = list(modality_features.keys())
                processed_features = {}
                
                # 处理每个模态的特征
                for modality, features in modality_features.items():
                    if features.ndim > 1:
                        features = np.mean(features, axis=0)
                    
                    # 标准化特征
                    normalized_features = (features - np.mean(features)) / (np.std(features) + 1e-8)
                    processed_features[modality] = normalized_features
                
                # 计算门控权重
                gate_inputs = []
                for modality in modality_list:
                    # 使用特征的统计信息作为门控输入
                    features = processed_features[modality]
                    gate_input = np.array([
                        np.mean(features),
                        np.std(features),
                        np.max(features),
                        np.min(features)
                    ])
                    gate_inputs.append(gate_input)
                
                gate_inputs = np.concatenate(gate_inputs)
                
                # 模拟门控网络
                gate_weights = np.random.rand(len(modality_list))
                gate_weights = gate_weights / np.sum(gate_weights)  # 归一化
                
                # 应用门控权重
                gated_features = []
                for i, modality in enumerate(modality_list):
                    gated_feature = processed_features[modality] * gate_weights[i]
                    gated_features.append(gated_feature)
                
                # 融合门控后的特征
                max_len = max(len(f) for f in gated_features)
                padded_features = []
                for features in gated_features:
                    if len(features) < max_len:
                        padded = np.pad(features, (0, max_len - len(features)))
                    else:
                        padded = features[:max_len]
                    padded_features.append(padded)
                
                unified_representation = np.sum(padded_features, axis=0)
                
                modality_weights = {
                    modality: gate_weights[i]
                    for i, modality in enumerate(modality_list)
                }
                
                return FusionResult(
                    unified_representation=unified_representation,
                    modality_weights=modality_weights,
                    confidence=0.85,
                    fusion_strategy="gated_fusion",
                    processing_time=0.003,
                    metadata={
                        "gate_weights": gate_weights,
                        "gate_inputs": gate_inputs
                    }
                )
            
            def hierarchical_fusion(self, modality_features):
                """分层融合：分层次融合不同模态"""
                # 第一层：相似模态融合
                text_audio_features = {}
                visual_features = {}
                
                for modality, features in modality_features.items():
                    if modality in ["text", "audio"]:
                        text_audio_features[modality] = features
                    else:  # image, video等视觉模态
                        visual_features[modality] = features
                
                # 融合文本和音频（语言相关模态）
                if text_audio_features:
                    linguistic_fusion = self.early_fusion(text_audio_features)
                    linguistic_repr = linguistic_fusion.unified_representation
                else:
                    linguistic_repr = np.array([])
                
                # 融合视觉模态
                if visual_features:
                    visual_fusion = self.early_fusion(visual_features)
                    visual_repr = visual_fusion.unified_representation
                else:
                    visual_repr = np.array([])
                
                # 第二层：跨模态融合
                if len(linguistic_repr) > 0 and len(visual_repr) > 0:
                    # 调整维度
                    min_dim = min(len(linguistic_repr), len(visual_repr))
                    linguistic_repr = linguistic_repr[:min_dim]
                    visual_repr = visual_repr[:min_dim]
                    
                    # 计算跨模态交互
                    interaction = linguistic_repr * visual_repr
                    unified_representation = np.concatenate([
                        linguistic_repr, visual_repr, interaction
                    ])
                    
                    # 计算权重
                    total_len = len(unified_representation)
                    modality_weights = {
                        "linguistic": len(linguistic_repr) / total_len,
                        "visual": len(visual_repr) / total_len,
                        "interaction": len(interaction) / total_len
                    }
                    
                elif len(linguistic_repr) > 0:
                    unified_representation = linguistic_repr
                    modality_weights = {"linguistic": 1.0}
                elif len(visual_repr) > 0:
                    unified_representation = visual_repr
                    modality_weights = {"visual": 1.0}
                else:
                    unified_representation = np.array([0.0])
                    modality_weights = {"empty": 1.0}
                
                return FusionResult(
                    unified_representation=unified_representation,
                    modality_weights=modality_weights,
                    confidence=0.88,
                    fusion_strategy="hierarchical_fusion",
                    processing_time=0.008,
                    metadata={
                        "linguistic_modalities": list(text_audio_features.keys()),
                        "visual_modalities": list(visual_features.keys())
                    }
                )
        
        return MockFusionStrategies()
    
    def test_early_fusion_strategy(self, mock_fusion_strategies):
        """测试早期融合策略"""
        # 准备测试数据
        modality_features = {
            "text": np.random.rand(768),
            "image": np.random.rand(2048),
            "audio": np.random.rand(512)
        }
        
        # 执行早期融合
        result = mock_fusion_strategies.early_fusion(modality_features)
        
        # 验证结果
        assert isinstance(result, FusionResult)
        assert result.fusion_strategy == "early_fusion"
        assert len(result.unified_representation) == 768 + 2048 + 512
        
        # 验证模态权重
        total_weight = sum(result.modality_weights.values())
        assert abs(total_weight - 1.0) < 1e-6
        
        # 验证权重与特征维度的关系
        assert result.modality_weights["image"] > result.modality_weights["text"]  # 图像特征维度更大
        assert result.modality_weights["text"] > result.modality_weights["audio"]
        
        print(f"早期融合结果:")
        print(f"  统一表示维度: {len(result.unified_representation)}")
        print(f"  模态权重: {result.modality_weights}")
        print(f"  置信度: {result.confidence}")
        print(f"  处理时间: {result.processing_time:.4f}s")
    
    def test_late_fusion_strategy(self, mock_fusion_strategies):
        """测试晚期融合策略"""
        modality_features = {
            "text": np.random.rand(768),
            "image": np.random.rand(2048), 
            "audio": np.random.rand(512)
        }
        
        result = mock_fusion_strategies.late_fusion(modality_features)
        
        # 验证结果
        assert result.fusion_strategy == "late_fusion"
        assert len(result.unified_representation) == 4  # 最终决策 + 3个模态决策
        
        # 验证模态权重基于置信度
        assert all(0 <= weight <= 1 for weight in result.modality_weights.values())
        assert abs(sum(result.modality_weights.values()) - 1.0) < 1e-6
        
        # 验证元数据
        assert "modality_decisions" in result.metadata
        assert "modality_confidences" in result.metadata
        
        print(f"晚期融合结果:")
        print(f"  模态决策: {result.metadata['modality_decisions']}")
        print(f"  模态置信度: {result.metadata['modality_confidences']}")
        print(f"  模态权重: {result.modality_weights}")
    
    def test_attention_fusion_strategy(self, mock_fusion_strategies):
        """测试注意力融合策略"""
        modality_features = {
            "text": np.random.rand(768),
            "image": np.random.rand(2048),
            "audio": np.random.rand(512)
        }
        
        result = mock_fusion_strategies.attention_fusion(modality_features)
        
        # 验证结果
        assert result.fusion_strategy == "attention_fusion"
        assert len(result.unified_representation) == 512  # 目标维度
        
        # 验证注意力权重
        assert "attention_weights_matrix" in result.metadata
        attention_matrix = result.metadata["attention_weights_matrix"]
        assert attention_matrix.shape == (3, 3)  # 3个模态
        
        # 验证注意力权重归一化
        for i in range(attention_matrix.shape[0]):
            weights_sum = np.sum(attention_matrix[i])
            assert abs(weights_sum - 1.0) < 1e-6
        
        print(f"注意力融合结果:")
        print(f"  注意力权重矩阵形状: {attention_matrix.shape}")
        print(f"  模态权重: {result.modality_weights}")
        print(f"  置信度: {result.confidence}")
    
    def test_fusion_strategy_comparison(self, mock_fusion_strategies):
        """测试不同融合策略的比较"""
        # 使用相同的输入测试所有策略
        modality_features = {
            "text": np.random.rand(100),
            "image": np.random.rand(200),
            "audio": np.random.rand(150)
        }
        
        strategies = ["early_fusion", "late_fusion", "attention_fusion", "gated_fusion", "hierarchical_fusion"]
        results = {}
        
        for strategy_name in strategies:
            strategy_func = mock_fusion_strategies.strategies[strategy_name]
            result = strategy_func(modality_features)
            results[strategy_name] = result
        
        # 比较不同策略的结果
        print("融合策略比较:")
        print(f"{'策略':<20} {'表示维度':<10} {'置信度':<8} {'处理时间':<10}")
        print("-" * 50)
        
        for strategy_name, result in results.items():
            print(f"{strategy_name:<20} {len(result.unified_representation):<10} "
                  f"{result.confidence:<8.3f} {result.processing_time:<10.4f}")
        
        # 验证所有策略都产生了有效结果
        for strategy_name, result in results.items():
            assert isinstance(result, FusionResult)
            assert len(result.unified_representation) > 0
            assert 0 <= result.confidence <= 1
            assert result.processing_time > 0
            assert sum(result.modality_weights.values()) > 0.99  # 允许小的数值误差


class TestMultimodalPerformanceOptimization:
    """多模态性能优化测试"""
    
    @pytest.mark.asyncio
    async def test_parallel_modality_processing(self):
        """测试并行模态处理"""
        class ParallelMultimodalProcessor:
            def __init__(self):
                self.processing_times = {}
            
            async def process_text(self, text_data):
                """处理文本模态"""
                start_time = time.time()
                await asyncio.sleep(0.1)  # 模拟文本处理时间
                processing_time = time.time() - start_time
                self.processing_times["text"] = processing_time
                
                return {
                    "modality": "text",
                    "features": np.random.rand(768),
                    "processing_time": processing_time
                }
            
            async def process_image(self, image_data):
                """处理图像模态"""
                start_time = time.time()
                await asyncio.sleep(0.15)  # 模拟图像处理时间
                processing_time = time.time() - start_time
                self.processing_times["image"] = processing_time
                
                return {
                    "modality": "image",
                    "features": np.random.rand(2048),
                    "processing_time": processing_time
                }
            
            async def process_audio(self, audio_data):
                """处理音频模态"""
                start_time = time.time()
                await asyncio.sleep(0.12)  # 模拟音频处理时间
                processing_time = time.time() - start_time
                self.processing_times["audio"] = processing_time
                
                return {
                    "modality": "audio", 
                    "features": np.random.rand(512),
                    "processing_time": processing_time
                }
            
            async def process_multimodal_parallel(self, multimodal_data):
                """并行处理多模态数据"""
                total_start_time = time.time()
                
                # 创建并行任务
                tasks = []
                if "text" in multimodal_data:
                    tasks.append(self.process_text(multimodal_data["text"]))
                if "image" in multimodal_data:
                    tasks.append(self.process_image(multimodal_data["image"]))
                if "audio" in multimodal_data:
                    tasks.append(self.process_audio(multimodal_data["audio"]))
                
                # 并行执行所有任务
                results = await asyncio.gather(*tasks)
                
                total_processing_time = time.time() - total_start_time
                
                return {
                    "results": results,
                    "total_processing_time": total_processing_time,
                    "individual_times": self.processing_times,
                    "parallel_efficiency": sum(self.processing_times.values()) / total_processing_time
                }
            
            async def process_multimodal_sequential(self, multimodal_data):
                """顺序处理多模态数据"""
                total_start_time = time.time()
                results = []
                
                if "text" in multimodal_data:
                    result = await self.process_text(multimodal_data["text"])
                    results.append(result)
                
                if "image" in multimodal_data:
                    result = await self.process_image(multimodal_data["image"])
                    results.append(result)
                
                if "audio" in multimodal_data:
                    result = await self.process_audio(multimodal_data["audio"])
                    results.append(result)
                
                total_processing_time = time.time() - total_start_time
                
                return {
                    "results": results,
                    "total_processing_time": total_processing_time,
                    "individual_times": self.processing_times
                }
        
        # 测试并行和顺序处理
        processor = ParallelMultimodalProcessor()
        
        test_data = {
            "text": "这是测试文本",
            "image": b"fake_image_data",
            "audio": b"fake_audio_data"
        }
        
        # 并行处理
        parallel_result = await processor.process_multimodal_parallel(test_data)
        
        # 重置处理时间记录
        processor.processing_times = {}
        
        # 顺序处理
        sequential_result = await processor.process_multimodal_sequential(test_data)
        
        # 比较性能
        parallel_time = parallel_result["total_processing_time"]
        sequential_time = sequential_result["total_processing_time"]
        speedup = sequential_time / parallel_time
        
        print(f"多模态处理性能比较:")
        print(f"  并行处理时间: {parallel_time:.3f}s")
        print(f"  顺序处理时间: {sequential_time:.3f}s")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  并行效率: {parallel_result['parallel_efficiency']:.2f}")
        
        # 验证并行处理的优势
        assert parallel_time < sequential_time, "并行处理应该比顺序处理更快"
        assert speedup > 1.5, f"加速比应该显著: {speedup:.2f}x"
        assert parallel_result["parallel_efficiency"] > 2.0, "并行效率应该大于2"
    
    def test_modality_caching_optimization(self):
        """测试模态缓存优化"""
        class CachedMultimodalProcessor:
            def __init__(self, cache_size=100):
                self.feature_cache = {}
                self.cache_size = cache_size
                self.cache_hits = 0
                self.cache_misses = 0
                self.processing_count = 0
            
            def _get_cache_key(self, modality, data):
                """生成缓存键"""
                if isinstance(data, str):
                    return f"{modality}:{hash(data)}"
                elif isinstance(data, bytes):
                    return f"{modality}:{hash(data)}"
                else:
                    return f"{modality}:{hash(str(data))}"
            
            def _evict_cache_if_needed(self):
                """缓存满时执行LRU淘汰"""
                if len(self.feature_cache) >= self.cache_size:
                    # 简单的FIFO淘汰策略
                    oldest_key = next(iter(self.feature_cache))
                    del self.feature_cache[oldest_key]
            
            def extract_features(self, modality, data):
                """提取特征（带缓存）"""
                cache_key = self._get_cache_key(modality, data)
                
                # 检查缓存
                if cache_key in self.feature_cache:
                    self.cache_hits += 1
                    cached_result = self.feature_cache[cache_key]
                    return {
                        **cached_result,
                        "from_cache": True,
                        "cache_key": cache_key
                    }
                
                # 缓存未命中，计算特征
                self.cache_misses += 1
                self.processing_count += 1
                
                # 模拟特征提取
                if modality == "text":
                    features = np.random.rand(768)
                    processing_time = 0.05
                elif modality == "image":
                    features = np.random.rand(2048)
                    processing_time = 0.1
                elif modality == "audio":
                    features = np.random.rand(512)
                    processing_time = 0.08
                else:
                    features = np.random.rand(256)
                    processing_time = 0.03
                
                time.sleep(processing_time)  # 模拟处理延迟
                
                result = {
                    "modality": modality,
                    "features": features,
                    "processing_time": processing_time,
                    "from_cache": False
                }
                
                # 存储到缓存
                self._evict_cache_if_needed()
                self.feature_cache[cache_key] = {
                    "modality": modality,
                    "features": features,
                    "processing_time": processing_time
                }
                
                return result
            
            def get_cache_stats(self):
                """获取缓存统计"""
                total_requests = self.cache_hits + self.cache_misses
                hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0
                
                return {
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "hit_rate": hit_rate,
                    "cache_size": len(self.feature_cache),
                    "processing_count": self.processing_count
                }
        
        # 测试缓存效果
        processor = CachedMultimodalProcessor(cache_size=10)
        
        # 准备测试数据
        test_data = [
            ("text", "这是第一个文本"),
            ("text", "这是第二个文本"),
            ("text", "这是第一个文本"),  # 重复，应该命中缓存
            ("image", b"image_data_1"),
            ("image", b"image_data_2"),
            ("image", b"image_data_1"),  # 重复，应该命中缓存
            ("audio", b"audio_data_1"),
            ("text", "这是第一个文本"),  # 再次重复
        ]
        
        # 执行特征提取
        results = []
        total_processing_time = 0
        
        for modality, data in test_data:
            start_time = time.time()
            result = processor.extract_features(modality, data)
            actual_time = time.time() - start_time
            
            results.append({
                **result,
                "actual_processing_time": actual_time
            })
            total_processing_time += actual_time
        
        # 分析缓存效果
        cache_stats = processor.get_cache_stats()
        
        cached_results = [r for r in results if r["from_cache"]]
        non_cached_results = [r for r in results if not r["from_cache"]]
        
        avg_cached_time = np.mean([r["actual_processing_time"] for r in cached_results])
        avg_non_cached_time = np.mean([r["actual_processing_time"] for r in non_cached_results])
        
        print(f"模态缓存优化结果:")
        print(f"  总请求数: {len(test_data)}")
        print(f"  缓存命中数: {cache_stats['cache_hits']}")
        print(f"  缓存未命中数: {cache_stats['cache_misses']}")
        print(f"  缓存命中率: {cache_stats['hit_rate']:.2%}")
        print(f"  实际处理次数: {cache_stats['processing_count']}")
        print(f"  平均缓存访问时间: {avg_cached_time:.4f}s")
        print(f"  平均非缓存处理时间: {avg_non_cached_time:.4f}s")
        print(f"  缓存加速比: {avg_non_cached_time / avg_cached_time:.2f}x")
        
        # 验证缓存效果
        assert cache_stats["hit_rate"] > 0.3, f"缓存命中率应该大于30%: {cache_stats['hit_rate']:.2%}"
        assert cache_stats["processing_count"] < len(test_data), "应该有缓存命中减少处理次数"
        assert avg_cached_time < avg_non_cached_time, "缓存访问应该比重新处理更快"
    
    def test_adaptive_fusion_optimization(self):
        """测试自适应融合优化"""
        class AdaptiveFusionOptimizer:
            def __init__(self):
                self.fusion_history = []
                self.performance_metrics = {}
            
            def evaluate_fusion_quality(self, fusion_result, ground_truth=None):
                """评估融合质量"""
                # 模拟质量评估指标
                representation = fusion_result.unified_representation
                
                # 1. 表示质量：基于特征分布
                feature_variance = np.var(representation)
                feature_sparsity = np.sum(np.abs(representation) < 0.1) / len(representation)
                
                # 2. 模态平衡性：检查模态权重分布
                weights = list(fusion_result.modality_weights.values())
                weight_entropy = -np.sum([w * np.log(w + 1e-8) for w in weights])
                max_entropy = np.log(len(weights))
                weight_balance = weight_entropy / max_entropy if max_entropy > 0 else 0
                
                # 3. 融合一致性：基于置信度
                confidence_score = fusion_result.confidence
                
                # 综合质量分数
                quality_score = (
                    0.3 * min(feature_variance, 1.0) +  # 特征丰富度
                    0.3 * (1.0 - feature_sparsity) +    # 特征密度
                    0.2 * weight_balance +               # 权重平衡
                    0.2 * confidence_score               # 置信度
                )
                
                return {
                    "quality_score": quality_score,
                    "feature_variance": feature_variance,
                    "feature_sparsity": feature_sparsity,
                    "weight_balance": weight_balance,
                    "confidence_score": confidence_score
                }
            
            def select_optimal_fusion_strategy(self, modality_features, available_strategies):
                """选择最优融合策略"""
                strategy_scores = {}
                
                # 分析输入特征特性
                modality_count = len(modality_features)
                feature_dims = {mod: len(feat) if feat.ndim == 1 else feat.shape[-1] 
                              for mod, feat in modality_features.items()}
                dim_variance = np.var(list(feature_dims.values()))
                
                # 基于历史性能选择策略
                for strategy in available_strategies:
                    base_score = 0.5  # 基础分数
                    
                    # 根据模态数量调整
                    if strategy == "early_fusion" and modality_count <= 2:
                        base_score += 0.2
                    elif strategy == "attention_fusion" and modality_count >= 3:
                        base_score += 0.3
                    elif strategy == "hierarchical_fusion" and modality_count >= 4:
                        base_score += 0.25
                    
                    # 根据特征维度差异调整
                    if strategy == "gated_fusion" and dim_variance > 1000:
                        base_score += 0.15
                    elif strategy == "early_fusion" and dim_variance < 100:
                        base_score += 0.1
                    
                    # 根据历史性能调整
                    if strategy in self.performance_metrics:
                        historical_score = self.performance_metrics[strategy]["avg_quality"]
                        base_score = 0.7 * base_score + 0.3 * historical_score
                    
                    strategy_scores[strategy] = base_score
                
                # 选择最高分数的策略
                optimal_strategy = max(strategy_scores, key=strategy_scores.get)
                
                return {
                    "selected_strategy": optimal_strategy,
                    "strategy_scores": strategy_scores,
                    "selection_reason": {
                        "modality_count": modality_count,
                        "dim_variance": dim_variance,
                        "feature_dims": feature_dims
                    }
                }
            
            def update_performance_metrics(self, strategy, quality_evaluation):
                """更新性能指标"""
                if strategy not in self.performance_metrics:
                    self.performance_metrics[strategy] = {
                        "quality_scores": [],
                        "avg_quality": 0.0,
                        "usage_count": 0
                    }
                
                metrics = self.performance_metrics[strategy]
                metrics["quality_scores"].append(quality_evaluation["quality_score"])
                metrics["usage_count"] += 1
                metrics["avg_quality"] = np.mean(metrics["quality_scores"])
                
                # 保持最近100次记录
                if len(metrics["quality_scores"]) > 100:
                    metrics["quality_scores"] = metrics["quality_scores"][-100:]
                    metrics["avg_quality"] = np.mean(metrics["quality_scores"])
        
        # 测试自适应融合优化
        optimizer = AdaptiveFusionOptimizer()
        
        # 模拟融合策略
        def mock_fusion(strategy, features):
            if strategy == "early_fusion":
                unified = np.concatenate([f.flatten() for f in features.values()])
                weights = {mod: len(f.flatten()) / len(unified) for mod, f in features.items()}
            elif strategy == "attention_fusion":
                unified = np.random.rand(512)
                weights = {mod: np.random.rand() for mod in features.keys()}
                total = sum(weights.values())
                weights = {mod: w/total for mod, w in weights.items()}
            else:  # late_fusion
                unified = np.random.rand(256)
                weights = {mod: 1.0/len(features) for mod in features.keys()}
            
            return FusionResult(
                unified_representation=unified,
                modality_weights=weights,
                confidence=np.random.rand() * 0.3 + 0.7,
                fusion_strategy=strategy,
                processing_time=np.random.rand() * 0.01
            )
        
        # 测试不同场景
        test_scenarios = [
            # 场景1：两个相似维度的模态
            {
                "name": "两模态相似维度",
                "features": {
                    "text": np.random.rand(512),
                    "audio": np.random.rand(480)
                }
            },
            # 场景2：多个不同维度的模态
            {
                "name": "多模态不同维度",
                "features": {
                    "text": np.random.rand(768),
                    "image": np.random.rand(2048),
                    "audio": np.random.rand(512),
                    "video": np.random.rand(1024)
                }
            },
            # 场景3：维度差异很大的模态
            {
                "name": "大维度差异",
                "features": {
                    "text": np.random.rand(100),
                    "image": np.random.rand(4096)
                }
            }
        ]
        
        available_strategies = ["early_fusion", "attention_fusion", "late_fusion"]
        
        print("自适应融合优化测试:")
        
        for scenario in test_scenarios:
            print(f"\n场景: {scenario['name']}")
            features = scenario["features"]
            
            # 选择最优策略
            selection_result = optimizer.select_optimal_fusion_strategy(
                features, available_strategies
            )
            
            selected_strategy = selection_result["selected_strategy"]
            print(f"  选择策略: {selected_strategy}")
            print(f"  策略分数: {selection_result['strategy_scores']}")
            
            # 执行融合
            fusion_result = mock_fusion(selected_strategy, features)
            
            # 评估质量
            quality_eval = optimizer.evaluate_fusion_quality(fusion_result)
            print(f"  融合质量: {quality_eval['quality_score']:.3f}")
            print(f"  权重平衡: {quality_eval['weight_balance']:.3f}")
            
            # 更新性能指标
            optimizer.update_performance_metrics(selected_strategy, quality_eval)
        
        # 显示最终性能统计
        print(f"\n策略性能统计:")
        for strategy, metrics in optimizer.performance_metrics.items():
            print(f"  {strategy}:")
            print(f"    平均质量: {metrics['avg_quality']:.3f}")
            print(f"    使用次数: {metrics['usage_count']}")
        
        # 验证自适应优化效果
        assert len(optimizer.performance_metrics) > 0, "应该有性能指标记录"
        
        # 验证不同场景选择了合适的策略
        for strategy, metrics in optimizer.performance_metrics.items():
            assert metrics["avg_quality"] > 0.3, f"{strategy}的平均质量应该合理"
            assert metrics["usage_count"] > 0, f"{strategy}应该被使用过"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
