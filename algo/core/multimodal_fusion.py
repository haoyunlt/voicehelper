"""
多模态融合架构 - v1.8.0
统一处理文本、图像、语音等多种模态信息的融合系统
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    STRUCTURED = "structured"  # 结构化数据

class FusionStrategy(Enum):
    """融合策略"""
    EARLY_FUSION = "early"      # 早期融合
    LATE_FUSION = "late"        # 后期融合
    HYBRID_FUSION = "hybrid"    # 混合融合
    ATTENTION_FUSION = "attention"  # 注意力融合
    CROSS_MODAL = "cross_modal"     # 跨模态融合

@dataclass
class ModalityInput:
    """模态输入"""
    modality: ModalityType
    data: Any
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class FusionResult:
    """融合结果"""
    fused_features: np.ndarray
    modality_weights: Dict[ModalityType, float]
    attention_scores: Dict[str, float] = field(default_factory=dict)
    fusion_confidence: float = 0.0
    processing_time: float = 0.0
    strategy_used: FusionStrategy = FusionStrategy.HYBRID_FUSION

class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self):
        self.feature_dims = {
            ModalityType.TEXT: 768,      # BERT-like embedding
            ModalityType.IMAGE: 2048,    # ResNet-like features
            ModalityType.AUDIO: 512,     # Audio features
            ModalityType.VIDEO: 1024,    # Video features
            ModalityType.STRUCTURED: 256  # Structured data features
        }
    
    async def extract_text_features(self, text_data: str) -> np.ndarray:
        """提取文本特征"""
        try:
            # 模拟BERT特征提取
            # 实际应使用真实的预训练模型
            words = text_data.split()
            
            # 简单的词向量模拟
            features = []
            for word in words[:50]:  # 限制长度
                # 基于词的hash生成特征
                word_hash = hash(word) % 1000
                word_feature = np.random.normal(0, 1, 16)  # 16维词向量
                word_feature[word_hash % 16] += 2  # 增强特定维度
                features.append(word_feature)
            
            if not features:
                features = [np.zeros(16)]
            
            # 池化操作
            text_features = np.mean(features, axis=0)
            
            # 扩展到目标维度
            if len(text_features) < self.feature_dims[ModalityType.TEXT]:
                padding = np.zeros(self.feature_dims[ModalityType.TEXT] - len(text_features))
                text_features = np.concatenate([text_features, padding])
            else:
                text_features = text_features[:self.feature_dims[ModalityType.TEXT]]
            
            return text_features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Text feature extraction error: {e}")
            return np.zeros(self.feature_dims[ModalityType.TEXT], dtype=np.float32)
    
    async def extract_image_features(self, image_data: Any) -> np.ndarray:
        """提取图像特征"""
        try:
            # 模拟ResNet特征提取
            # 实际应使用真实的CNN模型
            
            if hasattr(image_data, 'size'):
                # PIL Image
                width, height = image_data.size
                channels = len(image_data.getbands())
            elif isinstance(image_data, np.ndarray):
                height, width = image_data.shape[:2]
                channels = image_data.shape[2] if len(image_data.shape) > 2 else 1
            else:
                # 默认值
                width, height, channels = 224, 224, 3
            
            # 基于图像属性生成特征
            aspect_ratio = width / height
            size_feature = np.log(width * height + 1)
            
            # 模拟卷积特征
            conv_features = np.random.normal(0, 1, self.feature_dims[ModalityType.IMAGE] - 10)
            
            # 添加结构化特征
            structural_features = np.array([
                aspect_ratio, size_feature, channels,
                width / 1000, height / 1000,  # 归一化尺寸
                np.sin(aspect_ratio), np.cos(aspect_ratio),  # 三角特征
                np.log(channels + 1), np.sqrt(width), np.sqrt(height)
            ])
            
            image_features = np.concatenate([conv_features, structural_features])
            
            return image_features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Image feature extraction error: {e}")
            return np.zeros(self.feature_dims[ModalityType.IMAGE], dtype=np.float32)
    
    async def extract_audio_features(self, audio_data: bytes) -> np.ndarray:
        """提取音频特征"""
        try:
            # 模拟音频特征提取
            # 实际应使用MFCC、Mel频谱等特征
            
            audio_length = len(audio_data)
            
            # 基本统计特征
            basic_features = np.array([
                audio_length / 1000,  # 归一化长度
                np.log(audio_length + 1),
                audio_length % 100 / 100,  # 周期性特征
            ])
            
            # 模拟频谱特征
            spectral_features = np.random.normal(0, 1, 100)
            
            # 模拟MFCC特征
            mfcc_features = np.random.normal(0, 1, 39)  # 13 MFCC + delta + delta-delta
            
            # 模拟节奏特征
            rhythm_features = np.random.normal(0, 1, 20)
            
            # 组合特征
            audio_features = np.concatenate([
                basic_features, spectral_features, mfcc_features, rhythm_features
            ])
            
            # 调整到目标维度
            if len(audio_features) < self.feature_dims[ModalityType.AUDIO]:
                padding = np.zeros(self.feature_dims[ModalityType.AUDIO] - len(audio_features))
                audio_features = np.concatenate([audio_features, padding])
            else:
                audio_features = audio_features[:self.feature_dims[ModalityType.AUDIO]]
            
            return audio_features.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio feature extraction error: {e}")
            return np.zeros(self.feature_dims[ModalityType.AUDIO], dtype=np.float32)
    
    async def extract_structured_features(self, structured_data: Dict[str, Any]) -> np.ndarray:
        """提取结构化数据特征"""
        try:
            features = []
            
            for key, value in structured_data.items():
                if isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, str):
                    # 字符串哈希特征
                    features.append(hash(value) % 1000 / 1000)
                elif isinstance(value, bool):
                    features.append(1.0 if value else 0.0)
                elif isinstance(value, list):
                    features.append(len(value))
                else:
                    features.append(0.0)
            
            # 填充或截断到目标维度
            target_dim = self.feature_dims[ModalityType.STRUCTURED]
            if len(features) < target_dim:
                features.extend([0.0] * (target_dim - len(features)))
            else:
                features = features[:target_dim]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Structured feature extraction error: {e}")
            return np.zeros(self.feature_dims[ModalityType.STRUCTURED], dtype=np.float32)

class AttentionMechanism:
    """注意力机制"""
    
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        
        # 模拟注意力权重矩阵
        self.query_weights = np.random.normal(0, 0.1, (feature_dim, feature_dim))
        self.key_weights = np.random.normal(0, 0.1, (feature_dim, feature_dim))
        self.value_weights = np.random.normal(0, 0.1, (feature_dim, feature_dim))
    
    async def compute_attention(self, 
                              features: Dict[ModalityType, np.ndarray],
                              query_modality: Optional[ModalityType] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        计算跨模态注意力
        
        Args:
            features: 各模态特征
            query_modality: 查询模态（如果指定，以该模态为主导）
            
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: (融合特征, 注意力分数)
        """
        try:
            if not features:
                return np.zeros(self.feature_dim), {}
            
            # 标准化特征维度
            normalized_features = {}
            for modality, feature in features.items():
                if len(feature) > self.feature_dim:
                    normalized_features[modality] = feature[:self.feature_dim]
                else:
                    padding = np.zeros(self.feature_dim - len(feature))
                    normalized_features[modality] = np.concatenate([feature, padding])
            
            # 计算查询、键、值
            queries = {}
            keys = {}
            values = {}
            
            for modality, feature in normalized_features.items():
                queries[modality] = np.dot(feature, self.query_weights)
                keys[modality] = np.dot(feature, self.key_weights)
                values[modality] = np.dot(feature, self.value_weights)
            
            # 计算注意力分数
            attention_scores = {}
            modality_list = list(normalized_features.keys())
            
            if query_modality and query_modality in queries:
                # 以指定模态为查询
                query = queries[query_modality]
                for modality in modality_list:
                    key = keys[modality]
                    score = np.dot(query, key) / np.sqrt(self.feature_dim)
                    attention_scores[f"{query_modality.value}_to_{modality.value}"] = float(score)
            else:
                # 计算所有模态间的注意力
                for i, mod1 in enumerate(modality_list):
                    for j, mod2 in enumerate(modality_list):
                        query = queries[mod1]
                        key = keys[mod2]
                        score = np.dot(query, key) / np.sqrt(self.feature_dim)
                        attention_scores[f"{mod1.value}_to_{mod2.value}"] = float(score)
            
            # 应用softmax归一化
            score_values = list(attention_scores.values())
            if score_values:
                exp_scores = np.exp(score_values - np.max(score_values))
                softmax_scores = exp_scores / np.sum(exp_scores)
                
                for i, key in enumerate(attention_scores.keys()):
                    attention_scores[key] = float(softmax_scores[i])
            
            # 计算加权融合特征
            fused_feature = np.zeros(self.feature_dim)
            total_weight = 0
            
            for modality, value in values.items():
                # 计算该模态的平均注意力权重
                modality_weights = [
                    score for key, score in attention_scores.items()
                    if key.endswith(f"_to_{modality.value}")
                ]
                
                if modality_weights:
                    avg_weight = np.mean(modality_weights)
                    fused_feature += value * avg_weight
                    total_weight += avg_weight
            
            if total_weight > 0:
                fused_feature /= total_weight
            
            return fused_feature, attention_scores
            
        except Exception as e:
            logger.error(f"Attention computation error: {e}")
            return np.zeros(self.feature_dim), {}

class CrossModalAligner:
    """跨模态对齐器"""
    
    def __init__(self):
        self.alignment_matrices = {}
        self.semantic_space_dim = 256
    
    async def align_modalities(self, 
                             features: Dict[ModalityType, np.ndarray]) -> Dict[ModalityType, np.ndarray]:
        """
        对齐不同模态到统一语义空间
        
        Args:
            features: 各模态特征
            
        Returns:
            Dict[ModalityType, np.ndarray]: 对齐后的特征
        """
        try:
            aligned_features = {}
            
            for modality, feature in features.items():
                # 获取或创建对齐矩阵
                if modality not in self.alignment_matrices:
                    input_dim = len(feature)
                    self.alignment_matrices[modality] = np.random.normal(
                        0, 0.1, (input_dim, self.semantic_space_dim)
                    )
                
                # 线性变换到语义空间
                alignment_matrix = self.alignment_matrices[modality]
                if len(feature) == alignment_matrix.shape[0]:
                    aligned_feature = np.dot(feature, alignment_matrix)
                else:
                    # 调整特征维度
                    if len(feature) > alignment_matrix.shape[0]:
                        feature = feature[:alignment_matrix.shape[0]]
                    else:
                        padding = np.zeros(alignment_matrix.shape[0] - len(feature))
                        feature = np.concatenate([feature, padding])
                    aligned_feature = np.dot(feature, alignment_matrix)
                
                # L2归一化
                norm = np.linalg.norm(aligned_feature)
                if norm > 0:
                    aligned_feature = aligned_feature / norm
                
                aligned_features[modality] = aligned_feature
            
            return aligned_features
            
        except Exception as e:
            logger.error(f"Modal alignment error: {e}")
            return features

class MultiModalFusion:
    """多模态融合主类"""
    
    def __init__(self, fusion_strategy: FusionStrategy = FusionStrategy.HYBRID_FUSION):
        self.fusion_strategy = fusion_strategy
        self.feature_extractor = FeatureExtractor()
        self.attention_mechanism = AttentionMechanism()
        self.cross_modal_aligner = CrossModalAligner()
        
        # 模态权重（可学习）
        self.modality_weights = {
            ModalityType.TEXT: 0.4,
            ModalityType.IMAGE: 0.3,
            ModalityType.AUDIO: 0.2,
            ModalityType.VIDEO: 0.05,
            ModalityType.STRUCTURED: 0.05
        }
        
        # 融合历史
        self.fusion_history = []
    
    async def fuse_modalities(self, 
                            modality_inputs: List[ModalityInput],
                            query_context: Optional[str] = None) -> FusionResult:
        """
        融合多模态信息
        
        Args:
            modality_inputs: 模态输入列表
            query_context: 查询上下文（可选）
            
        Returns:
            FusionResult: 融合结果
        """
        start_time = time.time()
        
        try:
            if not modality_inputs:
                return FusionResult(
                    fused_features=np.zeros(256),
                    modality_weights={},
                    processing_time=time.time() - start_time
                )
            
            # 1. 特征提取
            extracted_features = {}
            for modal_input in modality_inputs:
                if modal_input.features is not None:
                    # 使用预提取的特征
                    extracted_features[modal_input.modality] = modal_input.features
                else:
                    # 提取特征
                    feature = await self._extract_features(modal_input)
                    extracted_features[modal_input.modality] = feature
            
            # 2. 跨模态对齐
            aligned_features = await self.cross_modal_aligner.align_modalities(extracted_features)
            
            # 3. 根据策略进行融合
            if self.fusion_strategy == FusionStrategy.EARLY_FUSION:
                fused_features, attention_scores = await self._early_fusion(aligned_features)
            elif self.fusion_strategy == FusionStrategy.LATE_FUSION:
                fused_features, attention_scores = await self._late_fusion(aligned_features, modality_inputs)
            elif self.fusion_strategy == FusionStrategy.ATTENTION_FUSION:
                fused_features, attention_scores = await self._attention_fusion(aligned_features, query_context)
            elif self.fusion_strategy == FusionStrategy.CROSS_MODAL:
                fused_features, attention_scores = await self._cross_modal_fusion(aligned_features)
            else:  # HYBRID_FUSION
                fused_features, attention_scores = await self._hybrid_fusion(aligned_features, modality_inputs, query_context)
            
            # 4. 计算融合置信度
            fusion_confidence = self._calculate_fusion_confidence(modality_inputs, attention_scores)
            
            # 5. 更新模态权重
            updated_weights = self._update_modality_weights(modality_inputs, attention_scores)
            
            processing_time = time.time() - start_time
            
            # 记录融合历史
            self.fusion_history.append({
                'timestamp': time.time(),
                'modalities': [inp.modality.value for inp in modality_inputs],
                'strategy': self.fusion_strategy.value,
                'confidence': fusion_confidence,
                'processing_time': processing_time
            })
            
            # 保持历史记录在合理范围内
            if len(self.fusion_history) > 1000:
                self.fusion_history = self.fusion_history[-500:]
            
            return FusionResult(
                fused_features=fused_features,
                modality_weights=updated_weights,
                attention_scores=attention_scores,
                fusion_confidence=fusion_confidence,
                processing_time=processing_time,
                strategy_used=self.fusion_strategy
            )
            
        except Exception as e:
            logger.error(f"Multimodal fusion error: {e}")
            return FusionResult(
                fused_features=np.zeros(256),
                modality_weights={},
                processing_time=time.time() - start_time
            )
    
    async def _extract_features(self, modal_input: ModalityInput) -> np.ndarray:
        """提取单个模态的特征"""
        try:
            if modal_input.modality == ModalityType.TEXT:
                return await self.feature_extractor.extract_text_features(modal_input.data)
            elif modal_input.modality == ModalityType.IMAGE:
                return await self.feature_extractor.extract_image_features(modal_input.data)
            elif modal_input.modality == ModalityType.AUDIO:
                return await self.feature_extractor.extract_audio_features(modal_input.data)
            elif modal_input.modality == ModalityType.STRUCTURED:
                return await self.feature_extractor.extract_structured_features(modal_input.data)
            else:
                logger.warning(f"Unsupported modality: {modal_input.modality}")
                return np.zeros(256)
                
        except Exception as e:
            logger.error(f"Feature extraction error for {modal_input.modality}: {e}")
            return np.zeros(256)
    
    async def _early_fusion(self, features: Dict[ModalityType, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
        """早期融合：直接连接特征"""
        try:
            concatenated_features = []
            attention_scores = {}
            
            for modality, feature in features.items():
                weight = self.modality_weights.get(modality, 0.1)
                weighted_feature = feature * weight
                concatenated_features.append(weighted_feature)
                attention_scores[f"early_{modality.value}"] = weight
            
            if concatenated_features:
                fused = np.concatenate(concatenated_features)
                # 降维到统一大小
                if len(fused) > 256:
                    fused = fused[:256]
                else:
                    padding = np.zeros(256 - len(fused))
                    fused = np.concatenate([fused, padding])
            else:
                fused = np.zeros(256)
            
            return fused, attention_scores
            
        except Exception as e:
            logger.error(f"Early fusion error: {e}")
            return np.zeros(256), {}
    
    async def _late_fusion(self, 
                         features: Dict[ModalityType, np.ndarray],
                         modality_inputs: List[ModalityInput]) -> Tuple[np.ndarray, Dict[str, float]]:
        """后期融合：分别处理后再融合"""
        try:
            processed_features = {}
            attention_scores = {}
            
            # 对每个模态分别处理
            for modality, feature in features.items():
                # 模拟模态特定的处理
                processed = self._process_modality_specific(feature, modality)
                processed_features[modality] = processed
                
                # 基于置信度计算权重
                confidence = next(
                    (inp.confidence for inp in modality_inputs if inp.modality == modality),
                    1.0
                )
                attention_scores[f"late_{modality.value}"] = confidence
            
            # 加权平均融合
            fused = np.zeros(256)
            total_weight = 0
            
            for modality, processed in processed_features.items():
                weight = attention_scores[f"late_{modality.value}"]
                if len(processed) >= 256:
                    fused += processed[:256] * weight
                else:
                    padded = np.zeros(256)
                    padded[:len(processed)] = processed
                    fused += padded * weight
                total_weight += weight
            
            if total_weight > 0:
                fused /= total_weight
            
            return fused, attention_scores
            
        except Exception as e:
            logger.error(f"Late fusion error: {e}")
            return np.zeros(256), {}
    
    async def _attention_fusion(self, 
                              features: Dict[ModalityType, np.ndarray],
                              query_context: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """注意力融合：使用注意力机制"""
        try:
            # 确定查询模态
            query_modality = None
            if query_context:
                # 简单的查询模态推断
                if any(word in query_context.lower() for word in ['图', '图片', '照片', '看']):
                    query_modality = ModalityType.IMAGE
                elif any(word in query_context.lower() for word in ['听', '声音', '音频']):
                    query_modality = ModalityType.AUDIO
                else:
                    query_modality = ModalityType.TEXT
            
            # 计算注意力
            fused_features, attention_scores = await self.attention_mechanism.compute_attention(
                features, query_modality
            )
            
            return fused_features, attention_scores
            
        except Exception as e:
            logger.error(f"Attention fusion error: {e}")
            return np.zeros(256), {}
    
    async def _cross_modal_fusion(self, features: Dict[ModalityType, np.ndarray]) -> Tuple[np.ndarray, Dict[str, float]]:
        """跨模态融合：模态间交互"""
        try:
            attention_scores = {}
            
            # 计算模态间相似度
            modality_list = list(features.keys())
            similarity_matrix = np.zeros((len(modality_list), len(modality_list)))
            
            for i, mod1 in enumerate(modality_list):
                for j, mod2 in enumerate(modality_list):
                    feat1 = features[mod1]
                    feat2 = features[mod2]
                    
                    # 调整特征维度
                    min_dim = min(len(feat1), len(feat2))
                    if min_dim > 0:
                        similarity = np.dot(feat1[:min_dim], feat2[:min_dim]) / (
                            np.linalg.norm(feat1[:min_dim]) * np.linalg.norm(feat2[:min_dim]) + 1e-8
                        )
                        similarity_matrix[i, j] = similarity
                        attention_scores[f"cross_{mod1.value}_{mod2.value}"] = float(similarity)
            
            # 基于相似度加权融合
            fused = np.zeros(256)
            total_weight = 0
            
            for i, modality in enumerate(modality_list):
                # 计算该模态与其他模态的平均相似度作为权重
                weight = np.mean(similarity_matrix[i, :])
                feature = features[modality]
                
                if len(feature) >= 256:
                    fused += feature[:256] * weight
                else:
                    padded = np.zeros(256)
                    padded[:len(feature)] = feature
                    fused += padded * weight
                
                total_weight += weight
            
            if total_weight > 0:
                fused /= total_weight
            
            return fused, attention_scores
            
        except Exception as e:
            logger.error(f"Cross-modal fusion error: {e}")
            return np.zeros(256), {}
    
    async def _hybrid_fusion(self, 
                           features: Dict[ModalityType, np.ndarray],
                           modality_inputs: List[ModalityInput],
                           query_context: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, float]]:
        """混合融合：结合多种策略"""
        try:
            # 1. 早期融合
            early_fused, early_scores = await self._early_fusion(features)
            
            # 2. 注意力融合
            attention_fused, attention_scores = await self._attention_fusion(features, query_context)
            
            # 3. 跨模态融合
            cross_fused, cross_scores = await self._cross_modal_fusion(features)
            
            # 4. 自适应权重组合
            strategy_weights = self._calculate_strategy_weights(modality_inputs, query_context)
            
            fused = (early_fused * strategy_weights['early'] + 
                    attention_fused * strategy_weights['attention'] + 
                    cross_fused * strategy_weights['cross'])
            
            # 合并注意力分数
            combined_scores = {}
            combined_scores.update({f"hybrid_early_{k}": v * strategy_weights['early'] for k, v in early_scores.items()})
            combined_scores.update({f"hybrid_attention_{k}": v * strategy_weights['attention'] for k, v in attention_scores.items()})
            combined_scores.update({f"hybrid_cross_{k}": v * strategy_weights['cross'] for k, v in cross_scores.items()})
            
            return fused, combined_scores
            
        except Exception as e:
            logger.error(f"Hybrid fusion error: {e}")
            return np.zeros(256), {}
    
    def _process_modality_specific(self, feature: np.ndarray, modality: ModalityType) -> np.ndarray:
        """模态特定处理"""
        try:
            if modality == ModalityType.TEXT:
                # 文本特征的特殊处理
                return feature * 1.1  # 轻微增强
            elif modality == ModalityType.IMAGE:
                # 图像特征的特殊处理
                return np.tanh(feature)  # 非线性变换
            elif modality == ModalityType.AUDIO:
                # 音频特征的特殊处理
                return np.abs(feature)  # 取绝对值
            else:
                return feature
                
        except Exception as e:
            logger.error(f"Modality-specific processing error: {e}")
            return feature
    
    def _calculate_strategy_weights(self, 
                                  modality_inputs: List[ModalityInput],
                                  query_context: Optional[str] = None) -> Dict[str, float]:
        """计算策略权重"""
        try:
            weights = {'early': 0.3, 'attention': 0.4, 'cross': 0.3}
            
            # 根据模态数量调整
            num_modalities = len(modality_inputs)
            if num_modalities == 1:
                weights = {'early': 0.7, 'attention': 0.2, 'cross': 0.1}
            elif num_modalities >= 3:
                weights = {'early': 0.2, 'attention': 0.3, 'cross': 0.5}
            
            # 根据查询上下文调整
            if query_context:
                if any(word in query_context.lower() for word in ['比较', '对比', '关系']):
                    weights['cross'] *= 1.5
                elif any(word in query_context.lower() for word in ['重点', '关注', '主要']):
                    weights['attention'] *= 1.3
            
            # 归一化
            total = sum(weights.values())
            return {k: v / total for k, v in weights.items()}
            
        except Exception as e:
            logger.error(f"Strategy weight calculation error: {e}")
            return {'early': 0.33, 'attention': 0.33, 'cross': 0.34}
    
    def _calculate_fusion_confidence(self, 
                                   modality_inputs: List[ModalityInput],
                                   attention_scores: Dict[str, float]) -> float:
        """计算融合置信度"""
        try:
            # 基于输入置信度
            input_confidences = [inp.confidence for inp in modality_inputs]
            avg_input_confidence = np.mean(input_confidences) if input_confidences else 0.5
            
            # 基于注意力分数的一致性
            if attention_scores:
                score_values = list(attention_scores.values())
                attention_consistency = 1.0 - np.std(score_values)
            else:
                attention_consistency = 0.5
            
            # 基于模态数量的奖励
            modality_bonus = min(len(modality_inputs) * 0.1, 0.3)
            
            # 综合置信度
            fusion_confidence = (avg_input_confidence * 0.5 + 
                               attention_consistency * 0.3 + 
                               modality_bonus * 0.2)
            
            return min(max(fusion_confidence, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Fusion confidence calculation error: {e}")
            return 0.5
    
    def _update_modality_weights(self, 
                               modality_inputs: List[ModalityInput],
                               attention_scores: Dict[str, float]) -> Dict[ModalityType, float]:
        """更新模态权重"""
        try:
            updated_weights = self.modality_weights.copy()
            
            # 基于注意力分数调整权重
            for modality in [inp.modality for inp in modality_inputs]:
                modality_attention_scores = [
                    score for key, score in attention_scores.items()
                    if modality.value in key
                ]
                
                if modality_attention_scores:
                    avg_attention = np.mean(modality_attention_scores)
                    # 轻微调整权重
                    updated_weights[modality] = (
                        updated_weights.get(modality, 0.1) * 0.9 + avg_attention * 0.1
                    )
            
            # 归一化权重
            total_weight = sum(updated_weights.values())
            if total_weight > 0:
                updated_weights = {k: v / total_weight for k, v in updated_weights.items()}
            
            return updated_weights
            
        except Exception as e:
            logger.error(f"Modality weight update error: {e}")
            return self.modality_weights
    
    def get_fusion_analytics(self) -> Dict[str, Any]:
        """获取融合分析统计"""
        try:
            if not self.fusion_history:
                return {}
            
            # 统计各种指标
            recent_history = self.fusion_history[-100:]  # 最近100次
            
            avg_confidence = np.mean([h['confidence'] for h in recent_history])
            avg_processing_time = np.mean([h['processing_time'] for h in recent_history])
            
            # 模态使用统计
            modality_usage = defaultdict(int)
            for history in recent_history:
                for modality in history['modalities']:
                    modality_usage[modality] += 1
            
            # 策略使用统计
            strategy_usage = defaultdict(int)
            for history in recent_history:
                strategy_usage[history['strategy']] += 1
            
            return {
                'total_fusions': len(self.fusion_history),
                'recent_fusions': len(recent_history),
                'average_confidence': avg_confidence,
                'average_processing_time': avg_processing_time,
                'modality_usage': dict(modality_usage),
                'strategy_usage': dict(strategy_usage),
                'current_modality_weights': self.modality_weights
            }
            
        except Exception as e:
            logger.error(f"Fusion analytics error: {e}")
            return {}

# 使用示例
async def main():
    """示例用法"""
    fusion_system = MultiModalFusion(FusionStrategy.HYBRID_FUSION)
    
    # 创建模拟输入
    text_input = ModalityInput(
        modality=ModalityType.TEXT,
        data="这是一张美丽的风景照片，展示了山川和湖泊。",
        confidence=0.9
    )
    
    # 模拟图像数据
    from PIL import Image
    mock_image = Image.new('RGB', (224, 224), color='lightblue')
    image_input = ModalityInput(
        modality=ModalityType.IMAGE,
        data=mock_image,
        confidence=0.8
    )
    
    audio_input = ModalityInput(
        modality=ModalityType.AUDIO,
        data=b"mock_audio_data" * 100,
        confidence=0.7
    )
    
    # 融合多模态信息
    result = await fusion_system.fuse_modalities(
        [text_input, image_input, audio_input],
        query_context="描述这张图片的内容"
    )
    
    print("=== 多模态融合结果 ===")
    print(f"融合特征维度: {result.fused_features.shape}")
    print(f"融合置信度: {result.fusion_confidence:.3f}")
    print(f"处理时间: {result.processing_time:.3f}秒")
    print(f"使用策略: {result.strategy_used.value}")
    
    print("\n模态权重:")
    for modality, weight in result.modality_weights.items():
        print(f"  {modality.value}: {weight:.3f}")
    
    print("\n注意力分数:")
    for key, score in list(result.attention_scores.items())[:5]:
        print(f"  {key}: {score:.3f}")
    
    # 获取分析统计
    analytics = fusion_system.get_fusion_analytics()
    if analytics:
        print(f"\n=== 融合统计 ===")
        print(f"总融合次数: {analytics['total_fusions']}")
        print(f"平均置信度: {analytics['average_confidence']:.3f}")
        print(f"平均处理时间: {analytics['average_processing_time']:.3f}秒")

if __name__ == "__main__":
    asyncio.run(main())
