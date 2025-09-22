"""
VoiceHelper 长文本处理系统
支持200K tokens上下文窗口，对标Claude 3.5
实现分层注意力、滑动窗口和上下文压缩
"""

import asyncio
import time
import logging
import json
import math
from typing import Dict, List, Optional, Tuple, Any, AsyncIterator
from dataclasses import dataclass
from collections import deque
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """文本块"""
    content: str
    start_pos: int
    end_pos: int
    importance_score: float
    chunk_type: str  # "header", "content", "summary", "code"
    metadata: Dict[str, Any]

@dataclass
class ContextWindow:
    """上下文窗口"""
    chunks: List[TextChunk]
    total_tokens: int
    compression_ratio: float
    processing_time: float

@dataclass
class LongTextResult:
    """长文本处理结果"""
    processed_content: str
    context_windows: List[ContextWindow]
    total_tokens: int
    compression_ratio: float
    processing_time: float
    summary: str
    key_points: List[str]

class TokenCounter:
    """Token计数器"""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """估算token数量 (简化版)"""
        # 中文：1个字符约等于1个token
        # 英文：1个单词约等于1-2个token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_words = len([w for w in text.split() if any(c.isalpha() for c in w)])
        other_chars = len(text) - chinese_chars - sum(len(w) for w in text.split() if any(c.isalpha() for c in w))
        
        return chinese_chars + english_words * 1.5 + other_chars * 0.5

class TextChunker:
    """文本分块器"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        
    def chunk_text(self, text: str) -> List[TextChunk]:
        """智能文本分块"""
        chunks = []
        
        # 按段落分割
        paragraphs = self._split_by_paragraphs(text)
        
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para_tokens = TokenCounter.count_tokens(para)
            current_tokens = TokenCounter.count_tokens(current_chunk)
            
            if current_tokens + para_tokens > self.chunk_size and current_chunk:
                # 创建当前块
                chunk = self._create_chunk(
                    current_chunk, 
                    current_start, 
                    current_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # 开始新块，保留重叠
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + para
                current_start = current_start + len(current_chunk) - len(overlap_text)
            else:
                current_chunk += para
        
        # 处理最后一块
        if current_chunk:
            chunk = self._create_chunk(
                current_chunk, 
                current_start, 
                current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """按段落分割文本"""
        # 按多种分隔符分割
        separators = ['\n\n', '\n', '。', '！', '？', '.', '!', '?']
        
        paragraphs = [text]
        for sep in separators:
            new_paragraphs = []
            for para in paragraphs:
                new_paragraphs.extend(para.split(sep))
            paragraphs = [p.strip() for p in new_paragraphs if p.strip()]
        
        return paragraphs
    
    def _get_overlap_text(self, text: str) -> str:
        """获取重叠文本"""
        if len(text) <= self.overlap:
            return text
        return text[-self.overlap:]
    
    def _create_chunk(self, content: str, start: int, end: int) -> TextChunk:
        """创建文本块"""
        # 计算重要性分数
        importance = self._calculate_importance(content)
        
        # 判断块类型
        chunk_type = self._classify_chunk(content)
        
        return TextChunk(
            content=content,
            start_pos=start,
            end_pos=end,
            importance_score=importance,
            chunk_type=chunk_type,
            metadata={
                "token_count": TokenCounter.count_tokens(content),
                "char_count": len(content),
                "has_code": "```" in content or "def " in content,
                "has_numbers": any(c.isdigit() for c in content),
                "language": self._detect_language(content)
            }
        )
    
    def _calculate_importance(self, content: str) -> float:
        """计算内容重要性"""
        score = 0.5  # 基础分数
        
        # 关键词权重
        keywords = ["重要", "关键", "核心", "主要", "总结", "结论", "问题", "解决", "方案"]
        for keyword in keywords:
            score += content.count(keyword) * 0.1
        
        # 结构化内容权重
        if any(marker in content for marker in ["#", "##", "###", "1.", "2.", "3."]):
            score += 0.2
        
        # 代码块权重
        if "```" in content or "def " in content:
            score += 0.3
        
        return min(score, 1.0)
    
    def _classify_chunk(self, content: str) -> str:
        """分类文本块"""
        if content.startswith("#") or content.startswith("##"):
            return "header"
        elif "```" in content or "def " in content or "class " in content:
            return "code"
        elif any(word in content for word in ["总结", "结论", "概述", "摘要"]):
            return "summary"
        else:
            return "content"
    
    def _detect_language(self, content: str) -> str:
        """检测语言"""
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        total_chars = len(content)
        
        if chinese_chars / max(total_chars, 1) > 0.3:
            return "chinese"
        else:
            return "english"

class HierarchicalAttention:
    """分层注意力机制"""
    
    def __init__(self):
        self.attention_weights = {}
        
    def compute_attention(self, chunks: List[TextChunk], query: str) -> List[float]:
        """计算注意力权重"""
        query_tokens = set(query.lower().split())
        attention_scores = []
        
        for chunk in chunks:
            # 基于内容相似度的注意力
            content_tokens = set(chunk.content.lower().split())
            overlap = len(query_tokens & content_tokens)
            content_score = overlap / max(len(query_tokens), 1)
            
            # 基于重要性的注意力
            importance_score = chunk.importance_score
            
            # 基于位置的注意力 (开头和结尾更重要)
            position_score = self._calculate_position_score(chunk, len(chunks))
            
            # 基于类型的注意力
            type_score = self._get_type_weight(chunk.chunk_type)
            
            # 综合注意力分数
            final_score = (
                content_score * 0.4 + 
                importance_score * 0.3 + 
                position_score * 0.2 + 
                type_score * 0.1
            )
            
            attention_scores.append(final_score)
        
        # 归一化
        total_score = sum(attention_scores)
        if total_score > 0:
            attention_scores = [score / total_score for score in attention_scores]
        
        return attention_scores
    
    def _calculate_position_score(self, chunk: TextChunk, total_chunks: int) -> float:
        """计算位置分数"""
        # 开头和结尾的块更重要
        chunk_index = chunk.start_pos / max(chunk.end_pos, 1)  # 简化的位置计算
        
        if chunk_index < 0.1 or chunk_index > 0.9:  # 开头或结尾
            return 1.0
        elif chunk_index < 0.3 or chunk_index > 0.7:  # 靠近开头或结尾
            return 0.8
        else:  # 中间部分
            return 0.6
    
    def _get_type_weight(self, chunk_type: str) -> float:
        """获取类型权重"""
        type_weights = {
            "header": 1.0,
            "summary": 0.9,
            "code": 0.8,
            "content": 0.7
        }
        return type_weights.get(chunk_type, 0.5)

class ContextCompressor:
    """上下文压缩器"""
    
    def __init__(self, target_ratio: float = 0.3):
        self.target_ratio = target_ratio
        
    async def compress_context(
        self, 
        chunks: List[TextChunk], 
        max_tokens: int = 200000
    ) -> List[TextChunk]:
        """压缩上下文"""
        total_tokens = sum(chunk.metadata["token_count"] for chunk in chunks)
        
        if total_tokens <= max_tokens:
            return chunks
        
        # 按重要性排序
        sorted_chunks = sorted(chunks, key=lambda x: x.importance_score, reverse=True)
        
        # 选择最重要的块
        selected_chunks = []
        current_tokens = 0
        target_tokens = int(max_tokens * self.target_ratio)
        
        for chunk in sorted_chunks:
            chunk_tokens = chunk.metadata["token_count"]
            if current_tokens + chunk_tokens <= target_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                # 如果是重要块，尝试截断
                if chunk.importance_score > 0.8:
                    remaining_tokens = target_tokens - current_tokens
                    if remaining_tokens > 100:  # 至少保留100个token
                        compressed_chunk = self._truncate_chunk(chunk, remaining_tokens)
                        selected_chunks.append(compressed_chunk)
                        break
        
        # 按原始顺序排序
        selected_chunks.sort(key=lambda x: x.start_pos)
        
        return selected_chunks
    
    def _truncate_chunk(self, chunk: TextChunk, max_tokens: int) -> TextChunk:
        """截断文本块"""
        content = chunk.content
        current_tokens = 0
        truncated_content = ""
        
        sentences = content.split('。')
        for sentence in sentences:
            sentence_tokens = TokenCounter.count_tokens(sentence)
            if current_tokens + sentence_tokens <= max_tokens:
                truncated_content += sentence + "。"
                current_tokens += sentence_tokens
            else:
                break
        
        # 创建新的截断块
        return TextChunk(
            content=truncated_content,
            start_pos=chunk.start_pos,
            end_pos=chunk.start_pos + len(truncated_content),
            importance_score=chunk.importance_score,
            chunk_type=chunk.chunk_type,
            metadata={
                **chunk.metadata,
                "token_count": current_tokens,
                "truncated": True
            }
        )

class SlidingWindowProcessor:
    """滑动窗口处理器"""
    
    def __init__(self, window_size: int = 50000, step_size: int = 25000):
        self.window_size = window_size
        self.step_size = step_size
        
    async def process_with_sliding_window(
        self, 
        chunks: List[TextChunk]
    ) -> List[ContextWindow]:
        """使用滑动窗口处理"""
        windows = []
        current_pos = 0
        
        while current_pos < len(chunks):
            # 选择当前窗口的块
            window_chunks = []
            window_tokens = 0
            
            for i in range(current_pos, len(chunks)):
                chunk = chunks[i]
                chunk_tokens = chunk.metadata["token_count"]
                
                if window_tokens + chunk_tokens <= self.window_size:
                    window_chunks.append(chunk)
                    window_tokens += chunk_tokens
                else:
                    break
            
            if window_chunks:
                # 创建上下文窗口
                window = ContextWindow(
                    chunks=window_chunks,
                    total_tokens=window_tokens,
                    compression_ratio=window_tokens / sum(c.metadata["token_count"] for c in window_chunks),
                    processing_time=0.0
                )
                windows.append(window)
            
            # 移动窗口
            current_pos += max(1, len(window_chunks) // 2)  # 50%重叠
            
            if current_pos >= len(chunks):
                break
        
        return windows

class LongContextProcessor:
    """长文本上下文处理器"""
    
    def __init__(self, max_context_tokens: int = 200000):
        self.max_context_tokens = max_context_tokens
        self.chunker = TextChunker()
        self.attention = HierarchicalAttention()
        self.compressor = ContextCompressor()
        self.sliding_window = SlidingWindowProcessor()
        
        # 性能统计
        self.total_processed = 0
        self.total_processing_time = 0.0
        
    async def process_long_text(
        self, 
        text: str, 
        query: Optional[str] = None,
        preserve_structure: bool = True
    ) -> LongTextResult:
        """处理长文本"""
        start_time = time.time()
        
        try:
            # 1. 文本分块
            chunks = self.chunker.chunk_text(text)
            logger.info(f"Text chunked into {len(chunks)} chunks")
            
            # 2. 计算注意力权重
            if query:
                attention_weights = self.attention.compute_attention(chunks, query)
                # 根据注意力权重调整重要性
                for i, chunk in enumerate(chunks):
                    chunk.importance_score = (chunk.importance_score + attention_weights[i]) / 2
            
            # 3. 上下文压缩
            compressed_chunks = await self.compressor.compress_context(chunks, self.max_context_tokens)
            logger.info(f"Context compressed to {len(compressed_chunks)} chunks")
            
            # 4. 滑动窗口处理
            context_windows = await self.sliding_window.process_with_sliding_window(compressed_chunks)
            
            # 5. 生成摘要和关键点
            summary = await self._generate_summary(compressed_chunks)
            key_points = await self._extract_key_points(compressed_chunks)
            
            # 6. 重构处理后的内容
            processed_content = self._reconstruct_content(compressed_chunks, preserve_structure)
            
            # 计算统计信息
            total_tokens = sum(chunk.metadata["token_count"] for chunk in chunks)
            compressed_tokens = sum(chunk.metadata["token_count"] for chunk in compressed_chunks)
            compression_ratio = compressed_tokens / max(total_tokens, 1)
            processing_time = time.time() - start_time
            
            # 更新统计
            self.total_processed += 1
            self.total_processing_time += processing_time
            
            result = LongTextResult(
                processed_content=processed_content,
                context_windows=context_windows,
                total_tokens=compressed_tokens,
                compression_ratio=compression_ratio,
                processing_time=processing_time,
                summary=summary,
                key_points=key_points
            )
            
            logger.info(f"Long text processed in {processing_time:.2f}s, compression ratio: {compression_ratio:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Long text processing error: {e}")
            # 返回原始文本作为fallback
            return LongTextResult(
                processed_content=text[:self.max_context_tokens],  # 简单截断
                context_windows=[],
                total_tokens=TokenCounter.count_tokens(text),
                compression_ratio=1.0,
                processing_time=time.time() - start_time,
                summary="处理出错，返回原始内容",
                key_points=["处理失败"]
            )
    
    async def _generate_summary(self, chunks: List[TextChunk]) -> str:
        """生成摘要"""
        # 选择最重要的几个块生成摘要
        important_chunks = sorted(chunks, key=lambda x: x.importance_score, reverse=True)[:5]
        
        summary_parts = []
        for chunk in important_chunks:
            # 提取每个块的关键句子
            sentences = chunk.content.split('。')
            if sentences:
                key_sentence = max(sentences, key=len)  # 简单选择最长的句子
                summary_parts.append(key_sentence.strip())
        
        return "。".join(summary_parts[:3]) + "。"  # 最多3句摘要
    
    async def _extract_key_points(self, chunks: List[TextChunk]) -> List[str]:
        """提取关键点"""
        key_points = []
        
        for chunk in chunks:
            if chunk.chunk_type == "header":
                # 标题作为关键点
                key_points.append(chunk.content.strip())
            elif chunk.importance_score > 0.8:
                # 高重要性内容的第一句
                sentences = chunk.content.split('。')
                if sentences:
                    key_points.append(sentences[0].strip())
        
        return key_points[:10]  # 最多10个关键点
    
    def _reconstruct_content(self, chunks: List[TextChunk], preserve_structure: bool) -> str:
        """重构内容"""
        if preserve_structure:
            # 保持原有结构
            content_parts = []
            for chunk in chunks:
                if chunk.chunk_type == "header":
                    content_parts.append(f"\n{chunk.content}\n")
                else:
                    content_parts.append(chunk.content)
            return "".join(content_parts)
        else:
            # 简单连接
            return " ".join(chunk.content for chunk in chunks)
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        avg_processing_time = (
            self.total_processing_time / self.total_processed 
            if self.total_processed > 0 else 0
        )
        
        return {
            "total_processed": self.total_processed,
            "average_processing_time_s": avg_processing_time,
            "max_context_tokens": self.max_context_tokens,
            "chunker_settings": {
                "chunk_size": self.chunker.chunk_size,
                "overlap": self.chunker.overlap
            }
        }

# 全局实例
long_context_processor = LongContextProcessor()

async def process_long_context(
    text: str,
    query: Optional[str] = None,
    max_tokens: int = 200000,
    preserve_structure: bool = True
) -> LongTextResult:
    """长文本处理便捷函数"""
    # 动态调整最大token数
    if max_tokens != long_context_processor.max_context_tokens:
        long_context_processor.max_context_tokens = max_tokens
    
    return await long_context_processor.process_long_text(
        text=text,
        query=query,
        preserve_structure=preserve_structure
    )

# 测试代码
if __name__ == "__main__":
    async def test_long_context():
        print("📄 测试长文本处理系统")
        print("=" * 50)
        
        # 生成测试长文本
        test_text = """
# 人工智能发展报告

## 1. 概述
人工智能（AI）是当今科技发展的重要方向，涉及机器学习、深度学习、自然语言处理等多个领域。

## 2. 技术发展
### 2.1 机器学习
机器学习是AI的核心技术之一，包括监督学习、无监督学习和强化学习。监督学习通过标注数据训练模型，无监督学习从数据中发现隐藏模式，强化学习通过与环境交互学习最优策略。

### 2.2 深度学习
深度学习基于神经网络，特别是深度神经网络。卷积神经网络（CNN）在图像识别方面表现出色，循环神经网络（RNN）和长短期记忆网络（LSTM）在序列数据处理方面有优势。

```python
# 示例代码：简单的神经网络
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

### 2.3 自然语言处理
自然语言处理（NLP）让机器理解和生成人类语言。关键技术包括词嵌入、注意力机制、Transformer架构等。

## 3. 应用领域
### 3.1 计算机视觉
在图像识别、目标检测、人脸识别等方面有广泛应用。

### 3.2 语音识别
将语音转换为文本，广泛应用于智能助手、语音输入等场景。

### 3.3 推荐系统
基于用户行为和偏好进行个性化推荐。

## 4. 挑战与机遇
### 4.1 技术挑战
- 数据质量和隐私保护
- 算法可解释性
- 计算资源需求
- 模型泛化能力

### 4.2 发展机遇
- 跨领域应用拓展
- 硬件技术进步
- 开源生态完善
- 产业数字化转型

## 5. 未来展望
人工智能将继续快速发展，在更多领域实现突破。关键发展方向包括：
1. 通用人工智能（AGI）
2. 多模态AI系统
3. 边缘AI计算
4. AI伦理和安全

## 6. 结论
人工智能技术正在深刻改变我们的生活和工作方式。我们需要在推动技术发展的同时，关注其社会影响和伦理问题，确保AI技术的健康发展。

总的来说，人工智能是一个充满机遇和挑战的领域，需要持续的研究和创新。
        """ * 10  # 重复10次模拟长文本
        
        print(f"原始文本长度: {len(test_text)} 字符")
        print(f"估算token数: {TokenCounter.count_tokens(test_text)}")
        
        # 测试长文本处理
        query = "人工智能的主要技术和应用领域"
        result = await process_long_context(
            text=test_text,
            query=query,
            max_tokens=50000,  # 50K tokens限制
            preserve_structure=True
        )
        
        print(f"\n处理结果:")
        print(f"  处理后token数: {result.total_tokens}")
        print(f"  压缩比: {result.compression_ratio:.2f}")
        print(f"  处理时间: {result.processing_time:.2f}s")
        print(f"  上下文窗口数: {len(result.context_windows)}")
        
        print(f"\n摘要:")
        print(f"  {result.summary}")
        
        print(f"\n关键点:")
        for i, point in enumerate(result.key_points[:5], 1):
            print(f"  {i}. {point}")
        
        print(f"\n处理后内容预览:")
        preview = result.processed_content[:500]
        print(f"  {preview}...")
        
        # 性能统计
        stats = long_context_processor.get_performance_stats()
        print(f"\n📊 性能统计:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return result.total_tokens <= 50000 and result.compression_ratio < 1.0
    
    # 运行测试
    import asyncio
    success = asyncio.run(test_long_context())
    print(f"\n🎯 测试{'通过' if success else '失败'}！")
