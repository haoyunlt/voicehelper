"""
长文本处理性能测试
验证200K tokens处理能力
"""

import asyncio
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from algo.core.long_context_processor import process_long_context, TokenCounter

async def test_200k_tokens():
    """测试200K tokens处理能力"""
    print("🧪 长文本处理性能测试 - 200K Tokens")
    print("=" * 60)
    
    # 生成大约200K tokens的文本
    base_text = """
# 深度学习与人工智能技术发展报告

## 1. 引言
深度学习作为人工智能的核心技术，在过去十年中取得了突破性进展。从最初的多层感知机到现在的大规模预训练模型，深度学习技术不断演进，推动着人工智能在各个领域的应用。

## 2. 技术发展历程
### 2.1 早期发展（1940s-1980s）
人工神经网络的概念最早可以追溯到1940年代。McCulloch和Pitts在1943年提出了第一个数学神经元模型，为后续的神经网络发展奠定了基础。1958年，Rosenblatt发明了感知机（Perceptron），这是第一个可以学习的人工神经网络。

### 2.2 第一次AI冬天（1970s-1980s）
由于计算能力的限制和理论的不完善，神经网络研究在1970年代遇到了瓶颈。Minsky和Papert在1969年发表的《Perceptrons》一书指出了单层感知机的局限性，导致了第一次AI冬天。

### 2.3 复兴期（1980s-1990s）
1986年，Rumelhart等人重新发现并推广了反向传播算法，使得多层神经网络的训练成为可能。这一突破重新点燃了人们对神经网络的兴趣。

### 2.4 深度学习时代（2000s至今）
2006年，Hinton等人提出了深度信念网络（Deep Belief Networks），标志着深度学习时代的开始。随后，卷积神经网络（CNN）、循环神经网络（RNN）、长短期记忆网络（LSTM）等架构相继出现。

## 3. 核心技术架构
### 3.1 卷积神经网络（CNN）
卷积神经网络是处理图像数据的首选架构。其核心思想是通过卷积操作提取局部特征，通过池化操作降低维度，最终通过全连接层进行分类或回归。

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

### 3.2 循环神经网络（RNN）
循环神经网络专门用于处理序列数据。传统的RNN存在梯度消失问题，因此发展出了LSTM和GRU等变体。

### 3.3 Transformer架构
2017年，Vaswani等人提出了Transformer架构，彻底改变了自然语言处理领域。Transformer完全基于注意力机制，摒弃了循环和卷积结构。

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.W_o(attention_output)
        return output
```

## 4. 预训练模型发展
### 4.1 BERT系列
BERT（Bidirectional Encoder Representations from Transformers）是Google在2018年发布的预训练语言模型。BERT通过掩码语言模型（MLM）和下一句预测（NSP）任务进行预训练。

### 4.2 GPT系列
GPT（Generative Pre-trained Transformer）系列模型由OpenAI开发，采用自回归的方式生成文本。从GPT-1到GPT-4，模型规模不断增大，性能也在持续提升。

### 4.3 T5模型
T5（Text-to-Text Transfer Transformer）将所有NLP任务统一为文本到文本的转换任务，这种统一的框架简化了模型设计和训练过程。

## 5. 计算机视觉应用
### 5.1 图像分类
图像分类是计算机视觉的基础任务。从AlexNet到ResNet，再到EfficientNet，模型架构不断优化，准确率持续提升。

### 5.2 目标检测
目标检测需要同时解决分类和定位问题。主要方法包括两阶段检测器（如R-CNN系列）和单阶段检测器（如YOLO系列）。

### 5.3 语义分割
语义分割要求对图像中的每个像素进行分类。FCN、U-Net、DeepLab等架构在这一任务上表现出色。

## 6. 自然语言处理应用
### 6.1 机器翻译
神经机器翻译（NMT）基于编码器-解码器架构，能够实现端到端的翻译。Transformer的出现进一步提升了翻译质量。

### 6.2 文本摘要
文本摘要分为抽取式和生成式两种。抽取式摘要选择原文中的重要句子，生成式摘要则生成新的摘要文本。

### 6.3 问答系统
问答系统需要理解问题并从知识库或文档中找到答案。BERT等预训练模型在阅读理解任务上取得了显著进展。

## 7. 强化学习
### 7.1 基础概念
强化学习通过与环境交互来学习最优策略。主要概念包括状态、动作、奖励、策略和价值函数。

### 7.2 深度强化学习
深度强化学习结合了深度学习和强化学习，能够处理高维状态空间。DQN、A3C、PPO等算法在游戏和机器人控制等领域取得了成功。

## 8. 生成对抗网络（GAN）
### 8.1 基本原理
GAN由生成器和判别器组成，两者通过对抗训练不断改进。生成器试图生成逼真的数据，判别器则试图区分真实数据和生成数据。

### 8.2 应用领域
GAN在图像生成、风格迁移、数据增强等方面有广泛应用。StyleGAN、CycleGAN等变体进一步扩展了GAN的应用范围。

## 9. 技术挑战与解决方案
### 9.1 数据质量问题
高质量的训练数据是深度学习成功的关键。数据清洗、标注质量控制、数据增强等技术有助于提升数据质量。

### 9.2 模型可解释性
深度学习模型通常被视为"黑盒"，缺乏可解释性。注意力可视化、梯度分析、LIME等方法有助于理解模型决策过程。

### 9.3 计算资源需求
大规模深度学习模型需要大量计算资源。模型压缩、知识蒸馏、量化等技术可以减少计算需求。

### 9.4 泛化能力
模型在训练数据上表现良好，但在新数据上可能表现不佳。正则化、数据增强、迁移学习等方法有助于提升泛化能力。

## 10. 未来发展趋势
### 10.1 多模态学习
未来的AI系统将能够处理多种模态的数据，如文本、图像、音频等。多模态学习将推动AI向更通用的方向发展。

### 10.2 自监督学习
自监督学习不需要人工标注，能够从大量无标签数据中学习有用的表示。这将大大降低AI系统的数据需求。

### 10.3 神经架构搜索
神经架构搜索（NAS）能够自动设计神经网络架构，有望发现比人工设计更优的架构。

### 10.4 联邦学习
联邦学习允许在不共享原始数据的情况下训练模型，这对保护隐私具有重要意义。

## 11. 伦理与社会影响
### 11.1 算法偏见
AI系统可能存在偏见，导致不公平的结果。需要在数据收集、模型训练和部署过程中注意公平性。

### 11.2 隐私保护
AI系统处理大量个人数据，隐私保护成为重要问题。差分隐私、同态加密等技术有助于保护用户隐私。

### 11.3 就业影响
AI自动化可能影响就业市场。需要考虑如何帮助受影响的工人转型和再培训。

## 12. 结论
深度学习技术在过去十年中取得了巨大进展，推动了人工智能在各个领域的应用。未来，随着技术的不断发展和完善，深度学习将在更多领域发挥重要作用。同时，我们也需要关注技术发展带来的挑战和社会影响，确保AI技术的健康发展。

深度学习的成功离不开算法创新、计算能力提升和数据积累。展望未来，多模态学习、自监督学习、神经架构搜索等新技术将进一步推动深度学习的发展。我们有理由相信，深度学习将继续为人类社会带来更多价值。
    """
    
    # 重复文本以达到约200K tokens
    # 先计算基础文本的token数
    base_tokens = TokenCounter.count_tokens(base_text)
    print(f"基础文本tokens: {base_tokens:,.0f}")
    
    # 计算需要重复多少次才能达到200K tokens
    target_tokens = 200000
    multiplier = max(int(target_tokens / base_tokens), 1)
    
    large_text = base_text * multiplier
    
    print(f"生成测试文本:")
    print(f"  重复倍数: {multiplier}")
    print(f"  字符数: {len(large_text):,}")
    
    # 计算token数
    token_count = TokenCounter.count_tokens(large_text)
    print(f"  估算tokens: {token_count:,.0f}")
    
    # 如果还不够，继续添加
    while token_count < 180000:  # 至少180K tokens
        large_text += base_text
        token_count = TokenCounter.count_tokens(large_text)
    
    print(f"  最终tokens: {token_count:,.0f}")
    
    # 测试不同的上下文窗口大小
    test_cases = [
        {"max_tokens": 50000, "name": "50K tokens"},
        {"max_tokens": 100000, "name": "100K tokens"},
        {"max_tokens": 200000, "name": "200K tokens"},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n🧪 测试 {case['name']} 上下文窗口:")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            result = await process_long_context(
                text=large_text,
                query="深度学习的核心技术和发展趋势",
                max_tokens=case["max_tokens"],
                preserve_structure=True
            )
            
            processing_time = time.time() - start_time
            
            print(f"  ✅ 处理成功")
            print(f"  📊 统计信息:")
            print(f"    - 原始tokens: {token_count:,.0f}")
            print(f"    - 处理后tokens: {result.total_tokens:,.0f}")
            print(f"    - 压缩比: {result.compression_ratio:.2%}")
            print(f"    - 处理时间: {processing_time:.2f}s")
            print(f"    - 上下文窗口数: {len(result.context_windows)}")
            print(f"    - 关键点数量: {len(result.key_points)}")
            
            # 验证结果质量
            quality_score = 0
            if result.total_tokens <= case["max_tokens"]:
                quality_score += 30  # 符合token限制
            if result.compression_ratio < 1.0:
                quality_score += 20  # 有效压缩
            if len(result.key_points) > 0:
                quality_score += 20  # 提取了关键点
            if len(result.summary) > 50:
                quality_score += 20  # 生成了摘要
            if processing_time < 5.0:
                quality_score += 10  # 处理速度合理
            
            print(f"    - 质量评分: {quality_score}/100")
            
            results.append({
                "case": case["name"],
                "success": True,
                "original_tokens": token_count,
                "processed_tokens": result.total_tokens,
                "compression_ratio": result.compression_ratio,
                "processing_time": processing_time,
                "quality_score": quality_score,
                "windows_count": len(result.context_windows)
            })
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
            results.append({
                "case": case["name"],
                "success": False,
                "error": str(e)
            })
    
    # 总结测试结果
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    success_count = sum(1 for r in results if r.get("success", False))
    
    print(f"成功率: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    if success_count > 0:
        avg_compression = sum(r.get("compression_ratio", 0) for r in results if r.get("success")) / success_count
        avg_processing_time = sum(r.get("processing_time", 0) for r in results if r.get("success")) / success_count
        avg_quality = sum(r.get("quality_score", 0) for r in results if r.get("success")) / success_count
        
        print(f"平均压缩比: {avg_compression:.2%}")
        print(f"平均处理时间: {avg_processing_time:.2f}s")
        print(f"平均质量评分: {avg_quality:.1f}/100")
    
    print(f"\n详细结果:")
    for result in results:
        if result.get("success"):
            print(f"  ✅ {result['case']}: {result['processed_tokens']:,} tokens, "
                  f"{result['compression_ratio']:.2%} 压缩, "
                  f"{result['processing_time']:.2f}s, "
                  f"质量 {result['quality_score']}/100")
        else:
            print(f"  ❌ {result['case']}: {result.get('error', 'Unknown error')}")
    
    # 判断测试是否通过
    # 至少要能处理200K tokens，压缩比合理，处理时间可接受
    test_passed = (
        success_count >= 2 and  # 至少2个测试用例成功
        any(r.get("case") == "200K tokens" and r.get("success") for r in results) and  # 200K测试成功
        avg_compression < 0.8 and  # 压缩比小于80%
        avg_processing_time < 10.0  # 处理时间小于10秒
    )
    
    print(f"\n🎯 测试结果: {'通过' if test_passed else '失败'}")
    
    return test_passed

if __name__ == "__main__":
    success = asyncio.run(test_200k_tokens())
    exit(0 if success else 1)
