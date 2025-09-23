"""
BGE + FAISS RAG 使用示例
演示如何使用新的BGE+FAISS检索系统
"""

import asyncio
import json
from typing import List, Dict, Any
from loguru import logger

from core.rag_factory import get_rag_factory, create_embedder, create_retriever
from core.rag.metrics import get_metrics_collector


async def main():
    """主函数"""
    logger.info("开始BGE+FAISS RAG示例")
    
    # 示例文档
    documents = [
        {
            "id": "doc_1",
            "title": "人工智能基础",
            "content": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、问题解决、感知和语言理解。",
            "source": "ai_textbook.pdf",
            "metadata": {"category": "technology", "author": "张三"}
        },
        {
            "id": "doc_2", 
            "title": "机器学习概述",
            "content": "机器学习是人工智能的一个子领域，专注于开发算法和统计模型，使计算机系统能够在没有明确编程的情况下从数据中学习和改进性能。",
            "source": "ml_guide.pdf",
            "metadata": {"category": "technology", "author": "李四"}
        },
        {
            "id": "doc_3",
            "title": "深度学习原理",
            "content": "深度学习是机器学习的一个分支，使用多层神经网络来建模和理解复杂的数据模式。它在图像识别、自然语言处理和语音识别等领域取得了突破性进展。",
            "source": "dl_paper.pdf", 
            "metadata": {"category": "technology", "author": "王五"}
        },
        {
            "id": "doc_4",
            "title": "自然语言处理",
            "content": "自然语言处理（NLP）是人工智能和语言学的交叉领域，专注于使计算机能够理解、解释和生成人类语言。NLP技术广泛应用于搜索引擎、翻译系统和聊天机器人。",
            "source": "nlp_handbook.pdf",
            "metadata": {"category": "technology", "author": "赵六"}
        }
    ]
    
    try:
        # 1. 获取RAG工厂
        factory = get_rag_factory()
        
        # 2. 构建索引
        logger.info("开始构建索引...")
        build_result = await factory.build_index_from_documents(
            documents=documents,
            tenant_id="demo_tenant",
            dataset_id="ai_knowledge"
        )
        logger.info(f"索引构建结果: {json.dumps(build_result, indent=2, ensure_ascii=False)}")
        
        # 3. 创建检索器
        logger.info("创建检索器...")
        retriever = create_retriever(
            tenant_id="demo_tenant",
            dataset_id="ai_knowledge"
        )
        
        # 4. 执行检索测试
        test_queries = [
            "什么是人工智能？",
            "机器学习和深度学习的区别是什么？",
            "NLP有哪些应用？",
            "神经网络如何工作？"
        ]
        
        logger.info("开始检索测试...")
        for query in test_queries:
            logger.info(f"\n查询: {query}")
            
            results = retriever.retrieve(
                query=query,
                top_k=3,
                score_threshold=0.1
            )
            
            logger.info(f"检索结果数: {len(results)}")
            for i, result in enumerate(results, 1):
                logger.info(f"  结果{i}: 分数={result['score']:.3f}, 来源={result['source']}")
                logger.info(f"    内容: {result['content'][:100]}...")
        
        # 5. 显示指标统计
        logger.info("\n=== 指标统计 ===")
        metrics_collector = get_metrics_collector()
        
        retrieval_stats = metrics_collector.get_retrieval_stats(window_minutes=10)
        logger.info(f"检索统计: {json.dumps(retrieval_stats, indent=2, ensure_ascii=False)}")
        
        index_stats = metrics_collector.get_index_stats()
        logger.info(f"索引统计: {json.dumps(index_stats, indent=2, ensure_ascii=False)}")
        
        cache_stats = metrics_collector.get_cache_stats()
        logger.info(f"缓存统计: {json.dumps(cache_stats, indent=2, ensure_ascii=False)}")
        
        # 6. 显示检索器统计
        retriever_stats = factory.get_retriever_stats(
            tenant_id="demo_tenant",
            dataset_id="ai_knowledge"
        )
        logger.info(f"检索器统计: {json.dumps(retriever_stats, indent=2, ensure_ascii=False)}")
        
        logger.info("BGE+FAISS RAG示例完成！")
        
    except Exception as e:
        logger.error(f"示例执行失败: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
