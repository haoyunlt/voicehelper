"""
GraphRAG - 图增强检索系统
基于知识图谱的高级RAG实现，提供更准确的语义理解和推理能力
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import networkx as nx
from neo4j import AsyncGraphDatabase
import json
from collections import defaultdict
from loguru import logger

from core.embeddings import EmbeddingService
from core.llm import LLMService
from pymilvus import Collection


@dataclass
class Entity:
    """实体"""
    id: str
    name: str
    type: str  # person, organization, location, concept, etc.
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class Relation:
    """关系"""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """图节点"""
    entity: Entity
    neighbors: Set[str] = field(default_factory=set)
    centrality: float = 0.0
    community: Optional[int] = None


@dataclass
class GraphPath:
    """图路径"""
    nodes: List[str]
    edges: List[Tuple[str, str, str]]  # (source, target, relation)
    score: float
    explanation: Optional[str] = None


class EntityExtractor:
    """实体抽取器"""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.entity_types = [
            "人物", "组织", "地点", "时间", "概念",
            "产品", "事件", "技术", "指标", "流程"
        ]
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """从文本中抽取实体"""
        prompt = f"""
        从以下文本中抽取所有重要实体：
        
        文本：{text}
        
        实体类型包括：{', '.join(self.entity_types)}
        
        请以JSON格式输出，格式如下：
        [
            {{
                "name": "实体名称",
                "type": "实体类型",
                "context": "实体在文本中的上下文",
                "attributes": {{}}
            }}
        ]
        """
        
        response = await self.llm.generate(prompt)
        
        try:
            entities_data = json.loads(response)
            entities = []
            
            for data in entities_data:
                entity = Entity(
                    id=self._generate_entity_id(data["name"], data["type"]),
                    name=data["name"],
                    type=data["type"],
                    attributes=data.get("attributes", {})
                )
                entities.append(entity)
            
            return entities
            
        except json.JSONDecodeError:
            logger.error(f"实体抽取结果解析失败: {response}")
            return []
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """生成实体ID"""
        import hashlib
        content = f"{entity_type}:{name}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
        self.relation_types = [
            "属于", "包含", "位于", "创建", "使用",
            "依赖", "相关", "影响", "导致", "参与",
            "管理", "拥有", "连接", "实现", "支持"
        ]
    
    async def extract_relations(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Relation]:
        """抽取实体间关系"""
        if len(entities) < 2:
            return []
        
        entity_names = [e.name for e in entities]
        entity_map = {e.name: e for e in entities}
        
        prompt = f"""
        分析以下实体之间的关系：
        
        实体列表：{', '.join(entity_names)}
        
        原文：{text}
        
        关系类型包括：{', '.join(self.relation_types)}
        
        请以JSON格式输出所有关系：
        [
            {{
                "source": "源实体名称",
                "target": "目标实体名称",
                "relation": "关系类型",
                "confidence": 0.9,
                "evidence": "支持这个关系的文本证据"
            }}
        ]
        """
        
        response = await self.llm.generate(prompt)
        
        try:
            relations_data = json.loads(response)
            relations = []
            
            for data in relations_data:
                source = entity_map.get(data["source"])
                target = entity_map.get(data["target"])
                
                if source and target:
                    relation = Relation(
                        source_id=source.id,
                        target_id=target.id,
                        relation_type=data["relation"],
                        confidence=data.get("confidence", 0.8),
                        attributes={"evidence": data.get("evidence", "")}
                    )
                    relations.append(relation)
            
            return relations
            
        except json.JSONDecodeError:
            logger.error(f"关系抽取结果解析失败: {response}")
            return []


class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self, neo4j_uri: Optional[str] = None):
        self.graph = nx.DiGraph()
        self.entity_index: Dict[str, Entity] = {}
        self.relation_index: Dict[Tuple[str, str], List[Relation]] = defaultdict(list)
        self.communities: Dict[int, Set[str]] = {}
        
        # Neo4j连接（可选）
        self.neo4j_driver = None
        if neo4j_uri:
            self.neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri)
    
    def add_entity(self, entity: Entity):
        """添加实体"""
        self.entity_index[entity.id] = entity
        self.graph.add_node(
            entity.id,
            name=entity.name,
            type=entity.type,
            **entity.attributes
        )
    
    def add_relation(self, relation: Relation):
        """添加关系"""
        self.relation_index[(relation.source_id, relation.target_id)].append(relation)
        self.graph.add_edge(
            relation.source_id,
            relation.target_id,
            relation_type=relation.relation_type,
            weight=relation.confidence,
            **relation.attributes
        )
    
    def get_neighbors(
        self,
        entity_id: str,
        max_hops: int = 2,
        relation_types: Optional[List[str]] = None
    ) -> List[GraphNode]:
        """获取邻居节点"""
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        visited = set()
        queue = [(entity_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_hops:
                continue
            
            visited.add(current_id)
            
            if depth > 0:  # 不包括起始节点
                entity = self.entity_index.get(current_id)
                if entity:
                    node = GraphNode(
                        entity=entity,
                        neighbors=set(self.graph.neighbors(current_id))
                    )
                    neighbors.append(node)
            
            # 添加下一层邻居
            if depth < max_hops:
                for neighbor_id in self.graph.neighbors(current_id):
                    if relation_types:
                        # 检查关系类型
                        edge_data = self.graph.get_edge_data(current_id, neighbor_id)
                        if edge_data.get("relation_type") not in relation_types:
                            continue
                    
                    queue.append((neighbor_id, depth + 1))
        
        return neighbors
    
    def find_path(
        self,
        source_id: str,
        target_id: str,
        max_length: int = 5
    ) -> Optional[GraphPath]:
        """查找两个实体间的路径"""
        if source_id not in self.graph or target_id not in self.graph:
            return None
        
        try:
            # 使用Dijkstra算法找最短路径
            path_nodes = nx.shortest_path(
                self.graph,
                source_id,
                target_id,
                weight='weight'
            )
            
            if len(path_nodes) > max_length:
                return None
            
            # 构建路径
            edges = []
            score = 1.0
            
            for i in range(len(path_nodes) - 1):
                source = path_nodes[i]
                target = path_nodes[i + 1]
                edge_data = self.graph.get_edge_data(source, target)
                
                edges.append((
                    source,
                    target,
                    edge_data.get("relation_type", "related")
                ))
                
                score *= edge_data.get("weight", 1.0)
            
            return GraphPath(
                nodes=path_nodes,
                edges=edges,
                score=score
            )
            
        except nx.NetworkXNoPath:
            return None
    
    def detect_communities(self) -> Dict[int, Set[str]]:
        """检测社区"""
        # 使用Louvain算法检测社区
        import community.community_louvain as community_louvain
        
        # 转换为无向图
        undirected_graph = self.graph.to_undirected()
        
        # 检测社区
        partition = community_louvain.best_partition(undirected_graph)
        
        # 整理社区结果
        communities = defaultdict(set)
        for node_id, community_id in partition.items():
            communities[community_id].add(node_id)
            
            # 更新节点的社区信息
            if node_id in self.entity_index:
                self.graph.nodes[node_id]['community'] = community_id
        
        self.communities = dict(communities)
        return self.communities
    
    def calculate_centrality(self) -> Dict[str, float]:
        """计算节点中心性"""
        # PageRank中心性
        pagerank = nx.pagerank(self.graph)
        
        # 度中心性
        degree_centrality = nx.degree_centrality(self.graph)
        
        # 介数中心性
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # 综合中心性
        centrality = {}
        for node_id in self.graph.nodes():
            centrality[node_id] = (
                pagerank.get(node_id, 0) * 0.4 +
                degree_centrality.get(node_id, 0) * 0.3 +
                betweenness_centrality.get(node_id, 0) * 0.3
            )
            
            # 更新节点属性
            self.graph.nodes[node_id]['centrality'] = centrality[node_id]
        
        return centrality
    
    async def persist_to_neo4j(self):
        """持久化到Neo4j"""
        if not self.neo4j_driver:
            return
        
        async with self.neo4j_driver.session() as session:
            # 创建实体节点
            for entity_id, entity in self.entity_index.items():
                await session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.name = $name,
                        e.type = $type,
                        e.attributes = $attributes
                    """,
                    id=entity_id,
                    name=entity.name,
                    type=entity.type,
                    attributes=json.dumps(entity.attributes)
                )
            
            # 创建关系
            for (source_id, target_id), relations in self.relation_index.items():
                for relation in relations:
                    await session.run(
                        """
                        MATCH (s:Entity {id: $source_id})
                        MATCH (t:Entity {id: $target_id})
                        MERGE (s)-[r:RELATED {type: $relation_type}]->(t)
                        SET r.confidence = $confidence,
                            r.attributes = $attributes
                        """,
                        source_id=source_id,
                        target_id=target_id,
                        relation_type=relation.relation_type,
                        confidence=relation.confidence,
                        attributes=json.dumps(relation.attributes)
                    )


class GraphRAG:
    """图增强检索系统"""
    
    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        milvus_collection: Optional[Collection] = None,
        neo4j_uri: Optional[str] = None
    ):
        self.llm = llm_service
        self.embedding_service = embedding_service
        self.milvus_collection = milvus_collection
        
        # 初始化组件
        self.entity_extractor = EntityExtractor(llm_service)
        self.relation_extractor = RelationExtractor(llm_service)
        self.knowledge_graph = KnowledgeGraph(neo4j_uri)
        
        # 缓存
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.info("GraphRAG系统初始化完成")
    
    async def build_knowledge_graph(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 10
    ):
        """构建知识图谱"""
        logger.info(f"开始构建知识图谱，文档数: {len(documents)}")
        
        # 批处理文档
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # 并行处理批次
            tasks = []
            for doc in batch:
                tasks.append(self._process_document(doc))
            
            results = await asyncio.gather(*tasks)
            
            # 添加到图谱
            for entities, relations in results:
                for entity in entities:
                    self.knowledge_graph.add_entity(entity)
                
                for relation in relations:
                    self.knowledge_graph.add_relation(relation)
        
        # 计算图属性
        self.knowledge_graph.calculate_centrality()
        self.knowledge_graph.detect_communities()
        
        # 持久化
        await self.knowledge_graph.persist_to_neo4j()
        
        logger.info(f"知识图谱构建完成，节点数: {len(self.knowledge_graph.entity_index)}")
    
    async def _process_document(
        self,
        document: Dict[str, Any]
    ) -> Tuple[List[Entity], List[Relation]]:
        """处理单个文档"""
        text = document.get("content", "")
        
        # 抽取实体
        entities = await self.entity_extractor.extract_entities(text)
        
        # 为实体生成嵌入
        for entity in entities:
            if entity.name not in self.embedding_cache:
                embedding = await self.embedding_service.embed_text(entity.name)
                self.embedding_cache[entity.name] = embedding
                entity.embedding = embedding
        
        # 抽取关系
        relations = await self.relation_extractor.extract_relations(entities, text)
        
        return entities, relations
    
    async def graph_enhanced_retrieval(
        self,
        query: str,
        top_k: int = 5,
        use_graph_traversal: bool = True,
        use_community_search: bool = True,
        use_path_reasoning: bool = True
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """图增强检索"""
        metadata = {
            "query": query,
            "techniques_used": []
        }
        
        all_results = []
        
        # 1. 从查询中抽取实体
        query_entities = await self.entity_extractor.extract_entities(query)
        metadata["query_entities"] = [e.name for e in query_entities]
        
        # 2. 向量检索（基线）
        if self.milvus_collection:
            vector_results = await self._vector_search(query, top_k * 2)
            all_results.extend(vector_results)
            metadata["techniques_used"].append("vector_search")
        
        # 3. 图遍历检索
        if use_graph_traversal and query_entities:
            graph_results = await self._graph_traversal_search(
                query_entities,
                max_hops=2
            )
            all_results.extend(graph_results)
            metadata["techniques_used"].append("graph_traversal")
        
        # 4. 社区检索
        if use_community_search and query_entities:
            community_results = await self._community_search(query_entities)
            all_results.extend(community_results)
            metadata["techniques_used"].append("community_search")
        
        # 5. 路径推理
        if use_path_reasoning and len(query_entities) >= 2:
            path_results = await self._path_reasoning_search(query_entities)
            all_results.extend(path_results)
            metadata["techniques_used"].append("path_reasoning")
        
        # 6. 融合排序
        final_results = self._fusion_ranking(all_results, top_k)
        
        metadata["total_retrieved"] = len(all_results)
        metadata["final_count"] = len(final_results)
        
        return final_results, metadata
    
    async def _vector_search(
        self,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        if not self.milvus_collection:
            return []
        
        # 生成查询向量
        query_vector = await self.embedding_service.embed_text(query)
        
        # Milvus检索
        search_params = {
            "metric_type": "IP",
            "params": {"nprobe": 10}
        }
        
        results = self.milvus_collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content", "metadata"]
        )
        
        search_results = []
        for hit in results[0]:
            search_results.append({
                "content": hit.entity.get("content"),
                "score": hit.score,
                "source": "vector_search",
                "metadata": hit.entity.get("metadata", {})
            })
        
        return search_results
    
    async def _graph_traversal_search(
        self,
        query_entities: List[Entity],
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        """图遍历检索"""
        results = []
        
        for entity in query_entities:
            # 查找图中的匹配实体
            matched_entities = self._find_matching_entities(entity)
            
            for matched_id in matched_entities:
                # 获取邻居节点
                neighbors = self.knowledge_graph.get_neighbors(
                    matched_id,
                    max_hops=max_hops
                )
                
                for neighbor in neighbors:
                    # 计算相关性分数
                    score = self._calculate_relevance_score(
                        entity,
                        neighbor.entity,
                        hops=len(neighbor.neighbors)
                    )
                    
                    results.append({
                        "content": self._entity_to_content(neighbor.entity),
                        "score": score,
                        "source": "graph_traversal",
                        "metadata": {
                            "entity": neighbor.entity.name,
                            "centrality": neighbor.entity.attributes.get("centrality", 0)
                        }
                    })
        
        return results
    
    async def _community_search(
        self,
        query_entities: List[Entity]
    ) -> List[Dict[str, Any]]:
        """社区检索"""
        results = []
        
        # 找出查询实体所属的社区
        query_communities = set()
        for entity in query_entities:
            matched_entities = self._find_matching_entities(entity)
            for matched_id in matched_entities:
                community = self.knowledge_graph.graph.nodes[matched_id].get("community")
                if community is not None:
                    query_communities.add(community)
        
        # 检索社区内的其他实体
        for community_id in query_communities:
            community_members = self.knowledge_graph.communities.get(community_id, set())
            
            for member_id in community_members:
                entity = self.knowledge_graph.entity_index.get(member_id)
                if entity:
                    # 计算社区内的相关性
                    score = self._calculate_community_relevance(entity, query_entities)
                    
                    results.append({
                        "content": self._entity_to_content(entity),
                        "score": score,
                        "source": "community_search",
                        "metadata": {
                            "entity": entity.name,
                            "community": community_id
                        }
                    })
        
        return results
    
    async def _path_reasoning_search(
        self,
        query_entities: List[Entity]
    ) -> List[Dict[str, Any]]:
        """路径推理检索"""
        results = []
        
        # 对查询实体两两之间找路径
        for i in range(len(query_entities)):
            for j in range(i + 1, len(query_entities)):
                entity1 = query_entities[i]
                entity2 = query_entities[j]
                
                # 查找匹配的实体
                matched1 = self._find_matching_entities(entity1)
                matched2 = self._find_matching_entities(entity2)
                
                for id1 in matched1:
                    for id2 in matched2:
                        # 查找路径
                        path = self.knowledge_graph.find_path(id1, id2)
                        
                        if path:
                            # 生成路径解释
                            explanation = await self._generate_path_explanation(path)
                            
                            results.append({
                                "content": explanation,
                                "score": path.score,
                                "source": "path_reasoning",
                                "metadata": {
                                    "path": [
                                        self.knowledge_graph.entity_index[node_id].name
                                        for node_id in path.nodes
                                    ],
                                    "relations": [
                                        edge[2] for edge in path.edges
                                    ]
                                }
                            })
        
        return results
    
    def _find_matching_entities(self, entity: Entity) -> List[str]:
        """查找匹配的实体"""
        matches = []
        
        for entity_id, stored_entity in self.knowledge_graph.entity_index.items():
            # 名称匹配
            if entity.name.lower() in stored_entity.name.lower() or \
               stored_entity.name.lower() in entity.name.lower():
                matches.append(entity_id)
            
            # 向量相似度匹配
            elif entity.embedding is not None and stored_entity.embedding is not None:
                similarity = np.dot(entity.embedding, stored_entity.embedding)
                if similarity > 0.8:
                    matches.append(entity_id)
        
        return matches
    
    def _calculate_relevance_score(
        self,
        query_entity: Entity,
        result_entity: Entity,
        hops: int
    ) -> float:
        """计算相关性分数"""
        # 基础分数
        base_score = 1.0 / (1 + hops)  # 距离衰减
        
        # 类型匹配加分
        type_bonus = 0.2 if query_entity.type == result_entity.type else 0
        
        # 中心性加分
        centrality_bonus = result_entity.attributes.get("centrality", 0) * 0.3
        
        return base_score + type_bonus + centrality_bonus
    
    def _calculate_community_relevance(
        self,
        entity: Entity,
        query_entities: List[Entity]
    ) -> float:
        """计算社区相关性"""
        # 简化实现：基于实体类型和属性的匹配度
        score = 0.5  # 基础分数
        
        for query_entity in query_entities:
            if entity.type == query_entity.type:
                score += 0.2
        
        return min(score, 1.0)
    
    def _entity_to_content(self, entity: Entity) -> str:
        """实体转换为内容"""
        content = f"{entity.name} ({entity.type})"
        
        if entity.attributes:
            attrs = ", ".join([f"{k}: {v}" for k, v in entity.attributes.items()])
            content += f" - {attrs}"
        
        return content
    
    async def _generate_path_explanation(self, path: GraphPath) -> str:
        """生成路径解释"""
        nodes_text = " -> ".join([
            self.knowledge_graph.entity_index[node_id].name
            for node_id in path.nodes
        ])
        
        relations_text = ", ".join([edge[2] for edge in path.edges])
        
        return f"路径: {nodes_text} (关系: {relations_text})"
    
    def _fusion_ranking(
        self,
        results: List[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """融合排序"""
        # 按来源分组
        by_source = defaultdict(list)
        for result in results:
            by_source[result["source"]].append(result)
        
        # 归一化分数
        for source, source_results in by_source.items():
            if not source_results:
                continue
            
            scores = [r["score"] for r in source_results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score > min_score:
                for result in source_results:
                    result["normalized_score"] = (
                        (result["score"] - min_score) / (max_score - min_score)
                    )
            else:
                for result in source_results:
                    result["normalized_score"] = 0.5
        
        # 加权融合
        weights = {
            "vector_search": 0.3,
            "graph_traversal": 0.3,
            "community_search": 0.2,
            "path_reasoning": 0.2
        }
        
        # 去重并计算最终分数
        unique_results = {}
        for result in results:
            content_hash = hash(result["content"])
            
            if content_hash not in unique_results:
                result["final_score"] = (
                    result["normalized_score"] * 
                    weights.get(result["source"], 0.25)
                )
                unique_results[content_hash] = result
            else:
                # 累加分数
                unique_results[content_hash]["final_score"] += (
                    result["normalized_score"] * 
                    weights.get(result["source"], 0.25)
                )
        
        # 排序并返回
        final_results = sorted(
            unique_results.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )
        
        return final_results[:top_k]
