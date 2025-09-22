"""
动态知识更新系统
实时信息获取和知识图谱构建
支持多源数据集成、知识抽取和图谱更新
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import feedparser
import networkx as nx
from datetime import datetime, timedelta
import re
from urllib.parse import urljoin, urlparse
import hashlib

logger = logging.getLogger(__name__)

class KnowledgeSourceType(Enum):
    RSS_FEED = "rss_feed"
    API_ENDPOINT = "api_endpoint"
    WEB_SCRAPING = "web_scraping"
    DATABASE = "database"
    FILE_SYSTEM = "file_system"

class EntityType(Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    PRODUCT = "product"
    TECHNOLOGY = "technology"

@dataclass
class KnowledgeSource:
    id: str
    name: str
    source_type: KnowledgeSourceType
    url: str
    update_frequency: int  # 秒
    last_updated: float
    is_active: bool
    metadata: Dict[str, Any]
    
@dataclass
class KnowledgeEntity:
    id: str
    name: str
    entity_type: EntityType
    description: str
    properties: Dict[str, Any]
    confidence: float
    source_id: str
    created_at: float
    updated_at: float

@dataclass
class KnowledgeRelation:
    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: Dict[str, Any]
    confidence: float
    source_id: str
    created_at: float

class DynamicKnowledgeSystem:
    """动态知识系统"""
    
    def __init__(self):
        self.knowledge_graph = nx.MultiDiGraph()
        self.entities: Dict[str, KnowledgeEntity] = {}
        self.relations: Dict[str, KnowledgeRelation] = {}
        self.sources: Dict[str, KnowledgeSource] = {}
        
        # 实时更新队列
        self.update_queue = asyncio.Queue()
        self.processing_tasks: Set[asyncio.Task] = set()
        
        # 知识抽取器
        self.extractors = {
            KnowledgeSourceType.RSS_FEED: self._extract_from_rss,
            KnowledgeSourceType.API_ENDPOINT: self._extract_from_api,
            KnowledgeSourceType.WEB_SCRAPING: self._extract_from_web,
        }
        
        # HTTP客户端
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """初始化知识系统"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        
        # 启动更新任务
        asyncio.create_task(self._update_scheduler())
        asyncio.create_task(self._process_updates())
        
        logger.info("Dynamic knowledge system initialized")
    
    async def cleanup(self):
        """清理资源"""
        if self.session:
            await self.session.close()
        
        # 取消所有处理任务
        for task in self.processing_tasks:
            task.cancel()
    
    def register_source(self, source: KnowledgeSource):
        """注册知识源"""
        self.sources[source.id] = source
        logger.info(f"Registered knowledge source: {source.name}")
    
    async def update_knowledge_from_source(self, source_id: str) -> bool:
        """从指定源更新知识"""
        source = self.sources.get(source_id)
        if not source or not source.is_active:
            return False
        
        try:
            extractor = self.extractors.get(source.source_type)
            if not extractor:
                logger.warning(f"No extractor for source type: {source.source_type}")
                return False
            
            # 提取知识
            entities, relations = await extractor(source)
            
            # 更新知识图谱
            await self._update_knowledge_graph(entities, relations, source_id)
            
            # 更新源的最后更新时间
            source.last_updated = time.time()
            
            logger.info(f"Updated knowledge from source: {source.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update knowledge from source {source.name}: {e}")
            return False
    
    async def _extract_from_rss(self, source: KnowledgeSource) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """从RSS源提取知识"""
        entities = []
        relations = []
        
        try:
            async with self.session.get(source.url) as response:
                content = await response.text()
                
            feed = feedparser.parse(content)
            
            for entry in feed.entries[:10]:  # 限制处理数量
                # 创建文章实体
                article_id = self._generate_entity_id(entry.title)
                article_entity = KnowledgeEntity(
                    id=article_id,
                    name=entry.title,
                    entity_type=EntityType.EVENT,
                    description=entry.get('summary', ''),
                    properties={
                        'url': entry.link,
                        'published': entry.get('published', ''),
                        'author': entry.get('author', ''),
                        'tags': [tag.term for tag in entry.get('tags', [])]
                    },
                    confidence=0.8,
                    source_id=source.id,
                    created_at=time.time(),
                    updated_at=time.time()
                )
                entities.append(article_entity)
                
                # 提取实体和关系
                extracted_entities, extracted_relations = await self._extract_entities_from_text(
                    entry.title + " " + entry.get('summary', ''),
                    source.id,
                    article_id
                )
                
                entities.extend(extracted_entities)
                relations.extend(extracted_relations)
                
        except Exception as e:
            logger.error(f"RSS extraction error: {e}")
        
        return entities, relations
    
    async def _extract_from_api(self, source: KnowledgeSource) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """从API端点提取知识"""
        entities = []
        relations = []
        
        try:
            headers = source.metadata.get('headers', {})
            params = source.metadata.get('params', {})
            
            async with self.session.get(source.url, headers=headers, params=params) as response:
                data = await response.json()
                
            # 根据API响应结构解析数据
            if isinstance(data, dict):
                if 'items' in data:
                    items = data['items']
                elif 'results' in data:
                    items = data['results']
                else:
                    items = [data]
            else:
                items = data
            
            for item in items[:20]:  # 限制处理数量
                if isinstance(item, dict):
                    # 创建实体
                    entity_name = item.get('name') or item.get('title') or str(item.get('id', ''))
                    if entity_name:
                        entity_id = self._generate_entity_id(entity_name)
                        entity = KnowledgeEntity(
                            id=entity_id,
                            name=entity_name,
                            entity_type=EntityType.CONCEPT,
                            description=item.get('description', ''),
                            properties=item,
                            confidence=0.7,
                            source_id=source.id,
                            created_at=time.time(),
                            updated_at=time.time()
                        )
                        entities.append(entity)
                        
        except Exception as e:
            logger.error(f"API extraction error: {e}")
        
        return entities, relations
    
    async def _extract_from_web(self, source: KnowledgeSource) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """从网页抓取提取知识"""
        entities = []
        relations = []
        
        try:
            async with self.session.get(source.url) as response:
                html_content = await response.text()
            
            # 简单的HTML解析（实际应用中应使用BeautifulSoup等库）
            # 提取标题
            title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                
                title_entity = KnowledgeEntity(
                    id=self._generate_entity_id(title),
                    name=title,
                    entity_type=EntityType.CONCEPT,
                    description=f"Web page: {title}",
                    properties={'url': source.url},
                    confidence=0.6,
                    source_id=source.id,
                    created_at=time.time(),
                    updated_at=time.time()
                )
                entities.append(title_entity)
            
            # 提取文本内容进行实体识别
            text_content = re.sub(r'<[^>]+>', ' ', html_content)
            text_content = re.sub(r'\s+', ' ', text_content).strip()
            
            if len(text_content) > 100:
                extracted_entities, extracted_relations = await self._extract_entities_from_text(
                    text_content[:2000],  # 限制文本长度
                    source.id
                )
                entities.extend(extracted_entities)
                relations.extend(extracted_relations)
                
        except Exception as e:
            logger.error(f"Web scraping error: {e}")
        
        return entities, relations
    
    async def _extract_entities_from_text(self, text: str, source_id: str, 
                                        parent_entity_id: Optional[str] = None) -> Tuple[List[KnowledgeEntity], List[KnowledgeRelation]]:
        """从文本中提取实体和关系"""
        entities = []
        relations = []
        
        # 简化的实体识别（实际应用中应使用NER模型）
        
        # 识别人名（大写字母开头的连续词）
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        persons = re.findall(person_pattern, text)
        
        for person in set(persons):
            entity_id = self._generate_entity_id(person)
            entity = KnowledgeEntity(
                id=entity_id,
                name=person,
                entity_type=EntityType.PERSON,
                description=f"Person mentioned in text",
                properties={'context': text[:200]},
                confidence=0.6,
                source_id=source_id,
                created_at=time.time(),
                updated_at=time.time()
            )
            entities.append(entity)
            
            # 如果有父实体，创建关系
            if parent_entity_id:
                relation = KnowledgeRelation(
                    id=f"rel_{parent_entity_id}_{entity_id}",
                    source_entity_id=parent_entity_id,
                    target_entity_id=entity_id,
                    relation_type="mentions",
                    properties={},
                    confidence=0.5,
                    source_id=source_id,
                    created_at=time.time()
                )
                relations.append(relation)
        
        # 识别组织名（包含特定关键词）
        org_keywords = ['公司', '企业', '组织', '机构', 'Company', 'Corp', 'Inc', 'Ltd']
        for keyword in org_keywords:
            pattern = rf'\b[A-Z][a-zA-Z\s]*{keyword}\b'
            orgs = re.findall(pattern, text, re.IGNORECASE)
            
            for org in set(orgs):
                entity_id = self._generate_entity_id(org)
                entity = KnowledgeEntity(
                    id=entity_id,
                    name=org.strip(),
                    entity_type=EntityType.ORGANIZATION,
                    description=f"Organization mentioned in text",
                    properties={'context': text[:200]},
                    confidence=0.7,
                    source_id=source_id,
                    created_at=time.time(),
                    updated_at=time.time()
                )
                entities.append(entity)
        
        # 识别技术概念
        tech_keywords = ['AI', '人工智能', '机器学习', '深度学习', '区块链', '云计算', 'API', 'SDK']
        for keyword in tech_keywords:
            if keyword.lower() in text.lower():
                entity_id = self._generate_entity_id(keyword)
                entity = KnowledgeEntity(
                    id=entity_id,
                    name=keyword,
                    entity_type=EntityType.TECHNOLOGY,
                    description=f"Technology concept: {keyword}",
                    properties={'context': text[:200]},
                    confidence=0.8,
                    source_id=source_id,
                    created_at=time.time(),
                    updated_at=time.time()
                )
                entities.append(entity)
        
        return entities, relations
    
    async def _update_knowledge_graph(self, entities: List[KnowledgeEntity], 
                                    relations: List[KnowledgeRelation], source_id: str):
        """更新知识图谱"""
        
        # 添加或更新实体
        for entity in entities:
            if entity.id in self.entities:
                # 更新现有实体
                existing = self.entities[entity.id]
                existing.updated_at = time.time()
                existing.confidence = max(existing.confidence, entity.confidence)
                existing.properties.update(entity.properties)
            else:
                # 添加新实体
                self.entities[entity.id] = entity
                self.knowledge_graph.add_node(
                    entity.id,
                    name=entity.name,
                    type=entity.entity_type.value,
                    properties=entity.properties
                )
        
        # 添加关系
        for relation in relations:
            if (relation.source_entity_id in self.entities and 
                relation.target_entity_id in self.entities):
                
                self.relations[relation.id] = relation
                self.knowledge_graph.add_edge(
                    relation.source_entity_id,
                    relation.target_entity_id,
                    relation_type=relation.relation_type,
                    properties=relation.properties,
                    confidence=relation.confidence
                )
    
    def _generate_entity_id(self, name: str) -> str:
        """生成实体ID"""
        normalized_name = name.lower().strip()
        return hashlib.md5(normalized_name.encode()).hexdigest()[:16]
    
    async def query_knowledge(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """查询知识"""
        results = []
        query_lower = query.lower()
        
        # 搜索实体
        for entity in self.entities.values():
            if query_lower in entity.name.lower() or query_lower in entity.description.lower():
                # 获取相关关系
                related_entities = []
                if self.knowledge_graph.has_node(entity.id):
                    neighbors = list(self.knowledge_graph.neighbors(entity.id))
                    for neighbor_id in neighbors[:5]:  # 限制相关实体数量
                        if neighbor_id in self.entities:
                            related_entities.append({
                                'id': neighbor_id,
                                'name': self.entities[neighbor_id].name,
                                'type': self.entities[neighbor_id].entity_type.value
                            })
                
                results.append({
                    'entity': asdict(entity),
                    'related_entities': related_entities
                })
                
                if len(results) >= max_results:
                    break
        
        return results
    
    async def get_entity_relationships(self, entity_id: str) -> Dict[str, Any]:
        """获取实体关系"""
        if entity_id not in self.entities:
            return {}
        
        entity = self.entities[entity_id]
        relationships = {
            'incoming': [],
            'outgoing': []
        }
        
        if self.knowledge_graph.has_node(entity_id):
            # 入边关系
            for pred in self.knowledge_graph.predecessors(entity_id):
                edge_data = self.knowledge_graph[pred][entity_id]
                relationships['incoming'].append({
                    'source_entity': {
                        'id': pred,
                        'name': self.entities[pred].name if pred in self.entities else pred
                    },
                    'relation_type': edge_data.get('relation_type', 'unknown'),
                    'confidence': edge_data.get('confidence', 0.0)
                })
            
            # 出边关系
            for succ in self.knowledge_graph.successors(entity_id):
                edge_data = self.knowledge_graph[entity_id][succ]
                relationships['outgoing'].append({
                    'target_entity': {
                        'id': succ,
                        'name': self.entities[succ].name if succ in self.entities else succ
                    },
                    'relation_type': edge_data.get('relation_type', 'unknown'),
                    'confidence': edge_data.get('confidence', 0.0)
                })
        
        return {
            'entity': asdict(entity),
            'relationships': relationships
        }
    
    async def _update_scheduler(self):
        """更新调度器"""
        while True:
            try:
                current_time = time.time()
                
                for source in self.sources.values():
                    if (source.is_active and 
                        current_time - source.last_updated >= source.update_frequency):
                        
                        # 添加到更新队列
                        await self.update_queue.put(source.id)
                
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"Update scheduler error: {e}")
                await asyncio.sleep(30)
    
    async def _process_updates(self):
        """处理更新队列"""
        while True:
            try:
                source_id = await self.update_queue.get()
                
                # 创建更新任务
                task = asyncio.create_task(self.update_knowledge_from_source(source_id))
                self.processing_tasks.add(task)
                
                # 清理完成的任务
                done_tasks = [t for t in self.processing_tasks if t.done()]
                for task in done_tasks:
                    self.processing_tasks.remove(task)
                
            except Exception as e:
                logger.error(f"Update processing error: {e}")
                await asyncio.sleep(5)
    
    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """获取知识统计"""
        entity_types = {}
        for entity in self.entities.values():
            entity_type = entity.entity_type.value
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        return {
            'total_entities': len(self.entities),
            'total_relations': len(self.relations),
            'total_sources': len(self.sources),
            'active_sources': len([s for s in self.sources.values() if s.is_active]),
            'entity_types': entity_types,
            'graph_nodes': self.knowledge_graph.number_of_nodes(),
            'graph_edges': self.knowledge_graph.number_of_edges()
        }

# 使用示例
async def create_dynamic_knowledge_system():
    """创建动态知识系统"""
    knowledge_system = DynamicKnowledgeSystem()
    await knowledge_system.initialize()
    
    # 注册一些示例知识源
    rss_source = KnowledgeSource(
        id="tech_news_rss",
        name="Tech News RSS",
        source_type=KnowledgeSourceType.RSS_FEED,
        url="https://feeds.feedburner.com/TechCrunch",
        update_frequency=3600,  # 每小时更新
        last_updated=0,
        is_active=True,
        metadata={}
    )
    knowledge_system.register_source(rss_source)
    
    api_source = KnowledgeSource(
        id="github_api",
        name="GitHub API",
        source_type=KnowledgeSourceType.API_ENDPOINT,
        url="https://api.github.com/search/repositories",
        update_frequency=7200,  # 每2小时更新
        last_updated=0,
        is_active=True,
        metadata={
            'params': {'q': 'artificial intelligence', 'sort': 'updated'},
            'headers': {'Accept': 'application/vnd.github.v3+json'}
        }
    )
    knowledge_system.register_source(api_source)
    
    return knowledge_system

if __name__ == "__main__":
    # 测试代码
    async def test_knowledge_system():
        knowledge_system = await create_dynamic_knowledge_system()
        
        # 手动触发更新
        await knowledge_system.update_knowledge_from_source("tech_news_rss")
        
        # 查询知识
        results = await knowledge_system.query_knowledge("artificial intelligence")
        print("查询结果:")
        for result in results[:3]:
            print(f"- {result['entity']['name']}: {result['entity']['description'][:100]}...")
        
        # 获取统计信息
        stats = knowledge_system.get_knowledge_statistics()
        print(f"\n知识统计: {json.dumps(stats, indent=2)}")
        
        await knowledge_system.cleanup()
    
    asyncio.run(test_knowledge_system())
