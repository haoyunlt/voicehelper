"""
文档解析模块
支持PDF、Word、PPT、Excel等文档的智能解析和内容提取
"""

import asyncio
import io
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import zipfile

from loguru import logger


class DocumentType(Enum):
    """文档类型"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    PPTX = "pptx"
    PPT = "ppt"
    XLSX = "xlsx"
    XLS = "xls"
    TXT = "txt"
    RTF = "rtf"
    ODT = "odt"
    ODP = "odp"
    ODS = "ods"


class ContentType(Enum):
    """内容类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    SHAPE = "shape"
    HYPERLINK = "hyperlink"
    METADATA = "metadata"


@dataclass
class DocumentMetadata:
    """文档元数据"""
    title: str = ""
    author: str = ""
    subject: str = ""
    keywords: str = ""
    creator: str = ""
    producer: str = ""
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    page_count: int = 0
    word_count: int = 0
    character_count: int = 0
    language: str = ""
    file_size: int = 0
    custom_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TextContent:
    """文本内容"""
    text: str
    page_number: int
    paragraph_index: int
    font_name: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False
    color: str = ""
    alignment: str = ""
    bounding_box: Optional[Tuple[float, float, float, float]] = None


@dataclass
class TableContent:
    """表格内容"""
    rows: List[List[str]]
    headers: List[str]
    page_number: int
    table_index: int
    row_count: int
    column_count: int
    has_header: bool = False
    caption: str = ""
    bounding_box: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ImageContent:
    """图像内容"""
    image_data: bytes
    image_format: str
    page_number: int
    image_index: int
    width: int
    height: int
    caption: str = ""
    alt_text: str = ""
    bounding_box: Optional[Tuple[float, float, float, float]] = None


@dataclass
class ParsedDocument:
    """解析后的文档"""
    document_type: DocumentType
    metadata: DocumentMetadata
    text_contents: List[TextContent]
    table_contents: List[TableContent]
    image_contents: List[ImageContent]
    hyperlinks: List[Dict[str, Any]]
    structure: Dict[str, Any]
    processing_time: float
    success: bool = True
    error_message: str = ""


class PDFParser:
    """PDF解析器"""
    
    def __init__(self):
        self.max_pages = 1000  # 最大页数限制
        self.extract_images = True
        self.extract_tables = True
    
    async def parse(self, file_path: str) -> ParsedDocument:
        """解析PDF文档"""
        start_time = time.time()
        
        try:
            # 这里应该使用真实的PDF解析库，如PyMuPDF、pdfplumber等
            # 为了示例，我们模拟解析过程
            
            await asyncio.sleep(1.0)  # 模拟处理时间
            
            # 模拟元数据
            metadata = DocumentMetadata(
                title="示例PDF文档",
                author="作者姓名",
                subject="文档主题",
                creation_date=datetime.now(),
                page_count=10,
                word_count=2500,
                character_count=15000,
                language="zh-CN",
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )
            
            # 模拟文本内容
            text_contents = []
            for page in range(1, 6):  # 模拟5页
                text_contents.append(TextContent(
                    text=f"这是第{page}页的文本内容。包含了重要的信息和数据分析结果。",
                    page_number=page,
                    paragraph_index=1,
                    font_name="Arial",
                    font_size=12.0,
                    bounding_box=(50, 100, 500, 120)
                ))
            
            # 模拟表格内容
            table_contents = [
                TableContent(
                    rows=[
                        ["项目", "数量", "价格"],
                        ["产品A", "100", "50.00"],
                        ["产品B", "200", "75.00"]
                    ],
                    headers=["项目", "数量", "价格"],
                    page_number=2,
                    table_index=1,
                    row_count=3,
                    column_count=3,
                    has_header=True,
                    caption="产品价格表"
                )
            ]
            
            # 模拟图像内容
            image_contents = []
            if self.extract_images:
                # 创建一个小的示例图像
                dummy_image = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xdb\x00\x00\x00\x00IEND\xaeB`\x82'
                
                image_contents.append(ImageContent(
                    image_data=dummy_image,
                    image_format="PNG",
                    page_number=3,
                    image_index=1,
                    width=100,
                    height=100,
                    caption="示例图表",
                    alt_text="数据可视化图表"
                ))
            
            # 模拟超链接
            hyperlinks = [
                {
                    "text": "参考链接",
                    "url": "https://example.com",
                    "page_number": 1,
                    "bounding_box": (100, 200, 200, 220)
                }
            ]
            
            # 文档结构
            structure = {
                "outline": [
                    {"title": "第一章 概述", "page": 1, "level": 1},
                    {"title": "1.1 背景", "page": 1, "level": 2},
                    {"title": "第二章 分析", "page": 3, "level": 1},
                    {"title": "2.1 数据分析", "page": 3, "level": 2},
                    {"title": "2.2 结果讨论", "page": 5, "level": 2}
                ],
                "page_layouts": [
                    {"page": 1, "layout": "title_page"},
                    {"page": 2, "layout": "text_with_table"},
                    {"page": 3, "layout": "text_with_image"}
                ]
            }
            
            processing_time = time.time() - start_time
            
            return ParsedDocument(
                document_type=DocumentType.PDF,
                metadata=metadata,
                text_contents=text_contents,
                table_contents=table_contents,
                image_contents=image_contents,
                hyperlinks=hyperlinks,
                structure=structure,
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PDF解析失败: {e}")
            
            return ParsedDocument(
                document_type=DocumentType.PDF,
                metadata=DocumentMetadata(),
                text_contents=[],
                table_contents=[],
                image_contents=[],
                hyperlinks=[],
                structure={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class WordParser:
    """Word文档解析器"""
    
    def __init__(self):
        self.extract_images = True
        self.extract_tables = True
        self.preserve_formatting = True
    
    async def parse(self, file_path: str) -> ParsedDocument:
        """解析Word文档"""
        start_time = time.time()
        
        try:
            # 这里应该使用python-docx或其他Word解析库
            # 为了示例，我们模拟解析过程
            
            await asyncio.sleep(0.8)  # 模拟处理时间
            
            # 检查文件类型
            doc_type = DocumentType.DOCX if file_path.endswith('.docx') else DocumentType.DOC
            
            # 模拟解析结果
            metadata = DocumentMetadata(
                title="示例Word文档",
                author="文档作者",
                subject="技术文档",
                creation_date=datetime.now(),
                page_count=5,
                word_count=1800,
                character_count=10800,
                language="zh-CN"
            )
            
            # 模拟文本内容
            text_contents = [
                TextContent(
                    text="文档标题",
                    page_number=1,
                    paragraph_index=1,
                    font_name="宋体",
                    font_size=16.0,
                    is_bold=True,
                    alignment="center"
                ),
                TextContent(
                    text="这是文档的正文内容，包含了详细的技术说明和操作指南。",
                    page_number=1,
                    paragraph_index=2,
                    font_name="宋体",
                    font_size=12.0,
                    alignment="left"
                )
            ]
            
            # 模拟表格内容
            table_contents = [
                TableContent(
                    rows=[
                        ["功能", "状态", "备注"],
                        ["用户登录", "已完成", "支持多种方式"],
                        ["数据导出", "开发中", "预计下周完成"]
                    ],
                    headers=["功能", "状态", "备注"],
                    page_number=2,
                    table_index=1,
                    row_count=3,
                    column_count=3,
                    has_header=True
                )
            ]
            
            processing_time = time.time() - start_time
            
            return ParsedDocument(
                document_type=doc_type,
                metadata=metadata,
                text_contents=text_contents,
                table_contents=table_contents,
                image_contents=[],
                hyperlinks=[],
                structure={"sections": ["标题", "正文", "表格", "结论"]},
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Word文档解析失败: {e}")
            
            return ParsedDocument(
                document_type=DocumentType.DOCX,
                metadata=DocumentMetadata(),
                text_contents=[],
                table_contents=[],
                image_contents=[],
                hyperlinks=[],
                structure={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class PowerPointParser:
    """PowerPoint解析器"""
    
    def __init__(self):
        self.extract_images = True
        self.extract_speaker_notes = True
    
    async def parse(self, file_path: str) -> ParsedDocument:
        """解析PowerPoint文档"""
        start_time = time.time()
        
        try:
            # 这里应该使用python-pptx或其他PPT解析库
            await asyncio.sleep(0.6)  # 模拟处理时间
            
            doc_type = DocumentType.PPTX if file_path.endswith('.pptx') else DocumentType.PPT
            
            # 模拟解析结果
            metadata = DocumentMetadata(
                title="项目演示文稿",
                author="演示者",
                subject="项目汇报",
                creation_date=datetime.now(),
                page_count=15,  # 幻灯片数量
                language="zh-CN"
            )
            
            # 模拟幻灯片内容
            text_contents = []
            for slide_num in range(1, 6):
                text_contents.extend([
                    TextContent(
                        text=f"幻灯片 {slide_num} 标题",
                        page_number=slide_num,
                        paragraph_index=1,
                        font_name="微软雅黑",
                        font_size=24.0,
                        is_bold=True
                    ),
                    TextContent(
                        text=f"这是第{slide_num}张幻灯片的内容要点。包含关键信息和数据展示。",
                        page_number=slide_num,
                        paragraph_index=2,
                        font_name="微软雅黑",
                        font_size=18.0
                    )
                ])
            
            processing_time = time.time() - start_time
            
            return ParsedDocument(
                document_type=doc_type,
                metadata=metadata,
                text_contents=text_contents,
                table_contents=[],
                image_contents=[],
                hyperlinks=[],
                structure={
                    "slides": [
                        {"slide": i, "title": f"幻灯片 {i} 标题", "layout": "title_content"}
                        for i in range(1, 6)
                    ]
                },
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"PowerPoint解析失败: {e}")
            
            return ParsedDocument(
                document_type=DocumentType.PPTX,
                metadata=DocumentMetadata(),
                text_contents=[],
                table_contents=[],
                image_contents=[],
                hyperlinks=[],
                structure={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class ExcelParser:
    """Excel解析器"""
    
    def __init__(self):
        self.max_rows = 10000  # 最大行数限制
        self.max_sheets = 50   # 最大工作表数量
    
    async def parse(self, file_path: str) -> ParsedDocument:
        """解析Excel文档"""
        start_time = time.time()
        
        try:
            # 这里应该使用openpyxl、xlrd或pandas等库
            await asyncio.sleep(0.5)  # 模拟处理时间
            
            doc_type = DocumentType.XLSX if file_path.endswith('.xlsx') else DocumentType.XLS
            
            # 模拟解析结果
            metadata = DocumentMetadata(
                title="数据分析表格",
                author="数据分析师",
                creation_date=datetime.now(),
                page_count=3,  # 工作表数量
                language="zh-CN"
            )
            
            # 模拟表格内容（每个工作表作为一个表格）
            table_contents = [
                TableContent(
                    rows=[
                        ["日期", "销售额", "利润", "增长率"],
                        ["2024-01", "100000", "20000", "5%"],
                        ["2024-02", "120000", "25000", "20%"],
                        ["2024-03", "110000", "22000", "-8%"]
                    ],
                    headers=["日期", "销售额", "利润", "增长率"],
                    page_number=1,  # 工作表1
                    table_index=1,
                    row_count=4,
                    column_count=4,
                    has_header=True,
                    caption="销售数据表"
                ),
                TableContent(
                    rows=[
                        ["产品", "库存", "成本", "售价"],
                        ["产品A", "500", "30", "50"],
                        ["产品B", "300", "45", "75"],
                        ["产品C", "200", "60", "100"]
                    ],
                    headers=["产品", "库存", "成本", "售价"],
                    page_number=2,  # 工作表2
                    table_index=2,
                    row_count=4,
                    column_count=4,
                    has_header=True,
                    caption="产品信息表"
                )
            ]
            
            processing_time = time.time() - start_time
            
            return ParsedDocument(
                document_type=doc_type,
                metadata=metadata,
                text_contents=[],
                table_contents=table_contents,
                image_contents=[],
                hyperlinks=[],
                structure={
                    "worksheets": [
                        {"name": "销售数据", "index": 1, "rows": 4, "columns": 4},
                        {"name": "产品信息", "index": 2, "rows": 4, "columns": 4},
                        {"name": "图表分析", "index": 3, "rows": 0, "columns": 0}
                    ]
                },
                processing_time=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Excel解析失败: {e}")
            
            return ParsedDocument(
                document_type=DocumentType.XLSX,
                metadata=DocumentMetadata(),
                text_contents=[],
                table_contents=[],
                image_contents=[],
                hyperlinks=[],
                structure={},
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )


class DocumentParserService:
    """文档解析服务"""
    
    def __init__(self):
        self.pdf_parser = PDFParser()
        self.word_parser = WordParser()
        self.ppt_parser = PowerPointParser()
        self.excel_parser = ExcelParser()
        
        # 支持的文档类型
        self.supported_types = {
            '.pdf': DocumentType.PDF,
            '.docx': DocumentType.DOCX,
            '.doc': DocumentType.DOC,
            '.pptx': DocumentType.PPTX,
            '.ppt': DocumentType.PPT,
            '.xlsx': DocumentType.XLSX,
            '.xls': DocumentType.XLS,
            '.txt': DocumentType.TXT,
            '.rtf': DocumentType.RTF
        }
        
        # 解析器映射
        self.parsers = {
            DocumentType.PDF: self.pdf_parser,
            DocumentType.DOCX: self.word_parser,
            DocumentType.DOC: self.word_parser,
            DocumentType.PPTX: self.ppt_parser,
            DocumentType.PPT: self.ppt_parser,
            DocumentType.XLSX: self.excel_parser,
            DocumentType.XLS: self.excel_parser
        }
        
        # 统计信息
        self.stats = {
            "total_documents": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "avg_processing_time": 0.0,
            "type_stats": {doc_type.value: {"count": 0, "avg_time": 0.0} for doc_type in DocumentType}
        }
    
    def detect_document_type(self, file_path: str) -> Optional[DocumentType]:
        """检测文档类型"""
        try:
            file_extension = Path(file_path).suffix.lower()
            return self.supported_types.get(file_extension)
        except Exception as e:
            logger.error(f"文档类型检测失败: {e}")
            return None
    
    async def parse_document(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
        options: Dict[str, Any] = None
    ) -> ParsedDocument:
        """解析文档"""
        start_time = time.time()
        options = options or {}
        
        try:
            # 检测文档类型
            if document_type is None:
                document_type = self.detect_document_type(file_path)
                if document_type is None:
                    raise ValueError(f"不支持的文档类型: {file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            # 获取对应的解析器
            parser = self.parsers.get(document_type)
            if parser is None:
                raise ValueError(f"暂不支持解析 {document_type.value} 类型的文档")
            
            # 执行解析
            logger.info(f"开始解析文档: {file_path} (类型: {document_type.value})")
            result = await parser.parse(file_path)
            
            # 更新统计信息
            total_time = time.time() - start_time
            self._update_stats(document_type, total_time, result.success)
            
            logger.info(f"文档解析完成: {file_path}, 耗时: {total_time:.2f}s, 成功: {result.success}")
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"文档解析失败: {file_path}, 错误: {e}")
            
            # 更新统计信息
            if document_type:
                self._update_stats(document_type, total_time, False)
            
            return ParsedDocument(
                document_type=document_type or DocumentType.PDF,
                metadata=DocumentMetadata(),
                text_contents=[],
                table_contents=[],
                image_contents=[],
                hyperlinks=[],
                structure={},
                processing_time=total_time,
                success=False,
                error_message=str(e)
            )
    
    async def parse_document_from_bytes(
        self,
        file_data: bytes,
        filename: str,
        document_type: Optional[DocumentType] = None,
        options: Dict[str, Any] = None
    ) -> ParsedDocument:
        """从字节数据解析文档"""
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as temp_file:
                temp_file.write(file_data)
                temp_file_path = temp_file.name
            
            try:
                # 解析临时文件
                result = await self.parse_document(temp_file_path, document_type, options)
                return result
            finally:
                # 清理临时文件
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"从字节数据解析文档失败: {e}")
            return ParsedDocument(
                document_type=document_type or DocumentType.PDF,
                metadata=DocumentMetadata(),
                text_contents=[],
                table_contents=[],
                image_contents=[],
                hyperlinks=[],
                structure={},
                processing_time=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def batch_parse_documents(
        self,
        file_paths: List[str],
        options: Dict[str, Any] = None
    ) -> List[ParsedDocument]:
        """批量解析文档"""
        tasks = []
        for file_path in file_paths:
            task = self.parse_document(file_path, options=options)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ParsedDocument(
                    document_type=DocumentType.PDF,
                    metadata=DocumentMetadata(),
                    text_contents=[],
                    table_contents=[],
                    image_contents=[],
                    hyperlinks=[],
                    structure={},
                    processing_time=0.0,
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def extract_text_summary(self, document: ParsedDocument, max_length: int = 500) -> str:
        """提取文档文本摘要"""
        try:
            # 合并所有文本内容
            all_text = []
            for text_content in document.text_contents:
                all_text.append(text_content.text)
            
            full_text = " ".join(all_text)
            
            # 简单的摘要提取（实际应用中可以使用更复杂的算法）
            if len(full_text) <= max_length:
                return full_text
            
            # 截取前面的内容
            summary = full_text[:max_length]
            
            # 在最后一个句号处截断
            last_period = summary.rfind('。')
            if last_period > max_length // 2:
                summary = summary[:last_period + 1]
            
            return summary + "..."
            
        except Exception as e:
            logger.error(f"提取文本摘要失败: {e}")
            return ""
    
    def extract_structured_data(self, document: ParsedDocument) -> Dict[str, Any]:
        """提取结构化数据"""
        try:
            structured_data = {
                "metadata": {
                    "title": document.metadata.title,
                    "author": document.metadata.author,
                    "page_count": document.metadata.page_count,
                    "word_count": document.metadata.word_count,
                    "creation_date": document.metadata.creation_date.isoformat() if document.metadata.creation_date else None
                },
                "content_summary": {
                    "text_blocks": len(document.text_contents),
                    "tables": len(document.table_contents),
                    "images": len(document.image_contents),
                    "hyperlinks": len(document.hyperlinks)
                },
                "tables": [],
                "key_information": []
            }
            
            # 提取表格数据
            for table in document.table_contents:
                table_data = {
                    "page": table.page_number,
                    "caption": table.caption,
                    "rows": table.row_count,
                    "columns": table.column_count,
                    "headers": table.headers,
                    "data": table.rows[1:] if table.has_header else table.rows  # 排除标题行
                }
                structured_data["tables"].append(table_data)
            
            # 提取关键信息（简单的关键词提取）
            text_summary = self.extract_text_summary(document, 1000)
            key_phrases = self._extract_key_phrases(text_summary)
            structured_data["key_information"] = key_phrases
            
            return structured_data
            
        except Exception as e:
            logger.error(f"提取结构化数据失败: {e}")
            return {}
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[str]:
        """提取关键短语（简单实现）"""
        try:
            # 这里应该使用更复杂的NLP技术，如TF-IDF、TextRank等
            # 简单实现：提取常见的重要词汇
            
            import re
            
            # 移除标点符号，分词
            words = re.findall(r'\b\w+\b', text.lower())
            
            # 过滤停用词（简化版）
            stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这'}
            
            filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
            
            # 统计词频
            word_freq = {}
            for word in filtered_words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # 按频率排序，取前N个
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            key_phrases = [word for word, freq in sorted_words[:max_phrases]]
            
            return key_phrases
            
        except Exception as e:
            logger.error(f"关键短语提取失败: {e}")
            return []
    
    def _update_stats(self, document_type: DocumentType, processing_time: float, success: bool):
        """更新统计信息"""
        self.stats["total_documents"] += 1
        
        if success:
            self.stats["successful_parses"] += 1
        else:
            self.stats["failed_parses"] += 1
        
        # 更新平均处理时间
        total_docs = self.stats["total_documents"]
        current_avg = self.stats["avg_processing_time"]
        self.stats["avg_processing_time"] = (
            (current_avg * (total_docs - 1) + processing_time) / total_docs
        )
        
        # 更新类型统计
        type_stats = self.stats["type_stats"][document_type.value]
        type_stats["count"] += 1
        current_type_avg = type_stats["avg_time"]
        type_count = type_stats["count"]
        type_stats["avg_time"] = (
            (current_type_avg * (type_count - 1) + processing_time) / type_count
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()
    
    def get_supported_types(self) -> List[str]:
        """获取支持的文档类型"""
        return list(self.supported_types.keys())


# 全局文档解析服务实例
document_parser_service = DocumentParserService()
