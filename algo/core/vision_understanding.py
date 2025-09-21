"""
视觉理解系统 - v1.8.0
支持图像理解、分析和描述的多模态视觉能力
"""

import asyncio
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import base64
import io
import time
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class ImageType(Enum):
    """图像类型"""
    PHOTO = "photo"              # 照片
    SCREENSHOT = "screenshot"    # 屏幕截图
    DOCUMENT = "document"        # 文档
    CHART = "chart"             # 图表
    DIAGRAM = "diagram"         # 图解
    ARTWORK = "artwork"         # 艺术作品
    MEME = "meme"              # 表情包
    UNKNOWN = "unknown"         # 未知类型

class VisionTask(Enum):
    """视觉任务类型"""
    DESCRIPTION = "description"           # 图像描述
    OBJECT_DETECTION = "object_detection" # 物体检测
    TEXT_EXTRACTION = "text_extraction"   # 文字提取
    SCENE_ANALYSIS = "scene_analysis"     # 场景分析
    CHART_ANALYSIS = "chart_analysis"     # 图表分析
    FACE_ANALYSIS = "face_analysis"       # 人脸分析
    SIMILARITY_SEARCH = "similarity"      # 相似度搜索

@dataclass
class DetectedObject:
    """检测到的物体"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    attributes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExtractedText:
    """提取的文本"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    language: str = "zh"

@dataclass
class VisionAnalysisResult:
    """视觉分析结果"""
    image_type: ImageType
    description: str
    objects: List[DetectedObject] = field(default_factory=list)
    extracted_texts: List[ExtractedText] = field(default_factory=list)
    scene_attributes: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    confidence: float = 0.0

class ImagePreprocessor:
    """图像预处理器"""
    
    def __init__(self):
        self.max_size = (1024, 1024)
        self.min_size = (224, 224)
        
        # 标准化变换
        self.normalize_transform = transforms.Compose([
            transforms.Resize(self.min_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    async def preprocess_image(self, image_data: Union[bytes, str, Image.Image]) -> Dict[str, Any]:
        """
        预处理图像数据
        
        Args:
            image_data: 图像数据（bytes, base64字符串或PIL Image）
            
        Returns:
            Dict: 预处理结果
        """
        try:
            # 转换为PIL Image
            if isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                # 假设是base64编码
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ValueError(f"Unsupported image data type: {type(image_data)}")
            
            # 转换为RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 获取原始尺寸
            original_size = image.size
            
            # 调整尺寸
            image_resized = self._resize_image(image)
            
            # 转换为numpy数组
            image_array = np.array(image_resized)
            
            # 转换为tensor
            image_tensor = self.normalize_transform(image_resized)
            
            return {
                'original_image': image,
                'processed_image': image_resized,
                'image_array': image_array,
                'image_tensor': image_tensor,
                'original_size': original_size,
                'processed_size': image_resized.size,
                'channels': len(image.getbands()),
                'format': image.format
            }
            
        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return {}
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """调整图像尺寸"""
        width, height = image.size
        
        # 如果图像太大，按比例缩小
        if width > self.max_size[0] or height > self.max_size[1]:
            ratio = min(self.max_size[0] / width, self.max_size[1] / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 如果图像太小，按比例放大
        elif width < self.min_size[0] and height < self.min_size[1]:
            ratio = max(self.min_size[0] / width, self.min_size[1] / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image

class ImageTypeClassifier:
    """图像类型分类器"""
    
    def __init__(self):
        # 图像类型特征
        self.type_features = {
            ImageType.SCREENSHOT: {
                'aspect_ratios': [(16, 9), (4, 3), (16, 10)],
                'typical_elements': ['ui', 'button', 'menu', 'window'],
                'color_patterns': 'high_contrast'
            },
            ImageType.DOCUMENT: {
                'aspect_ratios': [(8.5, 11), (1, 1.414)],  # A4等
                'typical_elements': ['text', 'paragraph', 'table'],
                'color_patterns': 'mostly_white_background'
            },
            ImageType.CHART: {
                'typical_elements': ['axis', 'legend', 'grid', 'data_points'],
                'color_patterns': 'structured_colors'
            },
            ImageType.PHOTO: {
                'typical_elements': ['natural_objects', 'people', 'landscape'],
                'color_patterns': 'natural_colors'
            }
        }
    
    async def classify_image_type(self, processed_image: Dict[str, Any]) -> Tuple[ImageType, float]:
        """
        分类图像类型
        
        Args:
            processed_image: 预处理后的图像数据
            
        Returns:
            Tuple[ImageType, float]: (图像类型, 置信度)
        """
        try:
            image_array = processed_image['image_array']
            original_size = processed_image['original_size']
            
            # 计算特征
            features = await self._extract_type_features(image_array, original_size)
            
            # 分类逻辑
            type_scores = {}
            
            # 检查宽高比
            aspect_ratio = original_size[0] / original_size[1]
            
            # 屏幕截图检测
            if 1.3 < aspect_ratio < 2.0:  # 常见屏幕比例
                type_scores[ImageType.SCREENSHOT] = 0.6
            
            # 文档检测
            if 0.7 < aspect_ratio < 0.8 or aspect_ratio > 2.5:  # 文档比例
                type_scores[ImageType.DOCUMENT] = 0.5
            
            # 基于颜色分布判断
            color_variance = features.get('color_variance', 0)
            if color_variance < 0.1:  # 低方差，可能是文档或图表
                type_scores[ImageType.DOCUMENT] = type_scores.get(ImageType.DOCUMENT, 0) + 0.3
                type_scores[ImageType.CHART] = type_scores.get(ImageType.CHART, 0) + 0.2
            
            # 基于边缘密度判断
            edge_density = features.get('edge_density', 0)
            if edge_density > 0.3:  # 高边缘密度，可能是截图或图表
                type_scores[ImageType.SCREENSHOT] = type_scores.get(ImageType.SCREENSHOT, 0) + 0.2
                type_scores[ImageType.CHART] = type_scores.get(ImageType.CHART, 0) + 0.3
            
            # 默认为照片
            if not type_scores:
                type_scores[ImageType.PHOTO] = 0.5
            
            # 选择得分最高的类型
            best_type = max(type_scores.items(), key=lambda x: x[1])
            return best_type[0], min(best_type[1], 1.0)
            
        except Exception as e:
            logger.error(f"Image type classification error: {e}")
            return ImageType.UNKNOWN, 0.0
    
    async def _extract_type_features(self, image_array: np.ndarray, original_size: Tuple[int, int]) -> Dict[str, float]:
        """提取图像类型特征"""
        try:
            # 颜色方差
            color_variance = np.var(image_array) / 255.0
            
            # 边缘检测
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # 亮度分布
            brightness = np.mean(gray) / 255.0
            
            # 对比度
            contrast = np.std(gray) / 255.0
            
            return {
                'color_variance': color_variance,
                'edge_density': edge_density,
                'brightness': brightness,
                'contrast': contrast,
                'aspect_ratio': original_size[0] / original_size[1]
            }
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}

class ObjectDetector:
    """物体检测器"""
    
    def __init__(self):
        # 预定义的物体类别
        self.object_classes = [
            'person', 'car', 'chair', 'table', 'computer', 'phone', 'book',
            'cup', 'bottle', 'dog', 'cat', 'tree', 'building', 'food',
            'clothing', 'furniture', 'electronics', 'animal', 'plant'
        ]
        
        # 模拟检测器权重
        self.detection_weights = {cls: np.random.random() for cls in self.object_classes}
    
    async def detect_objects(self, processed_image: Dict[str, Any]) -> List[DetectedObject]:
        """
        检测图像中的物体
        
        Args:
            processed_image: 预处理后的图像数据
            
        Returns:
            List[DetectedObject]: 检测到的物体列表
        """
        try:
            image_array = processed_image['image_array']
            height, width = image_array.shape[:2]
            
            # 模拟物体检测（实际应使用YOLO、RCNN等模型）
            detected_objects = []
            
            # 基于图像特征模拟检测
            features = await self._extract_detection_features(image_array)
            
            for obj_class in self.object_classes:
                # 模拟检测概率
                detection_prob = self._calculate_detection_probability(obj_class, features)
                
                if detection_prob > 0.3:  # 检测阈值
                    # 生成随机边界框
                    x1 = np.random.randint(0, width // 2)
                    y1 = np.random.randint(0, height // 2)
                    x2 = np.random.randint(x1 + 50, width)
                    y2 = np.random.randint(y1 + 50, height)
                    
                    detected_objects.append(DetectedObject(
                        label=obj_class,
                        confidence=detection_prob,
                        bbox=(x1, y1, x2, y2),
                        attributes={
                            'size': 'medium',
                            'color': self._estimate_object_color(image_array, (x1, y1, x2, y2))
                        }
                    ))
            
            # 按置信度排序
            detected_objects.sort(key=lambda x: x.confidence, reverse=True)
            
            # 返回前10个最可信的检测结果
            return detected_objects[:10]
            
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []
    
    async def _extract_detection_features(self, image_array: np.ndarray) -> Dict[str, float]:
        """提取用于物体检测的特征"""
        try:
            # 颜色直方图
            hist_r = cv2.calcHist([image_array], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([image_array], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([image_array], [2], None, [256], [0, 256])
            
            # 纹理特征
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            texture_variance = np.var(gray)
            
            # 形状特征
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            return {
                'color_complexity': np.std(hist_r) + np.std(hist_g) + np.std(hist_b),
                'texture_variance': texture_variance,
                'contour_count': len(contours),
                'avg_brightness': np.mean(gray),
                'edge_density': np.sum(edges > 0) / edges.size
            }
            
        except Exception as e:
            logger.error(f"Detection feature extraction error: {e}")
            return {}
    
    def _calculate_detection_probability(self, obj_class: str, features: Dict[str, float]) -> float:
        """计算物体检测概率"""
        base_prob = self.detection_weights.get(obj_class, 0.1)
        
        # 基于特征调整概率
        color_complexity = features.get('color_complexity', 0)
        texture_variance = features.get('texture_variance', 0)
        
        # 简单的概率调整逻辑
        if obj_class in ['person', 'car', 'building']:
            if color_complexity > 1000:
                base_prob *= 1.2
        elif obj_class in ['book', 'computer', 'phone']:
            if texture_variance < 500:
                base_prob *= 1.3
        
        return min(base_prob, 1.0)
    
    def _estimate_object_color(self, image_array: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """估算物体颜色"""
        x1, y1, x2, y2 = bbox
        roi = image_array[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 'unknown'
        
        # 计算平均颜色
        avg_color = np.mean(roi, axis=(0, 1))
        r, g, b = avg_color
        
        # 简单的颜色分类
        if r > g and r > b:
            return 'red'
        elif g > r and g > b:
            return 'green'
        elif b > r and b > g:
            return 'blue'
        elif r > 200 and g > 200 and b > 200:
            return 'white'
        elif r < 50 and g < 50 and b < 50:
            return 'black'
        else:
            return 'mixed'

class TextExtractor:
    """文本提取器（OCR）"""
    
    def __init__(self):
        self.supported_languages = ['zh', 'en', 'ja', 'ko']
    
    async def extract_text(self, processed_image: Dict[str, Any]) -> List[ExtractedText]:
        """
        从图像中提取文本
        
        Args:
            processed_image: 预处理后的图像数据
            
        Returns:
            List[ExtractedText]: 提取的文本列表
        """
        try:
            image_array = processed_image['image_array']
            
            # 模拟OCR处理（实际应使用PaddleOCR、Tesseract等）
            extracted_texts = []
            
            # 预处理用于OCR
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            # 查找文本区域
            text_regions = await self._find_text_regions(binary)
            
            for region in text_regions:
                # 模拟文本识别
                text_content = await self._recognize_text_in_region(gray, region)
                
                if text_content.strip():
                    extracted_texts.append(ExtractedText(
                        text=text_content,
                        confidence=0.8 + np.random.random() * 0.2,
                        bbox=region,
                        language=self._detect_language(text_content)
                    ))
            
            return extracted_texts
            
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return []
    
    async def _find_text_regions(self, binary_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """查找文本区域"""
        try:
            # 形态学操作连接文本
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary_image, kernel, iterations=2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # 过滤太小的区域
                if w > 20 and h > 10:
                    text_regions.append((x, y, x + w, y + h))
            
            return text_regions
            
        except Exception as e:
            logger.error(f"Text region detection error: {e}")
            return []
    
    async def _recognize_text_in_region(self, gray_image: np.ndarray, region: Tuple[int, int, int, int]) -> str:
        """识别区域内的文本"""
        # 模拟文本识别
        x1, y1, x2, y2 = region
        roi = gray_image[y1:y2, x1:x2]
        
        # 基于区域特征生成模拟文本
        region_hash = hash(roi.tobytes()) % 1000
        
        sample_texts = [
            "这是一段示例文本",
            "Hello World",
            "人工智能技术",
            "Machine Learning",
            "深度学习算法",
            "Computer Vision",
            "自然语言处理",
            "Data Science"
        ]
        
        return sample_texts[region_hash % len(sample_texts)]
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        # 简单的语言检测
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)
        
        if chinese_chars > english_chars:
            return 'zh'
        else:
            return 'en'

class SceneAnalyzer:
    """场景分析器"""
    
    def __init__(self):
        self.scene_categories = [
            'indoor', 'outdoor', 'office', 'home', 'street', 'nature',
            'restaurant', 'shop', 'vehicle', 'sports', 'event', 'art'
        ]
    
    async def analyze_scene(self, processed_image: Dict[str, Any], objects: List[DetectedObject]) -> Dict[str, Any]:
        """
        分析图像场景
        
        Args:
            processed_image: 预处理后的图像数据
            objects: 检测到的物体列表
            
        Returns:
            Dict: 场景分析结果
        """
        try:
            image_array = processed_image['image_array']
            
            # 基于物体推断场景
            scene_scores = {}
            
            for obj in objects:
                if obj.label in ['car', 'road', 'traffic_light']:
                    scene_scores['street'] = scene_scores.get('street', 0) + obj.confidence
                elif obj.label in ['tree', 'grass', 'mountain', 'sky']:
                    scene_scores['nature'] = scene_scores.get('nature', 0) + obj.confidence
                elif obj.label in ['desk', 'computer', 'chair']:
                    scene_scores['office'] = scene_scores.get('office', 0) + obj.confidence
                elif obj.label in ['bed', 'sofa', 'kitchen']:
                    scene_scores['home'] = scene_scores.get('home', 0) + obj.confidence
            
            # 基于颜色分析场景
            color_analysis = await self._analyze_color_distribution(image_array)
            
            # 基于光照分析场景
            lighting_analysis = await self._analyze_lighting(image_array)
            
            # 确定主要场景
            if scene_scores:
                primary_scene = max(scene_scores.items(), key=lambda x: x[1])
            else:
                primary_scene = ('unknown', 0.0)
            
            return {
                'primary_scene': primary_scene[0],
                'scene_confidence': primary_scene[1],
                'scene_scores': scene_scores,
                'color_analysis': color_analysis,
                'lighting_analysis': lighting_analysis,
                'indoor_outdoor': self._classify_indoor_outdoor(scene_scores, lighting_analysis),
                'time_of_day': lighting_analysis.get('estimated_time', 'unknown'),
                'weather': self._estimate_weather(color_analysis, lighting_analysis)
            }
            
        except Exception as e:
            logger.error(f"Scene analysis error: {e}")
            return {}
    
    async def _analyze_color_distribution(self, image_array: np.ndarray) -> Dict[str, Any]:
        """分析颜色分布"""
        try:
            # 计算主要颜色
            pixels = image_array.reshape(-1, 3)
            
            # K-means聚类找主要颜色
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            dominant_colors = kmeans.cluster_centers_
            color_percentages = np.bincount(kmeans.labels_) / len(kmeans.labels_)
            
            # 分析颜色温度
            avg_color = np.mean(pixels, axis=0)
            color_temperature = self._estimate_color_temperature(avg_color)
            
            return {
                'dominant_colors': dominant_colors.tolist(),
                'color_percentages': color_percentages.tolist(),
                'average_color': avg_color.tolist(),
                'color_temperature': color_temperature,
                'color_diversity': np.std(pixels, axis=0).tolist()
            }
            
        except Exception as e:
            logger.error(f"Color analysis error: {e}")
            return {}
    
    async def _analyze_lighting(self, image_array: np.ndarray) -> Dict[str, Any]:
        """分析光照条件"""
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # 亮度统计
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # 估算时间
            if brightness > 180:
                estimated_time = 'day'
            elif brightness > 100:
                estimated_time = 'evening'
            else:
                estimated_time = 'night'
            
            # 光照方向分析
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'estimated_time': estimated_time,
                'lighting_quality': 'good' if contrast > 30 else 'poor',
                'shadow_presence': 'high' if np.std(gradient_x) > 50 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Lighting analysis error: {e}")
            return {}
    
    def _estimate_color_temperature(self, avg_color: np.ndarray) -> str:
        """估算色温"""
        r, g, b = avg_color
        
        if b > r and b > g:
            return 'cool'  # 冷色调
        elif r > g and r > b:
            return 'warm'  # 暖色调
        else:
            return 'neutral'  # 中性
    
    def _classify_indoor_outdoor(self, scene_scores: Dict[str, float], lighting_analysis: Dict[str, Any]) -> str:
        """分类室内外"""
        outdoor_indicators = scene_scores.get('street', 0) + scene_scores.get('nature', 0)
        indoor_indicators = scene_scores.get('office', 0) + scene_scores.get('home', 0)
        
        if outdoor_indicators > indoor_indicators:
            return 'outdoor'
        elif indoor_indicators > outdoor_indicators:
            return 'indoor'
        else:
            # 基于光照判断
            brightness = lighting_analysis.get('brightness', 0)
            return 'outdoor' if brightness > 150 else 'indoor'
    
    def _estimate_weather(self, color_analysis: Dict[str, Any], lighting_analysis: Dict[str, Any]) -> str:
        """估算天气"""
        brightness = lighting_analysis.get('brightness', 0)
        color_temp = color_analysis.get('color_temperature', 'neutral')
        
        if brightness > 180 and color_temp == 'warm':
            return 'sunny'
        elif brightness < 100:
            return 'cloudy'
        elif color_temp == 'cool':
            return 'overcast'
        else:
            return 'unknown'

class VisionUnderstanding:
    """视觉理解主类"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.type_classifier = ImageTypeClassifier()
        self.object_detector = ObjectDetector()
        self.text_extractor = TextExtractor()
        self.scene_analyzer = SceneAnalyzer()
    
    async def understand_image(self, 
                             image_data: Union[bytes, str, Image.Image], 
                             tasks: Optional[List[VisionTask]] = None,
                             query: Optional[str] = None) -> VisionAnalysisResult:
        """
        综合图像理解和分析
        
        Args:
            image_data: 图像数据
            tasks: 要执行的视觉任务列表
            query: 用户查询（可选）
            
        Returns:
            VisionAnalysisResult: 分析结果
        """
        start_time = time.time()
        
        try:
            # 默认任务
            if tasks is None:
                tasks = [VisionTask.DESCRIPTION, VisionTask.OBJECT_DETECTION, VisionTask.SCENE_ANALYSIS]
            
            # 预处理图像
            processed_image = await self.preprocessor.preprocess_image(image_data)
            if not processed_image:
                raise ValueError("Image preprocessing failed")
            
            # 分类图像类型
            image_type, type_confidence = await self.type_classifier.classify_image_type(processed_image)
            
            # 并行执行各种分析任务
            analysis_tasks = []
            
            if VisionTask.OBJECT_DETECTION in tasks:
                analysis_tasks.append(self.object_detector.detect_objects(processed_image))
            else:
                analysis_tasks.append(asyncio.create_task(self._empty_list()))
            
            if VisionTask.TEXT_EXTRACTION in tasks:
                analysis_tasks.append(self.text_extractor.extract_text(processed_image))
            else:
                analysis_tasks.append(asyncio.create_task(self._empty_list()))
            
            # 等待分析完成
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            objects = results[0] if not isinstance(results[0], Exception) else []
            extracted_texts = results[1] if not isinstance(results[1], Exception) else []
            
            # 场景分析
            scene_attributes = {}
            if VisionTask.SCENE_ANALYSIS in tasks:
                scene_attributes = await self.scene_analyzer.analyze_scene(processed_image, objects)
            
            # 生成描述
            description = await self._generate_description(
                image_type, objects, extracted_texts, scene_attributes, query
            )
            
            processing_time = time.time() - start_time
            
            # 计算整体置信度
            overall_confidence = self._calculate_overall_confidence(
                type_confidence, objects, extracted_texts, scene_attributes
            )
            
            return VisionAnalysisResult(
                image_type=image_type,
                description=description,
                objects=objects,
                extracted_texts=extracted_texts,
                scene_attributes=scene_attributes,
                processing_time=processing_time,
                confidence=overall_confidence
            )
            
        except Exception as e:
            logger.error(f"Vision understanding error: {e}")
            return VisionAnalysisResult(
                image_type=ImageType.UNKNOWN,
                description=f"图像分析失败: {str(e)}",
                processing_time=time.time() - start_time,
                confidence=0.0
            )
    
    async def _empty_list(self) -> List:
        """返回空列表的异步函数"""
        return []
    
    async def _generate_description(self, 
                                 image_type: ImageType,
                                 objects: List[DetectedObject],
                                 texts: List[ExtractedText],
                                 scene_attrs: Dict[str, Any],
                                 query: Optional[str] = None) -> str:
        """生成图像描述"""
        try:
            description_parts = []
            
            # 基础描述
            description_parts.append(f"这是一张{image_type.value}类型的图像。")
            
            # 场景描述
            if scene_attrs:
                primary_scene = scene_attrs.get('primary_scene', 'unknown')
                indoor_outdoor = scene_attrs.get('indoor_outdoor', 'unknown')
                time_of_day = scene_attrs.get('time_of_day', 'unknown')
                
                if primary_scene != 'unknown':
                    description_parts.append(f"场景主要是{primary_scene}")
                    
                if indoor_outdoor != 'unknown':
                    description_parts.append(f"这是一个{indoor_outdoor}环境")
                    
                if time_of_day != 'unknown':
                    description_parts.append(f"拍摄时间可能是{time_of_day}")
            
            # 物体描述
            if objects:
                top_objects = objects[:5]  # 前5个最可信的物体
                object_names = [obj.label for obj in top_objects]
                description_parts.append(f"图像中包含：{', '.join(object_names)}")
            
            # 文本描述
            if texts:
                text_contents = [text.text for text in texts[:3]]  # 前3个文本
                description_parts.append(f"图像中的文字内容包括：{', '.join(text_contents)}")
            
            # 针对特定查询的回答
            if query:
                query_response = await self._answer_specific_query(query, objects, texts, scene_attrs)
                if query_response:
                    description_parts.append(f"针对您的问题：{query_response}")
            
            return "。".join(description_parts) + "。"
            
        except Exception as e:
            logger.error(f"Description generation error: {e}")
            return "无法生成图像描述。"
    
    async def _answer_specific_query(self, 
                                   query: str,
                                   objects: List[DetectedObject],
                                   texts: List[ExtractedText],
                                   scene_attrs: Dict[str, Any]) -> str:
        """回答特定查询"""
        query_lower = query.lower()
        
        # 物体相关查询
        if any(word in query_lower for word in ['有什么', '看到什么', '包含什么']):
            if objects:
                return f"我看到了{', '.join([obj.label for obj in objects[:5]])}"
        
        # 文字相关查询
        if any(word in query_lower for word in ['写了什么', '文字', '内容']):
            if texts:
                return f"文字内容是：{', '.join([text.text for text in texts[:3]])}"
        
        # 场景相关查询
        if any(word in query_lower for word in ['哪里', '什么地方', '场景']):
            primary_scene = scene_attrs.get('primary_scene', 'unknown')
            if primary_scene != 'unknown':
                return f"这个场景看起来像是{primary_scene}"
        
        # 时间相关查询
        if any(word in query_lower for word in ['什么时候', '时间', '白天', '晚上']):
            time_of_day = scene_attrs.get('time_of_day', 'unknown')
            if time_of_day != 'unknown':
                return f"看起来是{time_of_day}拍摄的"
        
        return ""
    
    def _calculate_overall_confidence(self, 
                                    type_confidence: float,
                                    objects: List[DetectedObject],
                                    texts: List[ExtractedText],
                                    scene_attrs: Dict[str, Any]) -> float:
        """计算整体置信度"""
        confidences = [type_confidence]
        
        if objects:
            avg_object_confidence = sum(obj.confidence for obj in objects) / len(objects)
            confidences.append(avg_object_confidence)
        
        if texts:
            avg_text_confidence = sum(text.confidence for text in texts) / len(texts)
            confidences.append(avg_text_confidence)
        
        if scene_attrs and 'scene_confidence' in scene_attrs:
            confidences.append(scene_attrs['scene_confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0

# 使用示例
async def main():
    """示例用法"""
    vision = VisionUnderstanding()
    
    # 创建模拟图像
    mock_image = Image.new('RGB', (800, 600), color='lightblue')
    
    # 分析图像
    result = await vision.understand_image(
        mock_image,
        tasks=[VisionTask.DESCRIPTION, VisionTask.OBJECT_DETECTION, VisionTask.SCENE_ANALYSIS],
        query="这张图片里有什么？"
    )
    
    print("=== 视觉理解结果 ===")
    print(f"图像类型: {result.image_type.value}")
    print(f"描述: {result.description}")
    print(f"检测到的物体数量: {len(result.objects)}")
    print(f"提取的文本数量: {len(result.extracted_texts)}")
    print(f"处理时间: {result.processing_time:.2f}秒")
    print(f"整体置信度: {result.confidence:.2f}")
    
    if result.objects:
        print("\n检测到的物体:")
        for i, obj in enumerate(result.objects[:5]):
            print(f"  {i+1}. {obj.label} (置信度: {obj.confidence:.2f})")
    
    if result.scene_attributes:
        print(f"\n场景分析:")
        print(f"  主要场景: {result.scene_attributes.get('primary_scene', 'unknown')}")
        print(f"  室内外: {result.scene_attributes.get('indoor_outdoor', 'unknown')}")

if __name__ == "__main__":
    asyncio.run(main())
