"""
VoiceHelper v1.23.0 - 移动端优化系统
实现移动端性能优化、触摸交互、离线功能
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import uuid

logger = logging.getLogger(__name__)

class DeviceType(Enum):
    """设备类型"""
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"
    WEARABLE = "wearable"

class NetworkType(Enum):
    """网络类型"""
    WIFI = "wifi"
    MOBILE_4G = "4g"
    MOBILE_5G = "5g"
    MOBILE_3G = "3g"
    OFFLINE = "offline"

class TouchGesture(Enum):
    """触摸手势"""
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"
    SWIPE_UP = "swipe_up"
    SWIPE_DOWN = "swipe_down"
    PINCH_ZOOM = "pinch_zoom"
    ROTATE = "rotate"

class PerformanceLevel(Enum):
    """性能级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"

@dataclass
class DeviceInfo:
    """设备信息"""
    device_id: str
    device_type: DeviceType
    screen_width: int
    screen_height: int
    pixel_ratio: float
    os_version: str
    browser_version: str
    memory_total: int
    memory_available: int
    cpu_cores: int
    network_type: NetworkType
    battery_level: float
    is_charging: bool

@dataclass
class TouchEvent:
    """触摸事件"""
    event_id: str
    gesture: TouchGesture
    coordinates: Tuple[int, int]
    timestamp: float
    pressure: float = 0.0
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class OptimizationConfig:
    """优化配置"""
    performance_level: PerformanceLevel
    enable_offline_mode: bool
    enable_gesture_controls: bool
    enable_voice_commands: bool
    enable_haptic_feedback: bool
    cache_size_mb: int
    max_concurrent_requests: int
    timeout_seconds: int

class MobilePerformanceOptimizer:
    """移动端性能优化器"""
    
    def __init__(self):
        self.device_profiles = {}
        self.performance_metrics = defaultdict(list)
        self.optimization_rules = {}
        
        # 初始化优化规则
        self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self):
        """初始化优化规则"""
        self.optimization_rules = {
            DeviceType.MOBILE: {
                "max_image_size": 1024,
                "max_video_size": 2048,
                "cache_duration": 3600,
                "compression_level": 0.8,
                "lazy_loading": True
            },
            DeviceType.TABLET: {
                "max_image_size": 2048,
                "max_video_size": 4096,
                "cache_duration": 7200,
                "compression_level": 0.9,
                "lazy_loading": True
            },
            DeviceType.WEARABLE: {
                "max_image_size": 512,
                "max_video_size": 1024,
                "cache_duration": 1800,
                "compression_level": 0.7,
                "lazy_loading": False
            }
        }
    
    def register_device(self, device_info: DeviceInfo) -> OptimizationConfig:
        """注册设备"""
        self.device_profiles[device_info.device_id] = device_info
        
        # 根据设备类型生成优化配置
        config = self._generate_optimization_config(device_info)
        
        logger.info(f"Registered device: {device_info.device_id} ({device_info.device_type.value})")
        return config
    
    def _generate_optimization_config(self, device_info: DeviceInfo) -> OptimizationConfig:
        """生成优化配置"""
        # 根据设备性能确定性能级别
        performance_score = self._calculate_performance_score(device_info)
        
        if performance_score >= 0.8:
            performance_level = PerformanceLevel.ULTRA
        elif performance_score >= 0.6:
            performance_level = PerformanceLevel.HIGH
        elif performance_score >= 0.4:
            performance_level = PerformanceLevel.MEDIUM
        else:
            performance_level = PerformanceLevel.LOW
        
        # 根据网络类型调整配置
        enable_offline = device_info.network_type == NetworkType.OFFLINE
        cache_size = self._calculate_cache_size(device_info)
        max_requests = self._calculate_max_requests(device_info)
        
        return OptimizationConfig(
            performance_level=performance_level,
            enable_offline_mode=enable_offline,
            enable_gesture_controls=True,
            enable_voice_commands=True,
            enable_haptic_feedback=device_info.device_type in [DeviceType.MOBILE, DeviceType.TABLET],
            cache_size_mb=cache_size,
            max_concurrent_requests=max_requests,
            timeout_seconds=30 if device_info.network_type == NetworkType.WIFI else 60
        )
    
    def _calculate_performance_score(self, device_info: DeviceInfo) -> float:
        """计算性能分数"""
        score = 0.0
        
        # 内存分数
        memory_score = min(device_info.memory_available / (1024 * 1024 * 1024), 1.0)  # GB
        score += memory_score * 0.3
        
        # CPU分数
        cpu_score = min(device_info.cpu_cores / 8, 1.0)
        score += cpu_score * 0.2
        
        # 屏幕分数
        screen_score = min((device_info.screen_width * device_info.screen_height) / (1920 * 1080), 1.0)
        score += screen_score * 0.2
        
        # 网络分数
        network_scores = {
            NetworkType.WIFI: 1.0,
            NetworkType.MOBILE_5G: 0.9,
            NetworkType.MOBILE_4G: 0.7,
            NetworkType.MOBILE_3G: 0.4,
            NetworkType.OFFLINE: 0.1
        }
        score += network_scores.get(device_info.network_type, 0.5) * 0.3
        
        return min(score, 1.0)
    
    def _calculate_cache_size(self, device_info: DeviceInfo) -> int:
        """计算缓存大小"""
        base_size = 50  # MB
        
        if device_info.device_type == DeviceType.MOBILE:
            return base_size
        elif device_info.device_type == DeviceType.TABLET:
            return base_size * 2
        elif device_info.device_type == DeviceType.WEARABLE:
            return base_size // 2
        else:
            return base_size
    
    def _calculate_max_requests(self, device_info: DeviceInfo) -> int:
        """计算最大并发请求数"""
        if device_info.network_type == NetworkType.WIFI:
            return 10
        elif device_info.network_type in [NetworkType.MOBILE_5G, NetworkType.MOBILE_4G]:
            return 5
        else:
            return 2
    
    async def optimize_content(self, content: Dict[str, Any], device_id: str) -> Dict[str, Any]:
        """优化内容"""
        if device_id not in self.device_profiles:
            return content
        
        device_info = self.device_profiles[device_id]
        rules = self.optimization_rules.get(device_info.device_type, {})
        
        optimized_content = content.copy()
        
        # 图片优化
        if "images" in content:
            optimized_content["images"] = await self._optimize_images(
                content["images"], rules
            )
        
        # 视频优化
        if "videos" in content:
            optimized_content["videos"] = await self._optimize_videos(
                content["videos"], rules
            )
        
        # 文本优化
        if "text" in content:
            optimized_content["text"] = await self._optimize_text(
                content["text"], device_info
            )
        
        return optimized_content
    
    async def _optimize_images(self, images: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化图片"""
        max_size = rules.get("max_image_size", 1024)
        compression_level = rules.get("compression_level", 0.8)
        
        optimized_images = []
        for image in images:
            optimized_image = image.copy()
            optimized_image["max_width"] = max_size
            optimized_image["compression"] = compression_level
            optimized_image["lazy_loading"] = rules.get("lazy_loading", True)
            optimized_images.append(optimized_image)
        
        return optimized_images
    
    async def _optimize_videos(self, videos: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """优化视频"""
        max_size = rules.get("max_video_size", 2048)
        
        optimized_videos = []
        for video in videos:
            optimized_video = video.copy()
            optimized_video["max_width"] = max_size
            optimized_video["auto_play"] = False  # 移动端禁用自动播放
            optimized_videos.append(optimized_video)
        
        return optimized_videos
    
    async def _optimize_text(self, text: str, device_info: DeviceInfo) -> str:
        """优化文本"""
        # 根据屏幕大小调整文本长度
        max_length = device_info.screen_width // 10  # 简化的文本长度计算
        
        if len(text) > max_length:
            return text[:max_length] + "..."
        
        return text

class TouchGestureRecognizer:
    """触摸手势识别器"""
    
    def __init__(self):
        self.gesture_patterns = {}
        self.touch_events = defaultdict(list)
        self.gesture_callbacks = {}
        
        # 初始化手势模式
        self._initialize_gesture_patterns()
    
    def _initialize_gesture_patterns(self):
        """初始化手势模式"""
        self.gesture_patterns = {
            TouchGesture.TAP: {
                "max_duration": 0.3,
                "max_movement": 10,
                "min_pressure": 0.1
            },
            TouchGesture.DOUBLE_TAP: {
                "max_interval": 0.5,
                "max_duration": 0.3,
                "max_movement": 10
            },
            TouchGesture.LONG_PRESS: {
                "min_duration": 0.5,
                "max_movement": 20
            },
            TouchGesture.SWIPE_LEFT: {
                "min_distance": 50,
                "max_duration": 0.5,
                "direction": "left"
            },
            TouchGesture.SWIPE_RIGHT: {
                "min_distance": 50,
                "max_duration": 0.5,
                "direction": "right"
            }
        }
    
    def register_gesture_callback(self, gesture: TouchGesture, callback: callable):
        """注册手势回调"""
        self.gesture_callbacks[gesture] = callback
        logger.info(f"Registered callback for gesture: {gesture.value}")
    
    async def process_touch_event(self, device_id: str, touch_event: TouchEvent) -> Optional[str]:
        """处理触摸事件"""
        try:
            # 记录触摸事件
            self.touch_events[device_id].append(touch_event)
            
            # 识别手势
            recognized_gesture = await self._recognize_gesture(device_id, touch_event)
            
            if recognized_gesture:
                # 执行手势回调
                if recognized_gesture in self.gesture_callbacks:
                    callback = self.gesture_callbacks[recognized_gesture]
                    result = await callback(touch_event)
                    logger.info(f"Gesture recognized: {recognized_gesture.value}")
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Touch event processing error: {e}")
            return None
    
    async def _recognize_gesture(self, device_id: str, touch_event: TouchEvent) -> Optional[TouchGesture]:
        """识别手势"""
        events = self.touch_events[device_id]
        
        # 检查点击
        if await self._is_tap(events):
            return TouchGesture.TAP
        
        # 检查双击
        if await self._is_double_tap(events):
            return TouchGesture.DOUBLE_TAP
        
        # 检查长按
        if await self._is_long_press(events):
            return TouchGesture.LONG_PRESS
        
        # 检查滑动手势
        swipe_gesture = await self._detect_swipe(events)
        if swipe_gesture:
            return swipe_gesture
        
        return None
    
    async def _is_tap(self, events: List[TouchEvent]) -> bool:
        """检查是否为点击"""
        if not events:
            return False
        
        latest_event = events[-1]
        pattern = self.gesture_patterns[TouchGesture.TAP]
        
        return (latest_event.duration <= pattern["max_duration"] and
                latest_event.pressure >= pattern["min_pressure"])
    
    async def _is_double_tap(self, events: List[TouchEvent]) -> bool:
        """检查是否为双击"""
        if len(events) < 2:
            return False
        
        pattern = self.gesture_patterns[TouchGesture.DOUBLE_TAP]
        latest_event = events[-1]
        previous_event = events[-2]
        
        time_interval = latest_event.timestamp - previous_event.timestamp
        
        return (time_interval <= pattern["max_interval"] and
                latest_event.duration <= pattern["max_duration"])
    
    async def _is_long_press(self, events: List[TouchEvent]) -> bool:
        """检查是否为长按"""
        if not events:
            return False
        
        latest_event = events[-1]
        pattern = self.gesture_patterns[TouchGesture.LONG_PRESS]
        
        return latest_event.duration >= pattern["min_duration"]
    
    async def _detect_swipe(self, events: List[TouchEvent]) -> Optional[TouchGesture]:
        """检测滑动手势"""
        if len(events) < 2:
            return None
        
        start_event = events[0]
        end_event = events[-1]
        
        dx = end_event.coordinates[0] - start_event.coordinates[0]
        dy = end_event.coordinates[1] - start_event.coordinates[1]
        distance = (dx**2 + dy**2)**0.5
        
        duration = end_event.timestamp - start_event.timestamp
        
        # 检查左滑
        if dx < -50 and abs(dy) < 50 and distance > 50 and duration < 0.5:
            return TouchGesture.SWIPE_LEFT
        
        # 检查右滑
        if dx > 50 and abs(dy) < 50 and distance > 50 and duration < 0.5:
            return TouchGesture.SWIPE_RIGHT
        
        return None

class OfflineCapabilityManager:
    """离线功能管理器"""
    
    def __init__(self):
        self.offline_data = {}
        self.sync_queue = []
        self.offline_capabilities = {
            "voice_recognition": True,
            "text_processing": True,
            "basic_ai": True,
            "data_storage": True,
            "caching": True
        }
    
    async def enable_offline_mode(self, device_id: str) -> Dict[str, Any]:
        """启用离线模式"""
        offline_config = {
            "enabled": True,
            "capabilities": self.offline_capabilities,
            "cache_size": 100,  # MB
            "sync_interval": 300,  # 5分钟
            "offline_data": self.offline_data.get(device_id, {})
        }
        
        logger.info(f"Enabled offline mode for device: {device_id}")
        return offline_config
    
    async def sync_offline_data(self, device_id: str, data: Dict[str, Any]) -> bool:
        """同步离线数据"""
        try:
            self.offline_data[device_id] = data
            self.sync_queue.append({
                "device_id": device_id,
                "data": data,
                "timestamp": time.time()
            })
            
            logger.info(f"Synced offline data for device: {device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Offline data sync error: {e}")
            return False
    
    async def get_offline_data(self, device_id: str) -> Dict[str, Any]:
        """获取离线数据"""
        return self.offline_data.get(device_id, {})
    
    def get_offline_capabilities(self) -> Dict[str, bool]:
        """获取离线功能"""
        return self.offline_capabilities

class MobileOptimizationSystem:
    """移动端优化系统"""
    
    def __init__(self):
        self.performance_optimizer = MobilePerformanceOptimizer()
        self.gesture_recognizer = TouchGestureRecognizer()
        self.offline_manager = OfflineCapabilityManager()
        self.active_devices = {}
        
    async def register_mobile_device(self, device_info: DeviceInfo) -> Dict[str, Any]:
        """注册移动设备"""
        # 生成优化配置
        config = self.performance_optimizer.register_device(device_info)
        
        # 注册手势回调
        await self._setup_gesture_callbacks()
        
        # 启用离线模式（如果需要）
        offline_config = None
        if device_info.network_type == NetworkType.OFFLINE:
            offline_config = await self.offline_manager.enable_offline_mode(device_info.device_id)
        
        # 保存设备信息
        self.active_devices[device_info.device_id] = {
            "device_info": device_info,
            "config": config,
            "offline_config": offline_config,
            "registered_at": time.time()
        }
        
        return {
            "device_id": device_info.device_id,
            "optimization_config": config.__dict__,
            "offline_config": offline_config,
            "gesture_controls": True,
            "performance_level": config.performance_level.value
        }
    
    async def _setup_gesture_callbacks(self):
        """设置手势回调"""
        # 注册各种手势的回调函数
        self.gesture_recognizer.register_gesture_callback(
            TouchGesture.TAP, self._handle_tap
        )
        self.gesture_recognizer.register_gesture_callback(
            TouchGesture.DOUBLE_TAP, self._handle_double_tap
        )
        self.gesture_recognizer.register_gesture_callback(
            TouchGesture.LONG_PRESS, self._handle_long_press
        )
        self.gesture_recognizer.register_gesture_callback(
            TouchGesture.SWIPE_LEFT, self._handle_swipe_left
        )
        self.gesture_recognizer.register_gesture_callback(
            TouchGesture.SWIPE_RIGHT, self._handle_swipe_right
        )
    
    async def _handle_tap(self, touch_event: TouchEvent) -> str:
        """处理点击"""
        return "tap_action"
    
    async def _handle_double_tap(self, touch_event: TouchEvent) -> str:
        """处理双击"""
        return "double_tap_action"
    
    async def _handle_long_press(self, touch_event: TouchEvent) -> str:
        """处理长按"""
        return "long_press_action"
    
    async def _handle_swipe_left(self, touch_event: TouchEvent) -> str:
        """处理左滑"""
        return "swipe_left_action"
    
    async def _handle_swipe_right(self, touch_event: TouchEvent) -> str:
        """处理右滑"""
        return "swipe_right_action"
    
    async def process_mobile_request(self, device_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """处理移动端请求"""
        if device_id not in self.active_devices:
            return {"error": "Device not registered"}
        
        device_info = self.active_devices[device_id]["device_info"]
        
        # 优化内容
        optimized_content = await self.performance_optimizer.optimize_content(
            request_data, device_id
        )
        
        # 处理触摸事件（如果有）
        if "touch_events" in request_data:
            for touch_event_data in request_data["touch_events"]:
                touch_event = TouchEvent(**touch_event_data)
                await self.gesture_recognizer.process_touch_event(device_id, touch_event)
        
        return {
            "optimized_content": optimized_content,
            "device_optimized": True,
            "performance_level": self.active_devices[device_id]["config"].performance_level.value
        }
    
    def get_mobile_stats(self) -> Dict[str, Any]:
        """获取移动端统计"""
        return {
            "active_devices": len(self.active_devices),
            "device_types": {
                device["device_info"].device_type.value: 1 
                for device in self.active_devices.values()
            },
            "offline_capabilities": self.offline_manager.get_offline_capabilities(),
            "gesture_callbacks": len(self.gesture_recognizer.gesture_callbacks),
            "offline_data_sync_queue": len(self.offline_manager.sync_queue)
        }

# 全局移动端优化系统实例
mobile_optimization_system = MobileOptimizationSystem()

async def register_mobile_device(device_info: DeviceInfo) -> Dict[str, Any]:
    """注册移动设备"""
    return await mobile_optimization_system.register_mobile_device(device_info)

async def process_mobile_request(device_id: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    """处理移动端请求"""
    return await mobile_optimization_system.process_mobile_request(device_id, request_data)

def get_mobile_stats() -> Dict[str, Any]:
    """获取移动端统计"""
    return mobile_optimization_system.get_mobile_stats()

if __name__ == "__main__":
    # 测试代码
    async def test_mobile_system():
        # 创建测试设备信息
        device_info = DeviceInfo(
            device_id="test_mobile_001",
            device_type=DeviceType.MOBILE,
            screen_width=375,
            screen_height=667,
            pixel_ratio=2.0,
            os_version="iOS 15.0",
            browser_version="Safari 15.0",
            memory_total=4096,
            memory_available=2048,
            cpu_cores=6,
            network_type=NetworkType.MOBILE_5G,
            battery_level=0.8,
            is_charging=False
        )
        
        # 注册设备
        result = await register_mobile_device(device_info)
        print("Device registration result:", result)
        
        # 处理请求
        request_data = {
            "content": "test content",
            "images": [{"url": "test.jpg", "size": 1024}],
            "touch_events": [{
                "event_id": "touch_001",
                "gesture": TouchGesture.TAP,
                "coordinates": (100, 200),
                "timestamp": time.time(),
                "pressure": 0.5,
                "duration": 0.2
            }]
        }
        
        response = await process_mobile_request("test_mobile_001", request_data)
        print("Mobile request response:", response)
        
        # 获取统计
        stats = get_mobile_stats()
        print("Mobile system stats:", stats)
    
    asyncio.run(test_mobile_system())
