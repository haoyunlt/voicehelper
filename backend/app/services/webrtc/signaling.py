"""
WebRTC信令服务 - 支持LiveKit/Daily/OpenAI Realtime
"""
import asyncio
import jwt
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import structlog

from app.config import settings

logger = structlog.get_logger()


class WebRTCSignalingService:
    """WebRTC信令服务基类"""
    
    def __init__(self):
        self.provider = self._detect_provider()
        logger.info(f"WebRTC signaling provider: {self.provider}")
    
    def _detect_provider(self) -> str:
        """检测可用的WebRTC提供商"""
        if settings.LIVEKIT_KEY and settings.LIVEKIT_SECRET:
            return "livekit"
        elif settings.OPENAI_API_KEY:
            return "openai_realtime"
        else:
            return "websocket_fallback"
    
    async def create_session_token(
        self,
        session_id: str,
        user_id: str,
        capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建会话令牌和连接信息"""
        
        if self.provider == "livekit":
            return await self._create_livekit_token(session_id, user_id, capabilities)
        elif self.provider == "openai_realtime":
            return await self._create_openai_realtime_token(session_id, user_id, capabilities)
        else:
            return await self._create_websocket_fallback(session_id, user_id, capabilities)
    
    async def _create_livekit_token(
        self,
        session_id: str,
        user_id: str,
        capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建LiveKit令牌"""
        try:
            from livekit import api
            
            # LiveKit房间名称
            room_name = f"voice_session_{session_id}"
            
            # 创建访问令牌
            token = api.AccessToken(
                api_key=settings.LIVEKIT_KEY,
                api_secret=settings.LIVEKIT_SECRET
            )
            
            # 设置身份和权限
            token = token.with_identity(user_id).with_name(f"User {user_id}")
            token = token.with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True
                )
            )
            
            # 设置过期时间
            expires_at = datetime.utcnow() + timedelta(hours=2)
            
            return {
                "token": token.to_jwt(),
                "expires_at": expires_at,
                "connection_info": {
                    "type": "webrtc",
                    "url": settings.LIVEKIT_URL,
                    "room_name": room_name,
                    "ice_servers": [
                        {"urls": ["stun:stun.l.google.com:19302"]},
                        {"urls": ["stun:stun1.l.google.com:19302"]}
                    ]
                }
            }
        
        except ImportError:
            logger.warning("LiveKit SDK not available, falling back to WebSocket")
            return await self._create_websocket_fallback(session_id, user_id, capabilities)
        except Exception as e:
            logger.error("Failed to create LiveKit token", error=str(e))
            return await self._create_websocket_fallback(session_id, user_id, capabilities)
    
    async def _create_openai_realtime_token(
        self,
        session_id: str,
        user_id: str,
        capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建OpenAI Realtime会话令牌"""
        try:
            # OpenAI Realtime使用API密钥直接连接
            expires_at = datetime.utcnow() + timedelta(hours=1)
            
            # 创建临时会话令牌
            session_token = jwt.encode(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "iat": int(time.time()),
                    "exp": int(expires_at.timestamp()),
                    "capabilities": capabilities
                },
                settings.OPENAI_API_KEY[:32],  # 使用API密钥的前32字符作为签名密钥
                algorithm="HS256"
            )
            
            return {
                "token": session_token,
                "expires_at": expires_at,
                "connection_info": {
                    "type": "webrtc",
                    "url": "wss://api.openai.com/v1/realtime",
                    "model": "gpt-4o-realtime-preview-2024-10-01",
                    "api_key": settings.OPENAI_API_KEY,
                    "ice_servers": [
                        {"urls": ["stun:stun.l.google.com:19302"]}
                    ]
                }
            }
        
        except Exception as e:
            logger.error("Failed to create OpenAI Realtime token", error=str(e))
            return await self._create_websocket_fallback(session_id, user_id, capabilities)
    
    async def _create_websocket_fallback(
        self,
        session_id: str,
        user_id: str,
        capabilities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """创建WebSocket回退连接"""
        expires_at = datetime.utcnow() + timedelta(hours=2)
        
        # 创建简单的会话令牌
        session_token = jwt.encode(
            {
                "session_id": session_id,
                "user_id": user_id,
                "iat": int(time.time()),
                "exp": int(expires_at.timestamp()),
                "capabilities": capabilities
            },
            "fallback_secret_key",  # 在生产环境中应该使用安全的密钥
            algorithm="HS256"
        )
        
        return {
            "token": session_token,
            "expires_at": expires_at,
            "connection_info": {
                "type": "websocket",
                "url": f"ws://localhost:{settings.PORT}/ws/voice/{session_id}",
                "protocol": "voicehelper-v1"
            }
        }


class LiveKitManager:
    """LiveKit房间和参与者管理"""
    
    def __init__(self):
        self.rooms = {}  # room_id -> room_info
        self.participants = {}  # participant_id -> participant_info
    
    async def create_room(self, room_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """创建LiveKit房间"""
        try:
            from livekit import api
            
            room_service = api.RoomService()
            
            # 创建房间配置
            room_config = api.CreateRoomRequest(
                name=room_name,
                empty_timeout=300,  # 5分钟空房间超时
                max_participants=10,
                metadata=json.dumps(config)
            )
            
            room = await room_service.create_room(room_config)
            
            self.rooms[room_name] = {
                "room": room,
                "created_at": datetime.utcnow(),
                "config": config,
                "participants": []
            }
            
            logger.info("LiveKit room created", room_name=room_name)
            return {"room_name": room_name, "sid": room.sid}
        
        except Exception as e:
            logger.error("Failed to create LiveKit room", error=str(e))
            raise
    
    async def add_participant(
        self, 
        room_name: str, 
        participant_id: str, 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """添加参与者到房间"""
        if room_name not in self.rooms:
            raise ValueError(f"Room {room_name} not found")
        
        participant_info = {
            "participant_id": participant_id,
            "joined_at": datetime.utcnow(),
            "metadata": metadata,
            "tracks": []
        }
        
        self.rooms[room_name]["participants"].append(participant_info)
        self.participants[participant_id] = participant_info
        
        logger.info(
            "Participant added to room",
            room_name=room_name,
            participant_id=participant_id
        )
        
        return participant_info
    
    async def remove_participant(self, room_name: str, participant_id: str):
        """从房间移除参与者"""
        if room_name in self.rooms:
            participants = self.rooms[room_name]["participants"]
            self.rooms[room_name]["participants"] = [
                p for p in participants if p["participant_id"] != participant_id
            ]
        
        if participant_id in self.participants:
            del self.participants[participant_id]
        
        logger.info(
            "Participant removed from room",
            room_name=room_name,
            participant_id=participant_id
        )
    
    async def cleanup_empty_rooms(self):
        """清理空房间"""
        empty_rooms = []
        
        for room_name, room_info in self.rooms.items():
            if not room_info["participants"]:
                # 检查房间是否超过空置时间
                empty_duration = datetime.utcnow() - room_info["created_at"]
                if empty_duration.total_seconds() > 300:  # 5分钟
                    empty_rooms.append(room_name)
        
        for room_name in empty_rooms:
            try:
                from livekit import api
                room_service = api.RoomService()
                await room_service.delete_room(api.DeleteRoomRequest(room=room_name))
                del self.rooms[room_name]
                logger.info("Empty room cleaned up", room_name=room_name)
            except Exception as e:
                logger.error("Failed to cleanup room", room_name=room_name, error=str(e))


class AudioMixer:
    """音频混音器 - 处理多路音频流"""
    
    def __init__(self):
        self.active_streams = {}
        self.system_audio_enabled = True
    
    async def add_audio_stream(
        self, 
        stream_id: str, 
        audio_source: Any,
        volume: float = 1.0
    ):
        """添加音频流"""
        self.active_streams[stream_id] = {
            "source": audio_source,
            "volume": volume,
            "enabled": True,
            "created_at": datetime.utcnow()
        }
        
        logger.info("Audio stream added", stream_id=stream_id, volume=volume)
    
    async def remove_audio_stream(self, stream_id: str):
        """移除音频流"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            logger.info("Audio stream removed", stream_id=stream_id)
    
    async def set_stream_volume(self, stream_id: str, volume: float):
        """设置流音量"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["volume"] = volume
            logger.debug("Stream volume updated", stream_id=stream_id, volume=volume)
    
    async def mute_stream(self, stream_id: str, muted: bool = True):
        """静音/取消静音流"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id]["enabled"] = not muted
            logger.info("Stream muted", stream_id=stream_id, muted=muted)
    
    async def play_system_sound(self, sound_type: str, volume: float = 0.5):
        """播放系统提示音"""
        if not self.system_audio_enabled:
            return
        
        # 这里可以集成系统提示音播放
        # 例如：连接音、断开音、错误音等
        logger.info("System sound played", sound_type=sound_type, volume=volume)
    
    def get_mixer_status(self) -> Dict[str, Any]:
        """获取混音器状态"""
        return {
            "active_streams": len(self.active_streams),
            "streams": {
                stream_id: {
                    "volume": info["volume"],
                    "enabled": info["enabled"],
                    "duration_seconds": (datetime.utcnow() - info["created_at"]).total_seconds()
                }
                for stream_id, info in self.active_streams.items()
            },
            "system_audio_enabled": self.system_audio_enabled
        }
