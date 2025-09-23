package config

import (
	"os"
	"strconv"
	"strings"
)

// Config 应用配置
type Config struct {
	Environment string `json:"environment"`
	Version     string `json:"version"`
	Port        int    `json:"port"`
	LogLevel    string `json:"log_level"`

	Server    ServerConfig    `json:"server"`
	Database  DatabaseConfig  `json:"database"`
	Redis     RedisConfig     `json:"redis"`
	JWT       JWTConfig       `json:"jwt"`
	Auth      AuthConfig      `json:"auth"`
	SSE       SSEConfig       `json:"sse"`
	WebSocket WebSocketConfig `json:"websocket"`
	Services  ServicesConfig  `json:"services"`
}

// ServerConfig 服务器配置
type ServerConfig struct {
	ReadTimeout  int `json:"read_timeout"`
	WriteTimeout int `json:"write_timeout"`
	IdleTimeout  int `json:"idle_timeout"`
}

// DatabaseConfig 数据库配置
type DatabaseConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Database string `json:"database"`
	Username string `json:"username"`
	Password string `json:"password"`
	SSLMode  string `json:"ssl_mode"`
}

// RedisConfig Redis配置
type RedisConfig struct {
	Host     string `json:"host"`
	Port     int    `json:"port"`
	Password string `json:"password"`
	Database int    `json:"database"`
}

// JWTConfig JWT配置
type JWTConfig struct {
	Secret     string `json:"secret"`
	ExpireHour int    `json:"expire_hour"`
}

// AuthConfig 认证配置
type AuthConfig struct {
	SkipPaths []string `json:"skip_paths"`
}

// SSEConfig SSE配置
type SSEConfig struct {
	MaxStreams        int `json:"max_streams"`
	EventQueueSize    int `json:"event_queue_size"`
	KeepAliveInterval int `json:"keep_alive_interval"`
	StreamTimeout     int `json:"stream_timeout"`
	MaxEventSize      int `json:"max_event_size"`
}

// WebSocketConfig WebSocket配置
type WebSocketConfig struct {
	MaxConnections    int `json:"max_connections"`
	SendQueueSize     int `json:"send_queue_size"`
	HeartbeatInterval int `json:"heartbeat_interval"`
	HeartbeatTimeout  int `json:"heartbeat_timeout"`
	ThrottleLimit     int `json:"throttle_limit"`
	MaxFrameSize      int `json:"max_frame_size"`
}

// ServicesConfig 服务配置
type ServicesConfig struct {
	AlgoServiceURL  string `json:"algo_service_url"`
	VoiceServiceURL string `json:"voice_service_url"`
}

// Load 加载配置
func Load() (*Config, error) {
	config := &Config{
		Environment: getEnv("ENVIRONMENT", "development"),
		Version:     getEnv("VERSION", "1.0.0"),
		Port:        getEnvInt("PORT", 8080),
		LogLevel:    getEnv("LOG_LEVEL", "info"),

		Server: ServerConfig{
			ReadTimeout:  getEnvInt("SERVER_READ_TIMEOUT", 30),
			WriteTimeout: getEnvInt("SERVER_WRITE_TIMEOUT", 30),
			IdleTimeout:  getEnvInt("SERVER_IDLE_TIMEOUT", 120),
		},

		Database: DatabaseConfig{
			Host:     getEnv("POSTGRES_HOST", "localhost"),
			Port:     getEnvInt("POSTGRES_PORT", 5432),
			Database: getEnv("POSTGRES_DB", "voicehelper"),
			Username: getEnv("POSTGRES_USER", "voicehelper"),
			Password: getEnv("POSTGRES_PASSWORD", "voicehelper123"),
			SSLMode:  getEnv("POSTGRES_SSL_MODE", "disable"),
		},

		Redis: RedisConfig{
			Host:     getEnv("REDIS_HOST", "localhost"),
			Port:     getEnvInt("REDIS_PORT", 6379),
			Password: getEnv("REDIS_PASSWORD", "redis123"),
			Database: getEnvInt("REDIS_DB", 0),
		},

		JWT: JWTConfig{
			Secret:     getEnv("JWT_SECRET", "your-secret-key"),
			ExpireHour: getEnvInt("JWT_EXPIRE_HOUR", 24),
		},

		Auth: AuthConfig{
			SkipPaths: getEnvStringSlice("AUTH_SKIP_PATHS", []string{
				"/health",
				"/metrics",
				"/api/v1/auth/",
			}),
		},

		SSE: SSEConfig{
			MaxStreams:        getEnvInt("SSE_MAX_STREAMS", 1000),
			EventQueueSize:    getEnvInt("SSE_EVENT_QUEUE_SIZE", 100),
			KeepAliveInterval: getEnvInt("SSE_KEEP_ALIVE_INTERVAL", 30),
			StreamTimeout:     getEnvInt("SSE_STREAM_TIMEOUT", 10),
			MaxEventSize:      getEnvInt("SSE_MAX_EVENT_SIZE", 65536),
		},

		WebSocket: WebSocketConfig{
			MaxConnections:    getEnvInt("WS_MAX_CONNECTIONS", 1000),
			SendQueueSize:     getEnvInt("WS_SEND_QUEUE_SIZE", 100),
			HeartbeatInterval: getEnvInt("WS_HEARTBEAT_INTERVAL", 30),
			HeartbeatTimeout:  getEnvInt("WS_HEARTBEAT_TIMEOUT", 60),
			ThrottleLimit:     getEnvInt("WS_THROTTLE_LIMIT", 50),
			MaxFrameSize:      getEnvInt("WS_MAX_FRAME_SIZE", 8192),
		},

		Services: ServicesConfig{
			AlgoServiceURL:  getEnv("ALGO_SERVICE_URL", "http://localhost:8000"),
			VoiceServiceURL: getEnv("VOICE_SERVICE_URL", "http://localhost:8001"),
		},
	}

	return config, nil
}

// 辅助函数

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvStringSlice(key string, defaultValue []string) []string {
	if value := os.Getenv(key); value != "" {
		return strings.Split(value, ",")
	}
	return defaultValue
}
