package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"voicehelper/backend/internal/handlers"
	"voicehelper/backend/pkg/cache"
	"voicehelper/backend/pkg/config"
	"voicehelper/backend/pkg/database"
	"voicehelper/backend/pkg/middleware"
)

func main() {
	// 加载配置
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// 设置日志级别
	level, err := logrus.ParseLevel(cfg.LogLevel)
	if err != nil {
		level = logrus.InfoLevel
	}
	logrus.SetLevel(level)
	logrus.SetFormatter(&logrus.JSONFormatter{})

	// 初始化数据库
	db, err := database.NewPostgresConnection(cfg.Database)
	if err != nil {
		logrus.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	// 初始化缓存
	redisClient, err := cache.NewRedisClient(cfg.Redis)
	if err != nil {
		logrus.Fatalf("Failed to connect to Redis: %v", err)
	}
	defer redisClient.Close()

	// 初始化事件总线
	eventBus := handlers.NewEventBus(redisClient)

	// 创建处理器
	sseConfig := handlers.SSEConfig{
		MaxStreams:        cfg.SSE.MaxStreams,
		EventQueueSize:    cfg.SSE.EventQueueSize,
		KeepAliveInterval: time.Duration(cfg.SSE.KeepAliveInterval) * time.Second,
		StreamTimeout:     time.Duration(cfg.SSE.StreamTimeout) * time.Minute,
		MaxEventSize:      cfg.SSE.MaxEventSize,
	}
	chatSSEHandler := handlers.NewChatSSEHandler(eventBus, sseConfig)

	wsConfig := handlers.WSConfig{
		MaxConnections:    cfg.WebSocket.MaxConnections,
		SendQueueSize:     cfg.WebSocket.SendQueueSize,
		HeartbeatInterval: time.Duration(cfg.WebSocket.HeartbeatInterval) * time.Second,
		HeartbeatTimeout:  time.Duration(cfg.WebSocket.HeartbeatTimeout) * time.Second,
		ThrottleLimit:     int32(cfg.WebSocket.ThrottleLimit),
		MaxFrameSize:      cfg.WebSocket.MaxFrameSize,
	}
	voiceWSHandler := handlers.NewVoiceWSHandler(eventBus, wsConfig)

	// 创建中间件
	authMiddleware := middleware.NewAuthMiddleware(cfg.JWT.Secret, cfg.Auth.SkipPaths)
	rbacMiddleware := middleware.NewRBACMiddleware()
	tenantMiddleware := middleware.NewTenantMiddleware()

	// 创建API处理器
	apiHandler := handlers.NewAPIHandler(
		chatSSEHandler,
		voiceWSHandler,
		authMiddleware,
		rbacMiddleware,
		tenantMiddleware,
	)

	// 设置Gin模式
	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	// 创建路由
	router := gin.New()

	// 添加全局中间件
	router.Use(gin.Logger())
	router.Use(gin.Recovery())
	router.Use(middleware.CORS())
	router.Use(middleware.RequestID())
	router.Use(middleware.RateLimit(redisClient))

	// 设置路由
	apiHandler.SetupRoutes(router)

	// 创建HTTP服务器
	server := &http.Server{
		Addr:         fmt.Sprintf(":%d", cfg.Port),
		Handler:      router,
		ReadTimeout:  time.Duration(cfg.Server.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.Server.WriteTimeout) * time.Second,
		IdleTimeout:  time.Duration(cfg.Server.IdleTimeout) * time.Second,
	}

	// 启动服务器
	go func() {
		logrus.WithFields(logrus.Fields{
			"port":        cfg.Port,
			"environment": cfg.Environment,
			"version":     cfg.Version,
		}).Info("Starting VoiceHelper Gateway Server")

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			logrus.Fatalf("Failed to start server: %v", err)
		}
	}()

	// 等待中断信号
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logrus.Info("Shutting down server...")

	// 优雅关闭
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logrus.Errorf("Server forced to shutdown: %v", err)
	}

	logrus.Info("Server exited")
}
