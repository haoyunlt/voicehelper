package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"chatbot-backend/internal/handler"
	"chatbot-backend/internal/service"
	"chatbot-backend/pkg/middleware"

	"github.com/gin-contrib/cors"
	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

func main() {
	// 初始化日志
	logrus.SetFormatter(&logrus.JSONFormatter{})
	logrus.SetLevel(logrus.InfoLevel)

	// 初始化服务
	services := service.NewServices()
	handlers := handler.NewHandlers(services)

	// 设置 Gin 模式
	if os.Getenv("GIN_MODE") == "" {
		gin.SetMode(gin.DebugMode)
	}

	// 创建路由
	r := gin.New()

	// 中间件
	r.Use(gin.Recovery())
	r.Use(middleware.Logger())
	r.Use(middleware.RequestID())
	r.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"http://localhost:3000"},
		AllowMethods:     []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowHeaders:     []string{"*"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: true,
		MaxAge:           12 * time.Hour,
	}))

	// 健康检查
	r.GET("/healthz", handlers.HealthCheck)

	// API 路由组
	api := r.Group("/api/v1")
	{
		// 认证相关（不需要JWT）
		auth := api.Group("/auth")
		{
			auth.POST("/wechat/miniprogram/login", handlers.WeChatMiniProgramLogin)
			auth.POST("/refresh", handlers.RefreshToken)
			auth.POST("/logout", handlers.Logout)
		}
		// 对话相关
		chat := api.Group("/chat")
		{
			chat.POST("/stream", handlers.ChatStream)
			chat.POST("/cancel", handlers.CancelChat)
		}

		// 语音相关
		voice := api.Group("/voice")
		{
			voice.GET("/stream", handlers.VoiceStream)
		}

		// 文档入库相关
		ingest := api.Group("/ingest")
		{
			ingest.POST("/upload", handlers.UploadFiles)
			ingest.POST("/url", handlers.IngestURL)
			ingest.GET("/tasks/:id", handlers.GetTask)
		}

		// 数据集管理
		datasets := api.Group("/datasets")
		{
			datasets.GET("", handlers.ListDatasets)
			datasets.GET("/:id", handlers.GetDataset)
		}

		// 搜索预览
		api.GET("/search", handlers.Search)
	}

	// 启动服务器
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	srv := &http.Server{
		Addr:    ":" + port,
		Handler: r,
	}

	// 优雅关闭
	go func() {
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("listen: %s\n", err)
		}
	}()

	logrus.Infof("Server started on port %s", port)

	// 等待中断信号
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logrus.Info("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Fatal("Server forced to shutdown:", err)
	}

	logrus.Info("Server exited")
}
