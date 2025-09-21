package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// 构建信息（由构建脚本注入）
var (
	Version   = "dev"
	BuildTime = "unknown"
	GitCommit = "unknown"
)

// Config 应用配置
type Config struct {
	Port        string
	Environment string
	LogLevel    string
}

// loadConfig 加载配置
func loadConfig() *Config {
	return &Config{
		Port:        getEnv("PORT", "8080"),
		Environment: getEnv("ENV", "development"),
		LogLevel:    getEnv("LOG_LEVEL", "info"),
	}
}

// getEnv 获取环境变量
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// setupLogger 设置日志
func setupLogger(level string) {
	logrus.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: time.RFC3339,
	})

	switch level {
	case "debug":
		logrus.SetLevel(logrus.DebugLevel)
	case "info":
		logrus.SetLevel(logrus.InfoLevel)
	case "warn":
		logrus.SetLevel(logrus.WarnLevel)
	case "error":
		logrus.SetLevel(logrus.ErrorLevel)
	default:
		logrus.SetLevel(logrus.InfoLevel)
	}
}

// setupRouter 设置路由
func setupRouter(config *Config) *gin.Engine {
	if config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// 健康检查
	router.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":     "ok",
			"version":    Version,
			"build_time": BuildTime,
			"git_commit": GitCommit,
			"timestamp":  time.Now().Unix(),
		})
	})

	// 版本信息
	router.GET("/version", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"version":    Version,
			"build_time": BuildTime,
			"git_commit": GitCommit,
		})
	})

	// API路由组
	api := router.Group("/api/v1")
	{
		api.GET("/ping", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{
				"message": "pong",
				"time":    time.Now().Unix(),
			})
		})
	}

	return router
}

// gracefulShutdown 优雅关闭
func gracefulShutdown(server *http.Server) {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	logrus.Info("正在关闭服务器...")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		logrus.WithError(err).Error("服务器强制关闭")
	}

	logrus.Info("服务器已关闭")
}

func main() {
	// 命令行参数
	var (
		showVersion = flag.Bool("version", false, "显示版本信息")
		showHelp    = flag.Bool("help", false, "显示帮助信息")
	)
	flag.Parse()

	if *showVersion {
		fmt.Printf("版本: %s\n", Version)
		fmt.Printf("构建时间: %s\n", BuildTime)
		fmt.Printf("Git提交: %s\n", GitCommit)
		return
	}

	if *showHelp {
		fmt.Println("Chatbot Backend Server")
		fmt.Println("")
		fmt.Println("Usage:")
		fmt.Println("  chatbot-backend [options]")
		fmt.Println("")
		fmt.Println("Options:")
		fmt.Println("  --version    显示版本信息")
		fmt.Println("  --help       显示帮助信息")
		fmt.Println("")
		fmt.Println("Environment Variables:")
		fmt.Println("  PORT         服务端口 (默认: 8080)")
		fmt.Println("  ENV          运行环境 (默认: development)")
		fmt.Println("  LOG_LEVEL    日志级别 (默认: info)")
		return
	}

	// 加载配置
	config := loadConfig()

	// 设置日志
	setupLogger(config.LogLevel)

	// 启动信息
	logrus.WithFields(logrus.Fields{
		"version":    Version,
		"build_time": BuildTime,
		"git_commit": GitCommit,
		"port":       config.Port,
		"env":        config.Environment,
	}).Info("启动 Chatbot Backend Server")

	// 设置路由
	router := setupRouter(config)

	// 创建HTTP服务器
	server := &http.Server{
		Addr:         ":" + config.Port,
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// 启动服务器
	go func() {
		logrus.Infof("服务器启动在端口 %s", config.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("服务器启动失败: %v", err)
		}
	}()

	// 优雅关闭
	gracefulShutdown(server)
}