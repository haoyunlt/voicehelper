package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"chatbot/common/errors"
	"chatbot/common/logger"

	"github.com/gin-gonic/gin"
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
	Host        string
	ServiceName string
}

// loadConfig 加载配置
func loadConfig() *Config {
	return &Config{
		Port:        getEnv("PORT", getEnv("GATEWAY_PORT", "8080")),
		Environment: getEnv("ENV", "development"),
		LogLevel:    getEnv("LOG_LEVEL", "info"),
		Host:        getEnv("HOST", "0.0.0.0"),
		ServiceName: getEnv("SERVICE_NAME", getEnv("GATEWAY_SERVICE_NAME", "voicehelper-gateway")),
	}
}

// getEnv 获取环境变量
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getLocalIP 获取本地IP地址
func getLocalIP() string {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()

	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP.String()
}

// setupRouter 设置路由
func setupRouter(config *Config, log logger.Logger) *gin.Engine {
	if config.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()

	// 使用自定义日志中间件
	router.Use(logger.GinLoggerMiddleware(log))
	router.Use(gin.Recovery())

	// 健康检查
	router.GET("/health", func(c *gin.Context) {
		log.Debug("Health check requested",
			logger.F("client_ip", c.ClientIP()),
			logger.F("user_agent", c.GetHeader("User-Agent")),
		)

		response := gin.H{
			"status":     "ok",
			"version":    Version,
			"build_time": BuildTime,
			"git_commit": GitCommit,
			"timestamp":  time.Now().Unix(),
			"service":    config.ServiceName,
			"host":       config.Host,
			"port":       config.Port,
		}

		c.JSON(http.StatusOK, response)

		log.Business("Health check completed",
			logger.F("response", response),
		)
	})

	// 版本信息
	router.GET("/version", func(c *gin.Context) {
		log.Debug("Version info requested",
			logger.F("client_ip", c.ClientIP()),
		)

		response := gin.H{
			"version":    Version,
			"build_time": BuildTime,
			"git_commit": GitCommit,
			"service":    config.ServiceName,
		}

		c.JSON(http.StatusOK, response)
	})

	// API路由组
	api := router.Group("/api/v1")
	{
		api.GET("/ping", func(c *gin.Context) {
			log.Debug("Ping requested",
				logger.F("client_ip", c.ClientIP()),
			)

			response := gin.H{
				"message": "pong",
				"time":    time.Now().Unix(),
				"service": config.ServiceName,
			}

			c.JSON(http.StatusOK, response)
		})

		// 错误测试端点
		api.GET("/error-test", func(c *gin.Context) {
			log.ErrorWithCode(errors.GatewayInternalError, "This is a test error",
				logger.F("client_ip", c.ClientIP()),
				logger.F("test_param", c.Query("param")),
			)

			errorInfo := errors.GetErrorInfo(errors.GatewayInternalError)
			c.JSON(errorInfo.HTTPStatus, gin.H{
				"error": errorInfo,
			})
		})
	}

	return router
}

// gracefulShutdown 优雅关闭
func gracefulShutdown(server *http.Server, log logger.Logger) {
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Info("正在关闭服务器...",
		logger.F("server_addr", server.Addr),
	)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.ErrorWithCode(errors.GatewayInternalError, "服务器强制关闭",
			logger.F("error", err.Error()),
			logger.F("server_addr", server.Addr),
		)
	}

	log.Info("服务器已关闭",
		logger.F("server_addr", server.Addr),
	)
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
		fmt.Println("VoiceHelper Backend Server")
		fmt.Println("")
		fmt.Println("Usage:")
		fmt.Println("  voicehelper-backend [options]")
		fmt.Println("")
		fmt.Println("Options:")
		fmt.Println("  --version    显示版本信息")
		fmt.Println("  --help       显示帮助信息")
		fmt.Println("")
		fmt.Println("Environment Variables:")
		fmt.Println("  PORT         服务端口 (默认: 8080)")
		fmt.Println("  HOST         绑定主机 (默认: 0.0.0.0)")
		fmt.Println("  ENV          运行环境 (默认: development)")
		fmt.Println("  LOG_LEVEL    日志级别 (默认: info)")
		fmt.Println("  SERVICE_NAME 服务名称 (默认: voicehelper-backend)")
		return
	}

	// 加载配置
	config := loadConfig()

	// 初始化日志器
	logger.InitDefaultLogger(config.ServiceName)
	log := logger.GetDefaultLogger()

	// 获取本地IP
	localIP := getLocalIP()

	// 启动信息
	log.Startup("启动 VoiceHelper Backend Server",
		logger.F("version", Version),
		logger.F("build_time", BuildTime),
		logger.F("git_commit", GitCommit),
		logger.F("service", config.ServiceName),
		logger.F("environment", config.Environment),
		logger.F("log_level", config.LogLevel),
		logger.F("host", config.Host),
		logger.F("port", config.Port),
		logger.F("local_ip", localIP),
		logger.F("pid", os.Getpid()),
	)

	// 设置路由
	router := setupRouter(config, log)

	// 创建HTTP服务器
	serverAddr := config.Host + ":" + config.Port
	server := &http.Server{
		Addr:         serverAddr,
		Handler:      router,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// 启动服务器
	go func() {
		log.Startup("服务器启动监听",
			logger.F("address", serverAddr),
			logger.F("local_ip", localIP),
			logger.F("port", config.Port),
			logger.F("host", config.Host),
			logger.F("service_url", fmt.Sprintf("http://%s:%s", localIP, config.Port)),
			logger.F("health_check_url", fmt.Sprintf("http://%s:%s/health", localIP, config.Port)),
			logger.F("api_base_url", fmt.Sprintf("http://%s:%s/api/v1", localIP, config.Port)),
		)

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.ErrorWithCode(errors.GatewayInternalError, "服务器启动失败",
				logger.F("error", err.Error()),
				logger.F("address", serverAddr),
				logger.F("local_ip", localIP),
				logger.F("port", config.Port),
			)
			os.Exit(1)
		}
	}()

	// 记录服务器就绪日志
	log.Startup("服务器启动完成，等待连接",
		logger.F("service_ready", true),
		logger.F("address", serverAddr),
		logger.F("local_ip", localIP),
		logger.F("port", config.Port),
		logger.F("service_url", fmt.Sprintf("http://%s:%s", localIP, config.Port)),
		logger.F("endpoints", []string{
			fmt.Sprintf("GET http://%s:%s/health", localIP, config.Port),
			fmt.Sprintf("GET http://%s:%s/version", localIP, config.Port),
			fmt.Sprintf("GET http://%s:%s/api/v1/ping", localIP, config.Port),
			fmt.Sprintf("GET http://%s:%s/api/v1/error-test", localIP, config.Port),
		}),
	)

	// 优雅关闭
	gracefulShutdown(server, log)
}
