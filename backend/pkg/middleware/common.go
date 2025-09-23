package middleware

import (
	"context"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"

	"voicehelper/backend/pkg/cache"
)

// CORS 跨域中间件
func CORS() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		origin := c.Request.Header.Get("Origin")

		// 允许的源列表（生产环境应该配置具体域名）
		allowedOrigins := []string{
			"http://localhost:3000",
			"http://localhost:3001",
			"http://localhost:3002",
			"https://voicehelper.ai",
		}

		// 检查是否为允许的源
		allowed := false
		for _, allowedOrigin := range allowedOrigins {
			if origin == allowedOrigin {
				allowed = true
				break
			}
		}

		if allowed || origin == "" {
			c.Header("Access-Control-Allow-Origin", origin)
		}

		c.Header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
		c.Header("Access-Control-Allow-Headers", "Origin, Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization, X-Tenant-ID, X-API-Key, X-Request-ID")
		c.Header("Access-Control-Expose-Headers", "Content-Length, X-New-Token, X-Request-ID")
		c.Header("Access-Control-Allow-Credentials", "true")
		c.Header("Access-Control-Max-Age", "86400")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(http.StatusNoContent)
			return
		}

		c.Next()
	})
}

// RateLimit 限流中间件
func RateLimit(redisClient *cache.RedisClient) gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		// 获取客户端标识
		clientID := getClientID(c)

		// 限流规则：每分钟最多100个请求
		limit := int64(100)
		window := time.Minute

		// Redis键
		key := fmt.Sprintf("rate_limit:%s", clientID)

		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()

		// 获取当前计数
		current, err := redisClient.Incr(ctx, key)
		if err != nil {
			logrus.WithError(err).Error("Failed to increment rate limit counter")
			c.Next()
			return
		}

		// 如果是第一次请求，设置过期时间
		if current == 1 {
			if err := redisClient.Expire(ctx, key, window); err != nil {
				logrus.WithError(err).Error("Failed to set rate limit expiration")
			}
		}

		// 检查是否超过限制
		if current > limit {
			c.Header("X-RateLimit-Limit", strconv.FormatInt(limit, 10))
			c.Header("X-RateLimit-Remaining", "0")
			c.Header("X-RateLimit-Reset", strconv.FormatInt(time.Now().Add(window).Unix(), 10))

			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "Rate limit exceeded",
				"retry_after": window.Seconds(),
			})
			c.Abort()
			return
		}

		// 设置响应头
		remaining := limit - current
		if remaining < 0 {
			remaining = 0
		}

		c.Header("X-RateLimit-Limit", strconv.FormatInt(limit, 10))
		c.Header("X-RateLimit-Remaining", strconv.FormatInt(remaining, 10))
		c.Header("X-RateLimit-Reset", strconv.FormatInt(time.Now().Add(window).Unix(), 10))

		c.Next()
	})
}

// getClientID 获取客户端标识
func getClientID(c *gin.Context) string {
	// 优先使用用户ID
	if userID := c.GetString("user_id"); userID != "" {
		return fmt.Sprintf("user:%s", userID)
	}

	// 使用租户ID
	if tenantID := c.GetString("tenant_id"); tenantID != "" {
		return fmt.Sprintf("tenant:%s", tenantID)
	}

	// 使用IP地址
	return fmt.Sprintf("ip:%s", c.ClientIP())
}

// Logging 日志中间件
func Logging() gin.HandlerFunc {
	return gin.LoggerWithFormatter(func(param gin.LogFormatterParams) string {
		logrus.WithFields(logrus.Fields{
			"timestamp":  param.TimeStamp.Format(time.RFC3339),
			"status":     param.StatusCode,
			"latency":    param.Latency,
			"client_ip":  param.ClientIP,
			"method":     param.Method,
			"path":       param.Path,
			"user_agent": param.Request.UserAgent(),
			"request_id": param.Keys["request_id"],
		}).Info("HTTP Request")

		return ""
	})
}

// Recovery 恢复中间件
func Recovery() gin.HandlerFunc {
	return gin.CustomRecovery(func(c *gin.Context, recovered interface{}) {
		logrus.WithFields(logrus.Fields{
			"error":      recovered,
			"request_id": c.GetString("request_id"),
			"path":       c.Request.URL.Path,
			"method":     c.Request.Method,
		}).Error("Panic recovered")

		c.JSON(http.StatusInternalServerError, gin.H{
			"error":      "Internal server error",
			"request_id": c.GetString("request_id"),
		})
	})
}

// Maintenance 维护模式中间件
func Maintenance() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		// TODO: 从配置或Redis中读取维护模式状态
		maintenanceMode := false

		if maintenanceMode {
			// 允许健康检查和管理接口
			if c.Request.URL.Path == "/health" ||
				c.Request.URL.Path == "/metrics" ||
				c.Request.URL.Path == "/api/v1/admin/maintenance" {
				c.Next()
				return
			}

			c.JSON(http.StatusServiceUnavailable, gin.H{
				"error":   "Service is under maintenance",
				"message": "Please try again later",
			})
			c.Abort()
			return
		}

		c.Next()
	})
}
