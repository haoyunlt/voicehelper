package middleware

import (
	"strconv"
	"time"

	"chatbot/pkg/metrics"

	"github.com/gin-gonic/gin"
)

// MetricsMiddleware 指标收集中间件
func MetricsMiddleware(collector *metrics.MetricsCollector) gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		// 处理请求
		c.Next()

		// 记录指标
		duration := time.Since(start)
		status := strconv.Itoa(c.Writer.Status())

		collector.RecordHTTPRequest(
			c.Request.Method,
			c.FullPath(),
			status,
			"backend",
			duration,
		)
	}
}

// HealthCheckMiddleware 健康检查中间件
func HealthCheckMiddleware(collector *metrics.MetricsCollector) gin.HandlerFunc {
	return func(c *gin.Context) {
		if c.Request.URL.Path == "/health" {
			c.JSON(200, gin.H{
				"status":    "healthy",
				"timestamp": time.Now().Unix(),
				"service":   "chatbot-backend",
			})
			c.Abort()
			return
		}
		c.Next()
	}
}
