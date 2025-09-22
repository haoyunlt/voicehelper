package middleware

import (
	"fmt"
	"net/http"
	"strconv"
	"time"
	"voicehelper/backend/pkg/ratelimit"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// RateLimitMiddleware 速率限制中间件
func RateLimitMiddleware(rateLimiter *ratelimit.RateLimiter, config ratelimit.RateLimitConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 获取客户端标识（IP地址）
		clientIP := c.ClientIP()
		key := fmt.Sprintf("ip:%s", clientIP)

		// 检查速率限制
		result, err := rateLimiter.Check(c.Request.Context(), key, config)
		if err != nil {
			logrus.WithError(err).Error("Rate limiter check failed")
			// 如果速率限制器出错，允许请求继续
			c.Next()
			return
		}

		// 设置响应头
		c.Header("X-RateLimit-Limit", strconv.FormatInt(result.Limit, 10))
		c.Header("X-RateLimit-Remaining", strconv.FormatInt(result.Remaining, 10))
		c.Header("X-RateLimit-Reset", strconv.FormatInt(result.ResetTime.Unix(), 10))

		// 如果超过限制，返回429错误
		if !result.Allowed {
			c.Header("Retry-After", strconv.FormatInt(int64(result.RetryAfter.Seconds()), 10))
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "Rate limit exceeded",
				"message":     "Too many requests",
				"retry_after": int64(result.RetryAfter.Seconds()),
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// APIKeyRateLimitMiddleware API密钥速率限制中间件
func APIKeyRateLimitMiddleware(rateLimiter *ratelimit.RateLimiter) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 获取API密钥
		apiKey := c.GetHeader("X-API-Key")
		if apiKey == "" {
			// 如果没有API密钥，跳过速率限制
			c.Next()
			return
		}

		// 从上下文获取API密钥信息（假设已经通过认证中间件验证）
		keyInfo, exists := c.Get("api_key_info")
		if !exists {
			c.Next()
			return
		}

		// 类型断言获取速率限制配置
		type APIKeyInfo struct {
			ID        string `json:"id"`
			RateLimit int    `json:"rate_limit"`
		}

		info, ok := keyInfo.(APIKeyInfo)
		if !ok || info.RateLimit <= 0 {
			c.Next()
			return
		}

		// 配置速率限制
		config := ratelimit.RateLimitConfig{
			Limit:  info.RateLimit,
			Window: time.Minute, // 每分钟限制
		}

		key := fmt.Sprintf("apikey:%s", info.ID)

		// 检查速率限制
		result, err := rateLimiter.CheckFixed(c.Request.Context(), key, config)
		if err != nil {
			logrus.WithError(err).Error("API key rate limiter check failed")
			// 如果速率限制器出错，允许请求继续
			c.Next()
			return
		}

		// 设置响应头
		c.Header("X-RateLimit-Limit", strconv.FormatInt(result.Limit, 10))
		c.Header("X-RateLimit-Remaining", strconv.FormatInt(result.Remaining, 10))
		c.Header("X-RateLimit-Reset", strconv.FormatInt(result.ResetTime.Unix(), 10))

		// 如果超过限制，返回429错误
		if !result.Allowed {
			c.Header("Retry-After", strconv.FormatInt(int64(result.RetryAfter.Seconds()), 10))
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "API key rate limit exceeded",
				"message":     "Too many requests for this API key",
				"retry_after": int64(result.RetryAfter.Seconds()),
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// UserRateLimitMiddleware 用户速率限制中间件
func UserRateLimitMiddleware(rateLimiter *ratelimit.RateLimiter, config ratelimit.RateLimitConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 获取用户ID
		userID, exists := c.Get("user_id")
		if !exists {
			// 如果没有用户ID，使用IP地址
			userID = c.ClientIP()
		}

		key := fmt.Sprintf("user:%v", userID)

		// 检查速率限制
		result, err := rateLimiter.Check(c.Request.Context(), key, config)
		if err != nil {
			logrus.WithError(err).Error("User rate limiter check failed")
			// 如果速率限制器出错，允许请求继续
			c.Next()
			return
		}

		// 设置响应头
		c.Header("X-RateLimit-Limit", strconv.FormatInt(result.Limit, 10))
		c.Header("X-RateLimit-Remaining", strconv.FormatInt(result.Remaining, 10))
		c.Header("X-RateLimit-Reset", strconv.FormatInt(result.ResetTime.Unix(), 10))

		// 如果超过限制，返回429错误
		if !result.Allowed {
			c.Header("Retry-After", strconv.FormatInt(int64(result.RetryAfter.Seconds()), 10))
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "User rate limit exceeded",
				"message":     "Too many requests for this user",
				"retry_after": int64(result.RetryAfter.Seconds()),
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// TenantRateLimitMiddleware 租户速率限制中间件
func TenantRateLimitMiddleware(rateLimiter *ratelimit.RateLimiter, config ratelimit.RateLimitConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 获取租户ID
		tenantID, exists := c.Get("tenant_id")
		if !exists {
			// 如果没有租户ID，跳过速率限制
			c.Next()
			return
		}

		key := fmt.Sprintf("tenant:%v", tenantID)

		// 检查速率限制
		result, err := rateLimiter.Check(c.Request.Context(), key, config)
		if err != nil {
			logrus.WithError(err).Error("Tenant rate limiter check failed")
			// 如果速率限制器出错，允许请求继续
			c.Next()
			return
		}

		// 设置响应头
		c.Header("X-RateLimit-Limit", strconv.FormatInt(result.Limit, 10))
		c.Header("X-RateLimit-Remaining", strconv.FormatInt(result.Remaining, 10))
		c.Header("X-RateLimit-Reset", strconv.FormatInt(result.ResetTime.Unix(), 10))

		// 如果超过限制，返回429错误
		if !result.Allowed {
			c.Header("Retry-After", strconv.FormatInt(int64(result.RetryAfter.Seconds()), 10))
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "Tenant rate limit exceeded",
				"message":     "Too many requests for this tenant",
				"retry_after": int64(result.RetryAfter.Seconds()),
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// EndpointRateLimitMiddleware 端点特定速率限制中间件
func EndpointRateLimitMiddleware(rateLimiter *ratelimit.RateLimiter, endpoint string, config ratelimit.RateLimitConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		// 组合键：IP + 端点
		clientIP := c.ClientIP()
		key := fmt.Sprintf("endpoint:%s:ip:%s", endpoint, clientIP)

		// 检查速率限制
		result, err := rateLimiter.Check(c.Request.Context(), key, config)
		if err != nil {
			logrus.WithError(err).Error("Endpoint rate limiter check failed")
			// 如果速率限制器出错，允许请求继续
			c.Next()
			return
		}

		// 设置响应头
		c.Header("X-RateLimit-Limit", strconv.FormatInt(result.Limit, 10))
		c.Header("X-RateLimit-Remaining", strconv.FormatInt(result.Remaining, 10))
		c.Header("X-RateLimit-Reset", strconv.FormatInt(result.ResetTime.Unix(), 10))

		// 如果超过限制，返回429错误
		if !result.Allowed {
			c.Header("Retry-After", strconv.FormatInt(int64(result.RetryAfter.Seconds()), 10))
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "Endpoint rate limit exceeded",
				"message":     fmt.Sprintf("Too many requests to %s", endpoint),
				"retry_after": int64(result.RetryAfter.Seconds()),
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// AdaptiveRateLimitMiddleware 自适应速率限制中间件
func AdaptiveRateLimitMiddleware(rateLimiter *ratelimit.RateLimiter, baseConfig ratelimit.RateLimitConfig) gin.HandlerFunc {
	return func(c *gin.Context) {
		clientIP := c.ClientIP()
		key := fmt.Sprintf("adaptive:ip:%s", clientIP)

		// 获取当前负载情况（这里简化实现）
		// 实际应用中可以根据系统负载、错误率等动态调整限制
		config := baseConfig

		// 检查是否是可信IP（例如内网IP）
		if isInternalIP(clientIP) {
			// 内网IP给予更高的限制
			config.Limit = config.Limit * 2
		}

		// 检查速率限制
		result, err := rateLimiter.Check(c.Request.Context(), key, config)
		if err != nil {
			logrus.WithError(err).Error("Adaptive rate limiter check failed")
			c.Next()
			return
		}

		// 设置响应头
		c.Header("X-RateLimit-Limit", strconv.FormatInt(result.Limit, 10))
		c.Header("X-RateLimit-Remaining", strconv.FormatInt(result.Remaining, 10))
		c.Header("X-RateLimit-Reset", strconv.FormatInt(result.ResetTime.Unix(), 10))

		if !result.Allowed {
			c.Header("Retry-After", strconv.FormatInt(int64(result.RetryAfter.Seconds()), 10))
			c.JSON(http.StatusTooManyRequests, gin.H{
				"error":       "Rate limit exceeded",
				"message":     "Too many requests",
				"retry_after": int64(result.RetryAfter.Seconds()),
			})
			c.Abort()
			return
		}

		c.Next()
	}
}

// isInternalIP 检查是否是内网IP
func isInternalIP(ip string) bool {
	// 简化实现，实际应用中应该更完善
	return ip == "127.0.0.1" || ip == "::1" ||
		(len(ip) >= 7 && ip[:7] == "192.168") ||
		(len(ip) >= 3 && ip[:3] == "10.") ||
		(len(ip) >= 7 && ip[:7] == "172.16")
}
