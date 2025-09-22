package ratelimit

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/sirupsen/logrus"
)

// RateLimiter Redis速率限制器
type RateLimiter struct {
	client *redis.Client
	prefix string
}

// RateLimitConfig 速率限制配置
type RateLimitConfig struct {
	Limit  int           // 限制次数
	Window time.Duration // 时间窗口
}

// RateLimitResult 速率限制结果
type RateLimitResult struct {
	Allowed    bool          // 是否允许
	Count      int64         // 当前计数
	Limit      int64         // 限制数量
	Remaining  int64         // 剩余次数
	ResetTime  time.Time     // 重置时间
	RetryAfter time.Duration // 重试间隔
}

// NewRateLimiter 创建Redis速率限制器
func NewRateLimiter(client *redis.Client, prefix string) *RateLimiter {
	return &RateLimiter{
		client: client,
		prefix: prefix,
	}
}

// Check 检查速率限制（滑动窗口算法）
func (rl *RateLimiter) Check(ctx context.Context, key string, config RateLimitConfig) (*RateLimitResult, error) {
	fullKey := rl.buildKey(key)
	now := time.Now()
	windowStart := now.Add(-config.Window)

	// 使用Lua脚本确保原子性
	luaScript := `
		local key = KEYS[1]
		local window_start = tonumber(ARGV[1])
		local now = tonumber(ARGV[2])
		local limit = tonumber(ARGV[3])
		local window_seconds = tonumber(ARGV[4])

		-- 清理过期记录
		redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)
		
		-- 获取当前计数
		local current = redis.call('ZCARD', key)
		
		-- 检查是否超限
		if current >= limit then
			local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
			local reset_time = window_start + window_seconds
			if #oldest > 0 then
				reset_time = oldest[2] + window_seconds
			end
			return {0, current, limit, 0, reset_time}
		end
		
		-- 添加当前请求
		redis.call('ZADD', key, now, now)
		redis.call('EXPIRE', key, window_seconds + 1)
		
		local new_count = current + 1
		local remaining = limit - new_count
		local reset_time = now + window_seconds
		
		return {1, new_count, limit, remaining, reset_time}
	`

	result, err := rl.client.Eval(ctx, luaScript, []string{fullKey},
		windowStart.Unix(),
		now.Unix(),
		config.Limit,
		int(config.Window.Seconds()),
	).Result()

	if err != nil {
		return nil, fmt.Errorf("failed to execute rate limit check: %w", err)
	}

	// 解析结果
	values, ok := result.([]interface{})
	if !ok || len(values) != 5 {
		return nil, fmt.Errorf("unexpected result format")
	}

	allowed := values[0].(int64) == 1
	count := values[1].(int64)
	limit := values[2].(int64)
	remaining := values[3].(int64)
	resetTime := time.Unix(values[4].(int64), 0)

	var retryAfter time.Duration
	if !allowed {
		retryAfter = time.Until(resetTime)
		if retryAfter < 0 {
			retryAfter = 0
		}
	}

	return &RateLimitResult{
		Allowed:    allowed,
		Count:      count,
		Limit:      limit,
		Remaining:  remaining,
		ResetTime:  resetTime,
		RetryAfter: retryAfter,
	}, nil
}

// CheckFixed 检查速率限制（固定窗口算法）
func (rl *RateLimiter) CheckFixed(ctx context.Context, key string, config RateLimitConfig) (*RateLimitResult, error) {
	now := time.Now()
	windowStart := now.Truncate(config.Window)
	fullKey := rl.buildKey(fmt.Sprintf("%s:%d", key, windowStart.Unix()))

	// 使用Lua脚本确保原子性
	luaScript := `
		local key = KEYS[1]
		local limit = tonumber(ARGV[1])
		local window_seconds = tonumber(ARGV[2])
		local now = tonumber(ARGV[3])
		
		local current = redis.call('GET', key)
		if current == false then
			current = 0
		else
			current = tonumber(current)
		end
		
		if current >= limit then
			local ttl = redis.call('TTL', key)
			local reset_time = now + ttl
			return {0, current, limit, 0, reset_time}
		end
		
		local new_count = redis.call('INCR', key)
		if new_count == 1 then
			redis.call('EXPIRE', key, window_seconds)
		end
		
		local remaining = limit - new_count
		local reset_time = now + window_seconds
		
		return {1, new_count, limit, remaining, reset_time}
	`

	result, err := rl.client.Eval(ctx, luaScript, []string{fullKey},
		config.Limit,
		int(config.Window.Seconds()),
		now.Unix(),
	).Result()

	if err != nil {
		return nil, fmt.Errorf("failed to execute fixed rate limit check: %w", err)
	}

	// 解析结果
	values, ok := result.([]interface{})
	if !ok || len(values) != 5 {
		return nil, fmt.Errorf("unexpected result format")
	}

	allowed := values[0].(int64) == 1
	count := values[1].(int64)
	limit := values[2].(int64)
	remaining := values[3].(int64)
	resetTime := time.Unix(values[4].(int64), 0)

	var retryAfter time.Duration
	if !allowed {
		retryAfter = time.Until(resetTime)
		if retryAfter < 0 {
			retryAfter = 0
		}
	}

	return &RateLimitResult{
		Allowed:    allowed,
		Count:      count,
		Limit:      limit,
		Remaining:  remaining,
		ResetTime:  resetTime,
		RetryAfter: retryAfter,
	}, nil
}

// CheckToken 检查令牌桶限制
func (rl *RateLimiter) CheckToken(ctx context.Context, key string, config RateLimitConfig) (*RateLimitResult, error) {
	fullKey := rl.buildKey(key)
	now := time.Now()

	// 令牌桶算法Lua脚本
	luaScript := `
		local key = KEYS[1]
		local capacity = tonumber(ARGV[1])
		local refill_rate = tonumber(ARGV[2])
		local now = tonumber(ARGV[3])
		local requested = tonumber(ARGV[4]) or 1
		
		local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
		local tokens = tonumber(bucket[1]) or capacity
		local last_refill = tonumber(bucket[2]) or now
		
		-- 计算需要添加的令牌数
		local elapsed = now - last_refill
		local tokens_to_add = math.floor(elapsed * refill_rate)
		tokens = math.min(capacity, tokens + tokens_to_add)
		
		-- 检查是否有足够的令牌
		if tokens >= requested then
			tokens = tokens - requested
			redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
			redis.call('EXPIRE', key, 3600) -- 1小时过期
			return {1, capacity - tokens, capacity, tokens, now + (capacity - tokens) / refill_rate}
		else
			redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
			redis.call('EXPIRE', key, 3600)
			local wait_time = (requested - tokens) / refill_rate
			return {0, capacity - tokens, capacity, tokens, now + wait_time}
		end
	`

	// 计算每秒补充的令牌数
	refillRate := float64(config.Limit) / config.Window.Seconds()

	result, err := rl.client.Eval(ctx, luaScript, []string{fullKey},
		config.Limit, // capacity
		refillRate,   // refill_rate
		now.Unix(),   // now
		1,            // requested tokens
	).Result()

	if err != nil {
		return nil, fmt.Errorf("failed to execute token bucket check: %w", err)
	}

	// 解析结果
	values, ok := result.([]interface{})
	if !ok || len(values) != 5 {
		return nil, fmt.Errorf("unexpected result format")
	}

	allowed := values[0].(int64) == 1
	count := values[1].(int64)
	limit := int64(config.Limit)
	remaining := values[3].(int64)
	resetTime := time.Unix(int64(values[4].(float64)), 0)

	var retryAfter time.Duration
	if !allowed {
		retryAfter = time.Until(resetTime)
		if retryAfter < 0 {
			retryAfter = 0
		}
	}

	return &RateLimitResult{
		Allowed:    allowed,
		Count:      count,
		Limit:      limit,
		Remaining:  remaining,
		ResetTime:  resetTime,
		RetryAfter: retryAfter,
	}, nil
}

// Reset 重置速率限制
func (rl *RateLimiter) Reset(ctx context.Context, key string) error {
	fullKey := rl.buildKey(key)
	return rl.client.Del(ctx, fullKey).Err()
}

// GetStatus 获取当前状态
func (rl *RateLimiter) GetStatus(ctx context.Context, key string) (int64, error) {
	fullKey := rl.buildKey(key)

	// 尝试获取计数器值
	val, err := rl.client.Get(ctx, fullKey).Result()
	if err != nil {
		if err == redis.Nil {
			return 0, nil
		}
		return 0, err
	}

	count, err := strconv.ParseInt(val, 10, 64)
	if err != nil {
		// 如果不是简单计数器，尝试获取有序集合大小
		count, err = rl.client.ZCard(ctx, fullKey).Result()
		if err != nil {
			return 0, err
		}
	}

	return count, nil
}

// buildKey 构建完整的键名
func (rl *RateLimiter) buildKey(key string) string {
	if rl.prefix == "" {
		return fmt.Sprintf("rate_limit:%s", key)
	}
	return fmt.Sprintf("%s:rate_limit:%s", rl.prefix, key)
}

// MultiCheck 批量检查多个键的速率限制
func (rl *RateLimiter) MultiCheck(ctx context.Context, keys []string, config RateLimitConfig) (map[string]*RateLimitResult, error) {
	results := make(map[string]*RateLimitResult)

	for _, key := range keys {
		result, err := rl.Check(ctx, key, config)
		if err != nil {
			logrus.WithError(err).Errorf("Failed to check rate limit for key: %s", key)
			// 在出错时允许请求通过，避免影响正常业务
			results[key] = &RateLimitResult{
				Allowed:   true,
				Count:     0,
				Limit:     int64(config.Limit),
				Remaining: int64(config.Limit),
				ResetTime: time.Now().Add(config.Window),
			}
			continue
		}
		results[key] = result
	}

	return results, nil
}

// Cleanup 清理过期的速率限制数据
func (rl *RateLimiter) Cleanup(ctx context.Context, pattern string) error {
	fullPattern := rl.buildKey(pattern)

	keys, err := rl.client.Keys(ctx, fullPattern).Result()
	if err != nil {
		return fmt.Errorf("failed to get keys for cleanup: %w", err)
	}

	if len(keys) == 0 {
		return nil
	}

	// 批量删除过期键
	pipe := rl.client.Pipeline()
	for _, key := range keys {
		pipe.Del(ctx, key)
	}

	_, err = pipe.Exec(ctx)
	return err
}
