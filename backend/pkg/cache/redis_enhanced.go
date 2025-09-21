package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/go-redis/redis/v8"
	"github.com/sirupsen/logrus"
)

// RedisCache Redis缓存增强版
type RedisCache struct {
	client *redis.Client
	prefix string
}

// NewRedisCache 创建Redis缓存
func NewRedisCache(addr, password string, db int) *RedisCache {
	client := redis.NewClient(&redis.Options{
		Addr:         addr,
		Password:     password,
		DB:           db,
		PoolSize:     50,
		MinIdleConns: 10,
		MaxRetries:   3,
		ReadTimeout:  3 * time.Second,
		WriteTimeout: 3 * time.Second,
	})

	// 测试连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := client.Ping(ctx).Err(); err != nil {
		logrus.WithError(err).Error("Failed to connect to Redis")
	} else {
		logrus.Info("Connected to Redis successfully")
	}

	return &RedisCache{
		client: client,
		prefix: "chatbot:",
	}
}

// Close 关闭连接
func (c *RedisCache) Close() error {
	return c.client.Close()
}

// buildKey 构建键
func (c *RedisCache) buildKey(key string) string {
	return c.prefix + key
}

// Get 获取缓存
func (c *RedisCache) Get(ctx context.Context, key string, dest interface{}) error {
	fullKey := c.buildKey(key)

	val, err := c.client.Get(ctx, fullKey).Result()
	if err == redis.Nil {
		return fmt.Errorf("key not found: %s", key)
	}
	if err != nil {
		return fmt.Errorf("failed to get key %s: %w", key, err)
	}

	if dest != nil {
		if err := json.Unmarshal([]byte(val), dest); err != nil {
			return fmt.Errorf("failed to unmarshal value: %w", err)
		}
	}

	return nil
}

// Set 设置缓存
func (c *RedisCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
	fullKey := c.buildKey(key)

	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %w", err)
	}

	if err := c.client.Set(ctx, fullKey, data, ttl).Err(); err != nil {
		return fmt.Errorf("failed to set key %s: %w", key, err)
	}

	return nil
}

// Delete 删除缓存
func (c *RedisCache) Delete(ctx context.Context, keys ...string) error {
	fullKeys := make([]string, len(keys))
	for i, key := range keys {
		fullKeys[i] = c.buildKey(key)
	}

	if err := c.client.Del(ctx, fullKeys...).Err(); err != nil {
		return fmt.Errorf("failed to delete keys: %w", err)
	}

	return nil
}

// Exists 检查键是否存在
func (c *RedisCache) Exists(ctx context.Context, key string) (bool, error) {
	fullKey := c.buildKey(key)

	n, err := c.client.Exists(ctx, fullKey).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check key existence: %w", err)
	}

	return n > 0, nil
}

// Expire 设置过期时间
func (c *RedisCache) Expire(ctx context.Context, key string, ttl time.Duration) error {
	fullKey := c.buildKey(key)

	if err := c.client.Expire(ctx, fullKey, ttl).Err(); err != nil {
		return fmt.Errorf("failed to set expiration: %w", err)
	}

	return nil
}

// Increment 自增
func (c *RedisCache) Increment(ctx context.Context, key string) (int64, error) {
	fullKey := c.buildKey(key)

	val, err := c.client.Incr(ctx, fullKey).Result()
	if err != nil {
		return 0, fmt.Errorf("failed to increment key %s: %w", key, err)
	}

	return val, nil
}

// IncrementWithExpire 自增并设置过期时间
func (c *RedisCache) IncrementWithExpire(ctx context.Context, key string, ttl time.Duration) (int64, error) {
	fullKey := c.buildKey(key)

	pipe := c.client.Pipeline()
	incr := pipe.Incr(ctx, fullKey)
	pipe.Expire(ctx, fullKey, ttl)

	if _, err := pipe.Exec(ctx); err != nil {
		return 0, fmt.Errorf("failed to increment with expire: %w", err)
	}

	return incr.Val(), nil
}

// GetMultiple 批量获取
func (c *RedisCache) GetMultiple(ctx context.Context, keys []string) (map[string]string, error) {
	fullKeys := make([]string, len(keys))
	for i, key := range keys {
		fullKeys[i] = c.buildKey(key)
	}

	vals, err := c.client.MGet(ctx, fullKeys...).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get multiple keys: %w", err)
	}

	result := make(map[string]string)
	for i, val := range vals {
		if val != nil {
			result[keys[i]] = val.(string)
		}
	}

	return result, nil
}

// SetMultiple 批量设置
func (c *RedisCache) SetMultiple(ctx context.Context, items map[string]interface{}, ttl time.Duration) error {
	pipe := c.client.Pipeline()

	for key, value := range items {
		fullKey := c.buildKey(key)
		data, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("failed to marshal value for key %s: %w", key, err)
		}
		pipe.Set(ctx, fullKey, data, ttl)
	}

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to set multiple keys: %w", err)
	}

	return nil
}

// Lock 分布式锁
func (c *RedisCache) Lock(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	fullKey := c.buildKey("lock:" + key)

	ok, err := c.client.SetNX(ctx, fullKey, "1", ttl).Result()
	if err != nil {
		return false, fmt.Errorf("failed to acquire lock: %w", err)
	}

	return ok, nil
}

// Unlock 释放锁
func (c *RedisCache) Unlock(ctx context.Context, key string) error {
	fullKey := c.buildKey("lock:" + key)

	if err := c.client.Del(ctx, fullKey).Err(); err != nil {
		return fmt.Errorf("failed to release lock: %w", err)
	}

	return nil
}

// RateLimiter 速率限制器
type RateLimiter struct {
	cache     *RedisCache
	keyPrefix string
}

// NewRateLimiter 创建速率限制器
func NewRateLimiter(cache *RedisCache) *RateLimiter {
	return &RateLimiter{
		cache:     cache,
		keyPrefix: "ratelimit:",
	}
}

// CheckRateLimit 检查速率限制
func (r *RateLimiter) CheckRateLimit(ctx context.Context, identifier string, limit int, window time.Duration) (bool, int, error) {
	key := r.keyPrefix + identifier

	// 使用滑动窗口算法
	now := time.Now().Unix()
	windowStart := now - int64(window.Seconds())

	pipe := r.cache.client.Pipeline()

	// 移除窗口外的记录
	pipe.ZRemRangeByScore(ctx, r.cache.buildKey(key), "0", fmt.Sprintf("%d", windowStart))

	// 添加当前请求
	pipe.ZAdd(ctx, r.cache.buildKey(key), &redis.Z{
		Score:  float64(now),
		Member: fmt.Sprintf("%d:%s", now, generateRequestID()),
	})

	// 计算窗口内的请求数
	pipe.ZCard(ctx, r.cache.buildKey(key))

	// 设置过期时间
	pipe.Expire(ctx, r.cache.buildKey(key), window)

	cmds, err := pipe.Exec(ctx)
	if err != nil {
		return false, 0, fmt.Errorf("failed to check rate limit: %w", err)
	}

	// 获取当前请求数
	count := cmds[2].(*redis.IntCmd).Val()

	// 检查是否超限
	allowed := count <= int64(limit)
	remaining := limit - int(count)
	if remaining < 0 {
		remaining = 0
	}

	return allowed, remaining, nil
}

// GetRateLimitInfo 获取速率限制信息
func (r *RateLimiter) GetRateLimitInfo(ctx context.Context, identifier string) (int, time.Time, error) {
	key := r.keyPrefix + identifier

	// 获取当前请求数
	count, err := r.cache.client.ZCard(ctx, r.cache.buildKey(key)).Result()
	if err != nil {
		return 0, time.Time{}, fmt.Errorf("failed to get rate limit info: %w", err)
	}

	// 获取TTL
	ttl, err := r.cache.client.TTL(ctx, r.cache.buildKey(key)).Result()
	if err != nil {
		return 0, time.Time{}, fmt.Errorf("failed to get TTL: %w", err)
	}

	resetTime := time.Now().Add(ttl)

	return int(count), resetTime, nil
}

// generateRequestID 生成请求ID
func generateRequestID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano())
}

// CacheManager 缓存管理器
type CacheManager struct {
	cache *RedisCache
}

// NewCacheManager 创建缓存管理器
func NewCacheManager(cache *RedisCache) *CacheManager {
	return &CacheManager{cache: cache}
}

// CacheUser 缓存用户信息
func (m *CacheManager) CacheUser(ctx context.Context, userID string, user interface{}) error {
	key := fmt.Sprintf("user:%s", userID)
	return m.cache.Set(ctx, key, user, 1*time.Hour)
}

// GetCachedUser 获取缓存的用户信息
func (m *CacheManager) GetCachedUser(ctx context.Context, userID string, dest interface{}) error {
	key := fmt.Sprintf("user:%s", userID)
	return m.cache.Get(ctx, key, dest)
}

// CacheSession 缓存会话信息
func (m *CacheManager) CacheSession(ctx context.Context, sessionID string, session interface{}) error {
	key := fmt.Sprintf("session:%s", sessionID)
	return m.cache.Set(ctx, key, session, 24*time.Hour)
}

// GetCachedSession 获取缓存的会话信息
func (m *CacheManager) GetCachedSession(ctx context.Context, sessionID string, dest interface{}) error {
	key := fmt.Sprintf("session:%s", sessionID)
	return m.cache.Get(ctx, key, dest)
}

// InvalidateUserCache 失效用户缓存
func (m *CacheManager) InvalidateUserCache(ctx context.Context, userID string) error {
	keys := []string{
		fmt.Sprintf("user:%s", userID),
		fmt.Sprintf("user:permissions:%s", userID),
		fmt.Sprintf("user:stats:%s", userID),
	}
	return m.cache.Delete(ctx, keys...)
}

// CacheSearchResult 缓存搜索结果
func (m *CacheManager) CacheSearchResult(ctx context.Context, query, datasetID string, results interface{}) error {
	key := fmt.Sprintf("search:%s:%s", datasetID, hashString(query))
	return m.cache.Set(ctx, key, results, 10*time.Minute)
}

// GetCachedSearchResult 获取缓存的搜索结果
func (m *CacheManager) GetCachedSearchResult(ctx context.Context, query, datasetID string, dest interface{}) error {
	key := fmt.Sprintf("search:%s:%s", datasetID, hashString(query))
	return m.cache.Get(ctx, key, dest)
}

// hashString 哈希字符串（用于缓存键）
func hashString(s string) string {
	// 简单的哈希实现，实际可以使用更复杂的算法
	h := 0
	for _, c := range s {
		h = 31*h + int(c)
	}
	return fmt.Sprintf("%x", h)
}

// WarmCache 缓存预热
func (m *CacheManager) WarmCache(ctx context.Context) error {
	logrus.Info("Starting cache warm-up...")

	// TODO: 实现缓存预热逻辑
	// 1. 加载热门用户数据
	// 2. 加载常用配置
	// 3. 加载热门数据集信息

	logrus.Info("Cache warm-up completed")
	return nil
}

// GetCacheStats 获取缓存统计信息
func (m *CacheManager) GetCacheStats(ctx context.Context) (map[string]interface{}, error) {
	info, err := m.cache.client.Info(ctx, "stats").Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get cache stats: %w", err)
	}

	// 解析INFO输出
	stats := make(map[string]interface{})
	stats["raw_info"] = info

	// 获取内存使用情况
	memInfo, err := m.cache.client.Info(ctx, "memory").Result()
	if err == nil {
		stats["memory_info"] = memInfo
	}

	// 获取键空间信息
	dbSize, err := m.cache.client.DBSize(ctx).Result()
	if err == nil {
		stats["total_keys"] = dbSize
	}

	return stats, nil
}
