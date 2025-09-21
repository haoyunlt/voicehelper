package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/sirupsen/logrus"
)

// RedisCache Redis缓存实现
type RedisCache struct {
	client *redis.Client
	prefix string
	ttl    time.Duration
}

// CacheConfig 缓存配置
type CacheConfig struct {
	Addr     string
	Password string
	DB       int
	Prefix   string
	TTL      time.Duration
}

// NewRedisCache 创建Redis缓存实例
func NewRedisCache(config CacheConfig) (*RedisCache, error) {
	client := redis.NewClient(&redis.Options{
		Addr:         config.Addr,
		Password:     config.Password,
		DB:           config.DB,
		PoolSize:     10,
		MinIdleConns: 5,
	})

	// 测试连接
	ctx := context.Background()
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("failed to connect to Redis: %w", err)
	}

	return &RedisCache{
		client: client,
		prefix: config.Prefix,
		ttl:    config.TTL,
	}, nil
}

// ==================== 基础操作 ====================

// Set 设置缓存
func (r *RedisCache) Set(ctx context.Context, key string, value interface{}, ttl ...time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %w", err)
	}

	expiration := r.ttl
	if len(ttl) > 0 {
		expiration = ttl[0]
	}

	fullKey := r.buildKey(key)
	return r.client.Set(ctx, fullKey, data, expiration).Err()
}

// Get 获取缓存
func (r *RedisCache) Get(ctx context.Context, key string, dest interface{}) error {
	fullKey := r.buildKey(key)
	data, err := r.client.Get(ctx, fullKey).Bytes()
	if err != nil {
		if err == redis.Nil {
			return ErrCacheMiss
		}
		return fmt.Errorf("failed to get cache: %w", err)
	}

	if err := json.Unmarshal(data, dest); err != nil {
		return fmt.Errorf("failed to unmarshal value: %w", err)
	}

	// 更新访问时间（用于LRU）
	r.client.Touch(ctx, fullKey)

	return nil
}

// Delete 删除缓存
func (r *RedisCache) Delete(ctx context.Context, keys ...string) error {
	fullKeys := make([]string, len(keys))
	for i, key := range keys {
		fullKeys[i] = r.buildKey(key)
	}
	return r.client.Del(ctx, fullKeys...).Err()
}

// Exists 检查缓存是否存在
func (r *RedisCache) Exists(ctx context.Context, key string) (bool, error) {
	fullKey := r.buildKey(key)
	count, err := r.client.Exists(ctx, fullKey).Result()
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

// ==================== 会话缓存 ====================

// SessionData 会话数据
type SessionData struct {
	ConversationID string                 `json:"conversation_id"`
	TenantID       string                 `json:"tenant_id"`
	UserID         string                 `json:"user_id"`
	Messages       []Message              `json:"messages"`
	Context        map[string]interface{} `json:"context"`
	CreatedAt      time.Time              `json:"created_at"`
	UpdatedAt      time.Time              `json:"updated_at"`
}

// Message 消息结构
type Message struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

// SetSession 设置会话缓存
func (r *RedisCache) SetSession(ctx context.Context, sessionID string, data *SessionData) error {
	key := fmt.Sprintf("session:%s", sessionID)
	data.UpdatedAt = time.Now()
	return r.Set(ctx, key, data, 30*time.Minute) // 会话缓存30分钟
}

// GetSession 获取会话缓存
func (r *RedisCache) GetSession(ctx context.Context, sessionID string) (*SessionData, error) {
	key := fmt.Sprintf("session:%s", sessionID)
	var data SessionData
	if err := r.Get(ctx, key, &data); err != nil {
		return nil, err
	}
	return &data, nil
}

// AppendMessage 追加消息到会话
func (r *RedisCache) AppendMessage(ctx context.Context, sessionID string, message Message) error {
	session, err := r.GetSession(ctx, sessionID)
	if err != nil {
		if err == ErrCacheMiss {
			// 创建新会话
			session = &SessionData{
				ConversationID: sessionID,
				Messages:       []Message{message},
				CreatedAt:      time.Now(),
			}
		} else {
			return err
		}
	} else {
		session.Messages = append(session.Messages, message)
	}

	return r.SetSession(ctx, sessionID, session)
}

// ==================== 语义缓存 ====================

// SemanticCache 语义缓存
type SemanticCache struct {
	Query      string                   `json:"query"`
	Embedding  []float32                `json:"embedding"`
	Response   string                   `json:"response"`
	References []map[string]interface{} `json:"references"`
	Metadata   map[string]interface{}   `json:"metadata"`
	HitCount   int                      `json:"hit_count"`
	CreatedAt  time.Time                `json:"created_at"`
	LastHitAt  time.Time                `json:"last_hit_at"`
}

// SetSemanticCache 设置语义缓存
func (r *RedisCache) SetSemanticCache(ctx context.Context, queryHash string, cache *SemanticCache) error {
	key := fmt.Sprintf("semantic:%s", queryHash)
	cache.CreatedAt = time.Now()
	cache.LastHitAt = time.Now()
	return r.Set(ctx, key, cache, 24*time.Hour) // 语义缓存24小时
}

// GetSemanticCache 获取语义缓存
func (r *RedisCache) GetSemanticCache(ctx context.Context, queryHash string) (*SemanticCache, error) {
	key := fmt.Sprintf("semantic:%s", queryHash)
	var cache SemanticCache
	if err := r.Get(ctx, key, &cache); err != nil {
		return nil, err
	}

	// 更新命中统计
	cache.HitCount++
	cache.LastHitAt = time.Now()
	r.Set(ctx, key, cache, 24*time.Hour)

	logrus.WithFields(logrus.Fields{
		"query_hash": queryHash,
		"hit_count":  cache.HitCount,
	}).Info("Semantic cache hit")

	return &cache, nil
}

// ==================== 热点数据缓存 ====================

// SetHotData 设置热点数据
func (r *RedisCache) SetHotData(ctx context.Context, category, key string, value interface{}) error {
	fullKey := fmt.Sprintf("hot:%s:%s", category, key)
	return r.Set(ctx, fullKey, value, 1*time.Hour) // 热点数据缓存1小时
}

// GetHotData 获取热点数据
func (r *RedisCache) GetHotData(ctx context.Context, category, key string, dest interface{}) error {
	fullKey := fmt.Sprintf("hot:%s:%s", category, key)
	return r.Get(ctx, fullKey, dest)
}

// ==================== 限流计数器 ====================

// IncrementCounter 增加计数器
func (r *RedisCache) IncrementCounter(ctx context.Context, key string, window time.Duration) (int64, error) {
	fullKey := r.buildKey(fmt.Sprintf("counter:%s", key))

	pipe := r.client.Pipeline()
	incr := pipe.Incr(ctx, fullKey)
	pipe.Expire(ctx, fullKey, window)

	if _, err := pipe.Exec(ctx); err != nil {
		return 0, err
	}

	return incr.Val(), nil
}

// GetCounter 获取计数器值
func (r *RedisCache) GetCounter(ctx context.Context, key string) (int64, error) {
	fullKey := r.buildKey(fmt.Sprintf("counter:%s", key))
	val, err := r.client.Get(ctx, fullKey).Int64()
	if err == redis.Nil {
		return 0, nil
	}
	return val, err
}

// ==================== 分布式锁 ====================

// AcquireLock 获取分布式锁
func (r *RedisCache) AcquireLock(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	fullKey := r.buildKey(fmt.Sprintf("lock:%s", key))
	ok, err := r.client.SetNX(ctx, fullKey, "1", ttl).Result()
	if err != nil {
		return false, err
	}
	return ok, nil
}

// ReleaseLock 释放分布式锁
func (r *RedisCache) ReleaseLock(ctx context.Context, key string) error {
	fullKey := r.buildKey(fmt.Sprintf("lock:%s", key))
	return r.client.Del(ctx, fullKey).Err()
}

// ==================== 批量操作 ====================

// MGet 批量获取
func (r *RedisCache) MGet(ctx context.Context, keys []string) (map[string]interface{}, error) {
	fullKeys := make([]string, len(keys))
	for i, key := range keys {
		fullKeys[i] = r.buildKey(key)
	}

	values, err := r.client.MGet(ctx, fullKeys...).Result()
	if err != nil {
		return nil, err
	}

	result := make(map[string]interface{})
	for i, value := range values {
		if value != nil {
			var data interface{}
			if err := json.Unmarshal([]byte(value.(string)), &data); err == nil {
				result[keys[i]] = data
			}
		}
	}

	return result, nil
}

// MSet 批量设置
func (r *RedisCache) MSet(ctx context.Context, items map[string]interface{}, ttl time.Duration) error {
	pipe := r.client.Pipeline()

	for key, value := range items {
		data, err := json.Marshal(value)
		if err != nil {
			return fmt.Errorf("failed to marshal value for key %s: %w", key, err)
		}
		fullKey := r.buildKey(key)
		pipe.Set(ctx, fullKey, data, ttl)
	}

	_, err := pipe.Exec(ctx)
	return err
}

// ==================== 缓存统计 ====================

// CacheStats 缓存统计
type CacheStats struct {
	TotalKeys        int64   `json:"total_keys"`
	UsedMemory       string  `json:"used_memory"`
	HitRate          float64 `json:"hit_rate"`
	EvictedKeys      int64   `json:"evicted_keys"`
	ConnectedClients int64   `json:"connected_clients"`
}

// GetStats 获取缓存统计
func (r *RedisCache) GetStats(ctx context.Context) (*CacheStats, error) {
	info, err := r.client.Info(ctx, "stats", "memory", "clients").Result()
	if err != nil {
		return nil, err
	}

	// 解析info字符串（简化实现）
	stats := &CacheStats{}
	// TODO: 解析Redis INFO输出

	return stats, nil
}

// ==================== 清理操作 ====================

// FlushPattern 清理匹配模式的键
func (r *RedisCache) FlushPattern(ctx context.Context, pattern string) error {
	fullPattern := r.buildKey(pattern)

	// 使用SCAN避免阻塞
	iter := r.client.Scan(ctx, 0, fullPattern, 100).Iterator()
	var keys []string

	for iter.Next(ctx) {
		keys = append(keys, iter.Val())

		// 批量删除
		if len(keys) >= 100 {
			if err := r.client.Del(ctx, keys...).Err(); err != nil {
				return err
			}
			keys = keys[:0]
		}
	}

	// 删除剩余的键
	if len(keys) > 0 {
		if err := r.client.Del(ctx, keys...).Err(); err != nil {
			return err
		}
	}

	return iter.Err()
}

// Close 关闭连接
func (r *RedisCache) Close() error {
	return r.client.Close()
}

// ==================== 辅助方法 ====================

func (r *RedisCache) buildKey(key string) string {
	if r.prefix != "" {
		return fmt.Sprintf("%s:%s", r.prefix, key)
	}
	return key
}

// ==================== 错误定义 ====================

var (
	ErrCacheMiss = fmt.Errorf("cache miss")
	ErrCacheFull = fmt.Errorf("cache full")
)
