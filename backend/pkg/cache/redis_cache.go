package cache

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/sirupsen/logrus"
)

// Cache 缓存接口
type Cache interface {
	Get(ctx context.Context, key string) (string, error)
	Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error
	Delete(ctx context.Context, key string) error
	Exists(ctx context.Context, key string) (bool, error)
	Expire(ctx context.Context, key string, expiration time.Duration) error

	// JSON操作
	GetJSON(ctx context.Context, key string, dest interface{}) error
	SetJSON(ctx context.Context, key string, value interface{}, expiration time.Duration) error

	// Hash操作
	HGet(ctx context.Context, key, field string) (string, error)
	HSet(ctx context.Context, key, field string, value interface{}) error
	HGetAll(ctx context.Context, key string) (map[string]string, error)
	HDel(ctx context.Context, key string, fields ...string) error

	// List操作
	LPush(ctx context.Context, key string, values ...interface{}) error
	RPush(ctx context.Context, key string, values ...interface{}) error
	LPop(ctx context.Context, key string) (string, error)
	RPop(ctx context.Context, key string) (string, error)
	LRange(ctx context.Context, key string, start, stop int64) ([]string, error)
	LLen(ctx context.Context, key string) (int64, error)

	// Set操作
	SAdd(ctx context.Context, key string, members ...interface{}) error
	SRem(ctx context.Context, key string, members ...interface{}) error
	SMembers(ctx context.Context, key string) ([]string, error)
	SIsMember(ctx context.Context, key string, member interface{}) (bool, error)

	// 批量操作
	MGet(ctx context.Context, keys ...string) ([]interface{}, error)
	MSet(ctx context.Context, pairs ...interface{}) error

	// 原子操作
	Incr(ctx context.Context, key string) (int64, error)
	Decr(ctx context.Context, key string) (int64, error)
	IncrBy(ctx context.Context, key string, value int64) (int64, error)

	// 锁操作
	Lock(ctx context.Context, key string, expiration time.Duration) (bool, error)
	Unlock(ctx context.Context, key string) error

	// 管道操作
	Pipeline() Pipeline

	// 清理
	FlushDB(ctx context.Context) error
	Close() error
}

// Pipeline 管道接口
type Pipeline interface {
	Set(key string, value interface{}, expiration time.Duration) Pipeline
	Get(key string) Pipeline
	Del(keys ...string) Pipeline
	Exec(ctx context.Context) ([]interface{}, error)
}

// RedisCache Redis缓存实现
type RedisCache struct {
	client *redis.Client
	prefix string
}

// NewRedisCache 创建Redis缓存
func NewRedisCache(client *redis.Client, prefix string) Cache {
	return &RedisCache{
		client: client,
		prefix: prefix,
	}
}

// buildKey 构建键名
func (c *RedisCache) buildKey(key string) string {
	if c.prefix == "" {
		return key
	}
	return fmt.Sprintf("%s:%s", c.prefix, key)
}

// Get 获取值
func (c *RedisCache) Get(ctx context.Context, key string) (string, error) {
	result, err := c.client.Get(ctx, c.buildKey(key)).Result()
	if err != nil {
		if err == redis.Nil {
			return "", ErrCacheMiss
		}
		return "", fmt.Errorf("failed to get cache key %s: %v", key, err)
	}
	return result, nil
}

// Set 设置值
func (c *RedisCache) Set(ctx context.Context, key string, value interface{}, expiration time.Duration) error {
	err := c.client.Set(ctx, c.buildKey(key), value, expiration).Err()
	if err != nil {
		return fmt.Errorf("failed to set cache key %s: %v", key, err)
	}
	return nil
}

// Delete 删除键
func (c *RedisCache) Delete(ctx context.Context, key string) error {
	err := c.client.Del(ctx, c.buildKey(key)).Err()
	if err != nil {
		return fmt.Errorf("failed to delete cache key %s: %v", key, err)
	}
	return nil
}

// Exists 检查键是否存在
func (c *RedisCache) Exists(ctx context.Context, key string) (bool, error) {
	result, err := c.client.Exists(ctx, c.buildKey(key)).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check cache key existence %s: %v", key, err)
	}
	return result > 0, nil
}

// Expire 设置过期时间
func (c *RedisCache) Expire(ctx context.Context, key string, expiration time.Duration) error {
	err := c.client.Expire(ctx, c.buildKey(key), expiration).Err()
	if err != nil {
		return fmt.Errorf("failed to set expiration for cache key %s: %v", key, err)
	}
	return nil
}

// GetJSON 获取JSON值
func (c *RedisCache) GetJSON(ctx context.Context, key string, dest interface{}) error {
	value, err := c.Get(ctx, key)
	if err != nil {
		return err
	}

	if err := json.Unmarshal([]byte(value), dest); err != nil {
		return fmt.Errorf("failed to unmarshal JSON for key %s: %v", key, err)
	}

	return nil
}

// SetJSON 设置JSON值
func (c *RedisCache) SetJSON(ctx context.Context, key string, value interface{}, expiration time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal JSON for key %s: %v", key, err)
	}

	return c.Set(ctx, key, string(data), expiration)
}

// HGet 获取Hash字段
func (c *RedisCache) HGet(ctx context.Context, key, field string) (string, error) {
	result, err := c.client.HGet(ctx, c.buildKey(key), field).Result()
	if err != nil {
		if err == redis.Nil {
			return "", ErrCacheMiss
		}
		return "", fmt.Errorf("failed to get hash field %s:%s: %v", key, field, err)
	}
	return result, nil
}

// HSet 设置Hash字段
func (c *RedisCache) HSet(ctx context.Context, key, field string, value interface{}) error {
	err := c.client.HSet(ctx, c.buildKey(key), field, value).Err()
	if err != nil {
		return fmt.Errorf("failed to set hash field %s:%s: %v", key, field, err)
	}
	return nil
}

// HGetAll 获取所有Hash字段
func (c *RedisCache) HGetAll(ctx context.Context, key string) (map[string]string, error) {
	result, err := c.client.HGetAll(ctx, c.buildKey(key)).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get all hash fields for key %s: %v", key, err)
	}
	return result, nil
}

// HDel 删除Hash字段
func (c *RedisCache) HDel(ctx context.Context, key string, fields ...string) error {
	err := c.client.HDel(ctx, c.buildKey(key), fields...).Err()
	if err != nil {
		return fmt.Errorf("failed to delete hash fields for key %s: %v", key, err)
	}
	return nil
}

// LPush 从左侧推入列表
func (c *RedisCache) LPush(ctx context.Context, key string, values ...interface{}) error {
	err := c.client.LPush(ctx, c.buildKey(key), values...).Err()
	if err != nil {
		return fmt.Errorf("failed to lpush to key %s: %v", key, err)
	}
	return nil
}

// RPush 从右侧推入列表
func (c *RedisCache) RPush(ctx context.Context, key string, values ...interface{}) error {
	err := c.client.RPush(ctx, c.buildKey(key), values...).Err()
	if err != nil {
		return fmt.Errorf("failed to rpush to key %s: %v", key, err)
	}
	return nil
}

// LPop 从左侧弹出列表元素
func (c *RedisCache) LPop(ctx context.Context, key string) (string, error) {
	result, err := c.client.LPop(ctx, c.buildKey(key)).Result()
	if err != nil {
		if err == redis.Nil {
			return "", ErrCacheMiss
		}
		return "", fmt.Errorf("failed to lpop from key %s: %v", key, err)
	}
	return result, nil
}

// RPop 从右侧弹出列表元素
func (c *RedisCache) RPop(ctx context.Context, key string) (string, error) {
	result, err := c.client.RPop(ctx, c.buildKey(key)).Result()
	if err != nil {
		if err == redis.Nil {
			return "", ErrCacheMiss
		}
		return "", fmt.Errorf("failed to rpop from key %s: %v", key, err)
	}
	return result, nil
}

// LRange 获取列表范围
func (c *RedisCache) LRange(ctx context.Context, key string, start, stop int64) ([]string, error) {
	result, err := c.client.LRange(ctx, c.buildKey(key), start, stop).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to lrange for key %s: %v", key, err)
	}
	return result, nil
}

// LLen 获取列表长度
func (c *RedisCache) LLen(ctx context.Context, key string) (int64, error) {
	result, err := c.client.LLen(ctx, c.buildKey(key)).Result()
	if err != nil {
		return 0, fmt.Errorf("failed to get list length for key %s: %v", key, err)
	}
	return result, nil
}

// SAdd 添加集合成员
func (c *RedisCache) SAdd(ctx context.Context, key string, members ...interface{}) error {
	err := c.client.SAdd(ctx, c.buildKey(key), members...).Err()
	if err != nil {
		return fmt.Errorf("failed to sadd to key %s: %v", key, err)
	}
	return nil
}

// SRem 删除集合成员
func (c *RedisCache) SRem(ctx context.Context, key string, members ...interface{}) error {
	err := c.client.SRem(ctx, c.buildKey(key), members...).Err()
	if err != nil {
		return fmt.Errorf("failed to srem from key %s: %v", key, err)
	}
	return nil
}

// SMembers 获取所有集合成员
func (c *RedisCache) SMembers(ctx context.Context, key string) ([]string, error) {
	result, err := c.client.SMembers(ctx, c.buildKey(key)).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get set members for key %s: %v", key, err)
	}
	return result, nil
}

// SIsMember 检查是否为集合成员
func (c *RedisCache) SIsMember(ctx context.Context, key string, member interface{}) (bool, error) {
	result, err := c.client.SIsMember(ctx, c.buildKey(key), member).Result()
	if err != nil {
		return false, fmt.Errorf("failed to check set membership for key %s: %v", key, err)
	}
	return result, nil
}

// MGet 批量获取
func (c *RedisCache) MGet(ctx context.Context, keys ...string) ([]interface{}, error) {
	buildKeys := make([]string, len(keys))
	for i, key := range keys {
		buildKeys[i] = c.buildKey(key)
	}

	result, err := c.client.MGet(ctx, buildKeys...).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to mget keys: %v", err)
	}
	return result, nil
}

// MSet 批量设置
func (c *RedisCache) MSet(ctx context.Context, pairs ...interface{}) error {
	if len(pairs)%2 != 0 {
		return fmt.Errorf("mset requires an even number of arguments")
	}

	// 构建带前缀的键值对
	buildPairs := make([]interface{}, len(pairs))
	for i := 0; i < len(pairs); i += 2 {
		key, ok := pairs[i].(string)
		if !ok {
			return fmt.Errorf("mset key must be string")
		}
		buildPairs[i] = c.buildKey(key)
		buildPairs[i+1] = pairs[i+1]
	}

	err := c.client.MSet(ctx, buildPairs...).Err()
	if err != nil {
		return fmt.Errorf("failed to mset: %v", err)
	}
	return nil
}

// Incr 递增
func (c *RedisCache) Incr(ctx context.Context, key string) (int64, error) {
	result, err := c.client.Incr(ctx, c.buildKey(key)).Result()
	if err != nil {
		return 0, fmt.Errorf("failed to incr key %s: %v", key, err)
	}
	return result, nil
}

// Decr 递减
func (c *RedisCache) Decr(ctx context.Context, key string) (int64, error) {
	result, err := c.client.Decr(ctx, c.buildKey(key)).Result()
	if err != nil {
		return 0, fmt.Errorf("failed to decr key %s: %v", key, err)
	}
	return result, nil
}

// IncrBy 按值递增
func (c *RedisCache) IncrBy(ctx context.Context, key string, value int64) (int64, error) {
	result, err := c.client.IncrBy(ctx, c.buildKey(key), value).Result()
	if err != nil {
		return 0, fmt.Errorf("failed to incrby key %s: %v", key, err)
	}
	return result, nil
}

// Lock 分布式锁
func (c *RedisCache) Lock(ctx context.Context, key string, expiration time.Duration) (bool, error) {
	lockKey := c.buildKey(fmt.Sprintf("lock:%s", key))
	result, err := c.client.SetNX(ctx, lockKey, "locked", expiration).Result()
	if err != nil {
		return false, fmt.Errorf("failed to acquire lock %s: %v", key, err)
	}
	return result, nil
}

// Unlock 释放分布式锁
func (c *RedisCache) Unlock(ctx context.Context, key string) error {
	lockKey := c.buildKey(fmt.Sprintf("lock:%s", key))
	err := c.client.Del(ctx, lockKey).Err()
	if err != nil {
		return fmt.Errorf("failed to release lock %s: %v", key, err)
	}
	return nil
}

// Pipeline 创建管道
func (c *RedisCache) Pipeline() Pipeline {
	return &RedisPipeline{
		pipe:   c.client.Pipeline(),
		prefix: c.prefix,
	}
}

// FlushDB 清空数据库
func (c *RedisCache) FlushDB(ctx context.Context) error {
	err := c.client.FlushDB(ctx).Err()
	if err != nil {
		return fmt.Errorf("failed to flush database: %v", err)
	}
	logrus.Warn("Redis database flushed")
	return nil
}

// Close 关闭连接
func (c *RedisCache) Close() error {
	return c.client.Close()
}

// RedisPipeline Redis管道实现
type RedisPipeline struct {
	pipe   redis.Pipeliner
	prefix string
}

// buildKey 构建键名
func (p *RedisPipeline) buildKey(key string) string {
	if p.prefix == "" {
		return key
	}
	return fmt.Sprintf("%s:%s", p.prefix, key)
}

// Set 设置值
func (p *RedisPipeline) Set(key string, value interface{}, expiration time.Duration) Pipeline {
	p.pipe.Set(context.Background(), p.buildKey(key), value, expiration)
	return p
}

// Get 获取值
func (p *RedisPipeline) Get(key string) Pipeline {
	p.pipe.Get(context.Background(), p.buildKey(key))
	return p
}

// Del 删除键
func (p *RedisPipeline) Del(keys ...string) Pipeline {
	buildKeys := make([]string, len(keys))
	for i, key := range keys {
		buildKeys[i] = p.buildKey(key)
	}
	p.pipe.Del(context.Background(), buildKeys...)
	return p
}

// Exec 执行管道
func (p *RedisPipeline) Exec(ctx context.Context) ([]interface{}, error) {
	cmds, err := p.pipe.Exec(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to execute pipeline: %v", err)
	}

	results := make([]interface{}, len(cmds))
	for i, cmd := range cmds {
		results[i] = cmd
	}

	return results, nil
}

// 错误定义
var (
	ErrCacheMiss = fmt.Errorf("cache miss")
)
