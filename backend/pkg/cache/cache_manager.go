package cache

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// CacheManager 缓存管理器
type CacheManager struct {
	caches map[string]Cache
	mutex  sync.RWMutex

	// 默认配置
	defaultTTL time.Duration

	// 统计
	hits   int64
	misses int64
	errors int64
}

// NewCacheManager 创建缓存管理器
func NewCacheManager(defaultTTL time.Duration) *CacheManager {
	return &CacheManager{
		caches:     make(map[string]Cache),
		defaultTTL: defaultTTL,
	}
}

// RegisterCache 注册缓存
func (cm *CacheManager) RegisterCache(name string, cache Cache) {
	cm.mutex.Lock()
	defer cm.mutex.Unlock()

	cm.caches[name] = cache
	logrus.WithField("cache_name", name).Info("Cache registered")
}

// GetCache 获取缓存
func (cm *CacheManager) GetCache(name string) (Cache, error) {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()

	cache, exists := cm.caches[name]
	if !exists {
		return nil, fmt.Errorf("cache %s not found", name)
	}

	return cache, nil
}

// GetStats 获取统计信息
func (cm *CacheManager) GetStats() map[string]interface{} {
	cm.mutex.RLock()
	defer cm.mutex.RUnlock()

	total := cm.hits + cm.misses
	hitRate := float64(0)
	if total > 0 {
		hitRate = float64(cm.hits) / float64(total)
	}

	return map[string]interface{}{
		"hits":     cm.hits,
		"misses":   cm.misses,
		"errors":   cm.errors,
		"hit_rate": hitRate,
		"total":    total,
		"caches":   len(cm.caches),
	}
}

// CacheStrategy 缓存策略
type CacheStrategy struct {
	manager *CacheManager
}

// NewCacheStrategy 创建缓存策略
func NewCacheStrategy(manager *CacheManager) *CacheStrategy {
	return &CacheStrategy{manager: manager}
}

// GetOrSet 获取或设置缓存
func (cs *CacheStrategy) GetOrSet(ctx context.Context, cacheName, key string, ttl time.Duration, fn func() (interface{}, error)) (interface{}, error) {
	cache, err := cs.manager.GetCache(cacheName)
	if err != nil {
		return nil, err
	}

	// 尝试从缓存获取
	var result interface{}
	err = cache.GetJSON(ctx, key, &result)
	if err == nil {
		cs.manager.hits++
		return result, nil
	}

	if err != ErrCacheMiss {
		cs.manager.errors++
		logrus.WithError(err).WithField("key", key).Error("Cache get error")
	} else {
		cs.manager.misses++
	}

	// 缓存未命中，执行函数获取数据
	result, err = fn()
	if err != nil {
		return nil, err
	}

	// 设置缓存
	if err := cache.SetJSON(ctx, key, result, ttl); err != nil {
		cs.manager.errors++
		logrus.WithError(err).WithField("key", key).Error("Cache set error")
	}

	return result, nil
}

// InvalidatePattern 按模式失效缓存
func (cs *CacheStrategy) InvalidatePattern(ctx context.Context, cacheName, pattern string) error {
	// 这里需要根据具体的缓存实现来处理模式匹配
	// Redis可以使用SCAN命令
	logrus.WithFields(logrus.Fields{
		"cache_name": cacheName,
		"pattern":    pattern,
	}).Info("Cache pattern invalidation requested")

	return nil
}

// 预定义的缓存键构建器

// UserCacheKey 用户缓存键
func UserCacheKey(userID string) string {
	return fmt.Sprintf("user:%s", userID)
}

// TenantCacheKey 租户缓存键
func TenantCacheKey(tenantID string) string {
	return fmt.Sprintf("tenant:%s", tenantID)
}

// ConversationCacheKey 会话缓存键
func ConversationCacheKey(conversationID string) string {
	return fmt.Sprintf("conversation:%s", conversationID)
}

// MessagesCacheKey 消息缓存键
func MessagesCacheKey(conversationID string, page int) string {
	return fmt.Sprintf("messages:%s:page:%d", conversationID, page)
}

// DocumentCacheKey 文档缓存键
func DocumentCacheKey(documentID string) string {
	return fmt.Sprintf("document:%s", documentID)
}

// VoiceSessionCacheKey 语音会话缓存键
func VoiceSessionCacheKey(sessionID string) string {
	return fmt.Sprintf("voice_session:%s", sessionID)
}

// APIKeyCacheKey API密钥缓存键
func APIKeyCacheKey(keyHash string) string {
	return fmt.Sprintf("api_key:%s", keyHash)
}

// SearchResultsCacheKey 搜索结果缓存键
func SearchResultsCacheKey(tenantID, query string, page int) string {
	return fmt.Sprintf("search:%s:%s:page:%d", tenantID, query, page)
}

// 缓存TTL常量
const (
	ShortTTL  = 5 * time.Minute  // 短期缓存
	MediumTTL = 30 * time.Minute // 中期缓存
	LongTTL   = 2 * time.Hour    // 长期缓存
	DayTTL    = 24 * time.Hour   // 日缓存
)

// CacheConfig 缓存配置
type CacheConfig struct {
	Name       string        `json:"name"`
	TTL        time.Duration `json:"ttl"`
	MaxSize    int64         `json:"max_size"`
	Enabled    bool          `json:"enabled"`
	Compressed bool          `json:"compressed"`
}

// DefaultCacheConfigs 默认缓存配置
var DefaultCacheConfigs = map[string]CacheConfig{
	"users": {
		Name:    "users",
		TTL:     MediumTTL,
		Enabled: true,
	},
	"tenants": {
		Name:    "tenants",
		TTL:     LongTTL,
		Enabled: true,
	},
	"conversations": {
		Name:    "conversations",
		TTL:     ShortTTL,
		Enabled: true,
	},
	"messages": {
		Name:    "messages",
		TTL:     MediumTTL,
		Enabled: true,
	},
	"documents": {
		Name:    "documents",
		TTL:     LongTTL,
		Enabled: true,
	},
	"voice_sessions": {
		Name:    "voice_sessions",
		TTL:     ShortTTL,
		Enabled: true,
	},
	"api_keys": {
		Name:    "api_keys",
		TTL:     LongTTL,
		Enabled: true,
	},
	"search_results": {
		Name:    "search_results",
		TTL:     ShortTTL,
		Enabled: true,
	},
}

// CacheWrapper 缓存包装器
type CacheWrapper struct {
	cache    Cache
	config   CacheConfig
	strategy *CacheStrategy
}

// NewCacheWrapper 创建缓存包装器
func NewCacheWrapper(cache Cache, config CacheConfig, strategy *CacheStrategy) *CacheWrapper {
	return &CacheWrapper{
		cache:    cache,
		config:   config,
		strategy: strategy,
	}
}

// Get 获取缓存
func (cw *CacheWrapper) Get(ctx context.Context, key string, dest interface{}) error {
	if !cw.config.Enabled {
		return ErrCacheMiss
	}

	return cw.cache.GetJSON(ctx, key, dest)
}

// Set 设置缓存
func (cw *CacheWrapper) Set(ctx context.Context, key string, value interface{}) error {
	if !cw.config.Enabled {
		return nil
	}

	return cw.cache.SetJSON(ctx, key, value, cw.config.TTL)
}

// Delete 删除缓存
func (cw *CacheWrapper) Delete(ctx context.Context, key string) error {
	if !cw.config.Enabled {
		return nil
	}

	return cw.cache.Delete(ctx, key)
}

// GetOrSet 获取或设置缓存
func (cw *CacheWrapper) GetOrSet(ctx context.Context, key string, fn func() (interface{}, error)) (interface{}, error) {
	if !cw.config.Enabled {
		return fn()
	}

	return cw.strategy.GetOrSet(ctx, cw.config.Name, key, cw.config.TTL, fn)
}

// 全局缓存管理器
var (
	globalCacheManager *CacheManager
	cacheManagerOnce   sync.Once
)

// GetGlobalCacheManager 获取全局缓存管理器
func GetGlobalCacheManager() *CacheManager {
	cacheManagerOnce.Do(func() {
		globalCacheManager = NewCacheManager(MediumTTL)
	})
	return globalCacheManager
}

// InitializeCacheManager 初始化缓存管理器
func InitializeCacheManager(defaultTTL time.Duration) *CacheManager {
	globalCacheManager = NewCacheManager(defaultTTL)
	return globalCacheManager
}
