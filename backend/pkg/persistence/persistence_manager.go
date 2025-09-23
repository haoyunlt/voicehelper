package persistence

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/sirupsen/logrus"

	"voicehelper/backend/internal/repository"
	"voicehelper/backend/pkg/cache"
	"voicehelper/backend/pkg/database"
	"voicehelper/backend/pkg/validation"
)

// PersistenceManager 持久化管理器
type PersistenceManager struct {
	// 数据库连接
	db *sql.DB

	// 缓存
	cacheManager *cache.CacheManager

	// 仓库
	tenantRepo       repository.TenantRepository
	userRepo         repository.UserRepository
	conversationRepo repository.ConversationRepository
	documentRepo     repository.DocumentRepository
	voiceSessionRepo repository.VoiceSessionRepository

	// 迁移管理器
	migrationManager *database.MigrationManager

	// 配置
	config *PersistenceConfig

	// 状态
	initialized bool
	mutex       sync.RWMutex
}

// PersistenceConfig 持久化配置
type PersistenceConfig struct {
	DatabaseURL       string        `json:"database_url"`
	RedisURL          string        `json:"redis_url"`
	CacheEnabled      bool          `json:"cache_enabled"`
	CacheDefaultTTL   time.Duration `json:"cache_default_ttl"`
	MigrationEnabled  bool          `json:"migration_enabled"`
	ValidationEnabled bool          `json:"validation_enabled"`
}

// DefaultPersistenceConfig 默认持久化配置
func DefaultPersistenceConfig() *PersistenceConfig {
	return &PersistenceConfig{
		CacheEnabled:      true,
		CacheDefaultTTL:   30 * time.Minute,
		MigrationEnabled:  true,
		ValidationEnabled: true,
	}
}

// NewPersistenceManager 创建持久化管理器
func NewPersistenceManager(db *sql.DB, redisClient *redis.Client, config *PersistenceConfig) *PersistenceManager {
	if config == nil {
		config = DefaultPersistenceConfig()
	}

	pm := &PersistenceManager{
		db:     db,
		config: config,
	}

	// 初始化缓存管理器
	if config.CacheEnabled && redisClient != nil {
		pm.cacheManager = cache.NewCacheManager(config.CacheDefaultTTL)

		// 注册各种缓存
		for name := range cache.DefaultCacheConfigs {
			redisCache := cache.NewRedisCache(redisClient, fmt.Sprintf("voicehelper:%s", name))
			pm.cacheManager.RegisterCache(name, redisCache)
		}
	}

	// 初始化仓库
	pm.initializeRepositories()

	// 初始化迁移管理器
	if config.MigrationEnabled {
		pm.migrationManager = database.NewMigrationManager(db)
		pm.migrationManager.RegisterCoreMigrations()
	}

	return pm
}

// initializeRepositories 初始化仓库
func (pm *PersistenceManager) initializeRepositories() {
	pm.tenantRepo = repository.NewPostgresTenantRepository(pm.db)
	pm.userRepo = repository.NewPostgresUserRepository(pm.db)
	pm.conversationRepo = repository.NewPostgresConversationRepository(pm.db)
	pm.documentRepo = repository.NewPostgresDocumentRepository(pm.db)
	pm.voiceSessionRepo = repository.NewPostgresVoiceSessionRepository(pm.db)
}

// Initialize 初始化持久化系统
func (pm *PersistenceManager) Initialize(ctx context.Context) error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	if pm.initialized {
		return nil
	}

	logrus.Info("Initializing persistence manager")

	// 运行数据库迁移
	if pm.config.MigrationEnabled && pm.migrationManager != nil {
		logrus.Info("Running database migrations")
		if err := pm.migrationManager.Migrate(); err != nil {
			return fmt.Errorf("failed to run migrations: %v", err)
		}
	}

	// 验证数据库连接
	if err := pm.db.PingContext(ctx); err != nil {
		return fmt.Errorf("failed to ping database: %v", err)
	}

	pm.initialized = true
	logrus.Info("Persistence manager initialized successfully")

	return nil
}

// GetTenantRepository 获取租户仓库
func (pm *PersistenceManager) GetTenantRepository() repository.TenantRepository {
	return pm.tenantRepo
}

// GetUserRepository 获取用户仓库
func (pm *PersistenceManager) GetUserRepository() repository.UserRepository {
	return pm.userRepo
}

// GetConversationRepository 获取会话仓库
func (pm *PersistenceManager) GetConversationRepository() repository.ConversationRepository {
	return pm.conversationRepo
}

// GetDocumentRepository 获取文档仓库
func (pm *PersistenceManager) GetDocumentRepository() repository.DocumentRepository {
	return pm.documentRepo
}

// GetVoiceSessionRepository 获取语音会话仓库
func (pm *PersistenceManager) GetVoiceSessionRepository() repository.VoiceSessionRepository {
	return pm.voiceSessionRepo
}

// GetCacheManager 获取缓存管理器
func (pm *PersistenceManager) GetCacheManager() *cache.CacheManager {
	return pm.cacheManager
}

// GetMigrationManager 获取迁移管理器
func (pm *PersistenceManager) GetMigrationManager() *database.MigrationManager {
	return pm.migrationManager
}

// 带缓存的仓库操作

// GetTenantWithCache 带缓存获取租户
func (pm *PersistenceManager) GetTenantWithCache(ctx context.Context, tenantID string) (*repository.Tenant, error) {
	if pm.config.ValidationEnabled {
		if errs := validation.ValidateTenantID(tenantID); errs.HasErrors() {
			return nil, fmt.Errorf("validation failed: %v", errs)
		}
	}

	if pm.cacheManager == nil {
		return pm.tenantRepo.GetByTenantID(ctx, tenantID)
	}

	cacheKey := cache.TenantCacheKey(tenantID)
	strategy := cache.NewCacheStrategy(pm.cacheManager)

	result, err := strategy.GetOrSet(ctx, "tenants", cacheKey, cache.LongTTL, func() (interface{}, error) {
		return pm.tenantRepo.GetByTenantID(ctx, tenantID)
	})

	if err != nil {
		return nil, err
	}

	tenant, ok := result.(*repository.Tenant)
	if !ok {
		return nil, fmt.Errorf("invalid tenant data in cache")
	}

	return tenant, nil
}

// CreateTenantWithCache 带缓存创建租户
func (pm *PersistenceManager) CreateTenantWithCache(ctx context.Context, tenant *repository.Tenant) error {
	if pm.config.ValidationEnabled {
		if errs := validation.ValidateTenantID(tenant.TenantID); errs.HasErrors() {
			return fmt.Errorf("validation failed: %v", errs)
		}
		if errs := validation.ValidatePlan(tenant.Plan); errs.HasErrors() {
			return fmt.Errorf("validation failed: %v", errs)
		}
	}

	if err := pm.tenantRepo.Create(ctx, tenant); err != nil {
		return err
	}

	// 清除相关缓存
	if pm.cacheManager != nil {
		cacheKey := cache.TenantCacheKey(tenant.TenantID)
		if cache, err := pm.cacheManager.GetCache("tenants"); err == nil {
			cache.Delete(ctx, cacheKey)
		}
	}

	return nil
}

// GetUserWithCache 带缓存获取用户
func (pm *PersistenceManager) GetUserWithCache(ctx context.Context, userID string) (*repository.User, error) {
	if pm.config.ValidationEnabled {
		if errs := validation.ValidateUserID(userID); errs.HasErrors() {
			return nil, fmt.Errorf("validation failed: %v", errs)
		}
	}

	if pm.cacheManager == nil {
		return pm.userRepo.GetByID(ctx, userID)
	}

	cacheKey := cache.UserCacheKey(userID)
	strategy := cache.NewCacheStrategy(pm.cacheManager)

	result, err := strategy.GetOrSet(ctx, "users", cacheKey, cache.MediumTTL, func() (interface{}, error) {
		return pm.userRepo.GetByID(ctx, userID)
	})

	if err != nil {
		return nil, err
	}

	user, ok := result.(*repository.User)
	if !ok {
		return nil, fmt.Errorf("invalid user data in cache")
	}

	return user, nil
}

// GetConversationWithCache 带缓存获取会话
func (pm *PersistenceManager) GetConversationWithCache(ctx context.Context, conversationID string) (*repository.Conversation, error) {
	if pm.cacheManager == nil {
		return pm.conversationRepo.Get(ctx, conversationID)
	}

	cacheKey := cache.ConversationCacheKey(conversationID)
	strategy := cache.NewCacheStrategy(pm.cacheManager)

	result, err := strategy.GetOrSet(ctx, "conversations", cacheKey, cache.ShortTTL, func() (interface{}, error) {
		return pm.conversationRepo.Get(ctx, conversationID)
	})

	if err != nil {
		return nil, err
	}

	conversation, ok := result.(*repository.Conversation)
	if !ok {
		return nil, fmt.Errorf("invalid conversation data in cache")
	}

	return conversation, nil
}

// CreateMessageWithCache 带缓存创建消息
func (pm *PersistenceManager) CreateMessageWithCache(ctx context.Context, message *repository.Message) error {
	if pm.config.ValidationEnabled {
		if errs := validation.ValidateMessageContent(message.Content); errs.HasErrors() {
			return fmt.Errorf("validation failed: %v", errs)
		}
		if errs := validation.ValidateMessageRole(message.Role); errs.HasErrors() {
			return fmt.Errorf("validation failed: %v", errs)
		}
	}

	if err := pm.conversationRepo.AddMessage(ctx, message); err != nil {
		return err
	}

	// 清除相关缓存
	if pm.cacheManager != nil {
		// 清除会话缓存
		conversationKey := cache.ConversationCacheKey(message.ConversationID)
		if cache, err := pm.cacheManager.GetCache("conversations"); err == nil {
			cache.Delete(ctx, conversationKey)
		}

		// 清除消息列表缓存（可能有多页）
		if cacheInstance, err := pm.cacheManager.GetCache("messages"); err == nil {
			for page := 1; page <= 10; page++ { // 清除前10页缓存
				messageKey := cache.MessagesCacheKey(message.ConversationID, page)
				cacheInstance.Delete(ctx, messageKey)
			}
		}
	}

	return nil
}

// GetVoiceSessionWithCache 带缓存获取语音会话
func (pm *PersistenceManager) GetVoiceSessionWithCache(ctx context.Context, sessionID string) (*repository.VoiceSession, error) {
	if pm.cacheManager == nil {
		return pm.voiceSessionRepo.GetByID(ctx, sessionID)
	}

	cacheKey := cache.VoiceSessionCacheKey(sessionID)
	strategy := cache.NewCacheStrategy(pm.cacheManager)

	result, err := strategy.GetOrSet(ctx, "voice_sessions", cacheKey, cache.ShortTTL, func() (interface{}, error) {
		return pm.voiceSessionRepo.GetByID(ctx, sessionID)
	})

	if err != nil {
		return nil, err
	}

	session, ok := result.(*repository.VoiceSession)
	if !ok {
		return nil, fmt.Errorf("invalid voice session data in cache")
	}

	return session, nil
}

// Health 健康检查
func (pm *PersistenceManager) Health(ctx context.Context) error {
	// 检查数据库连接
	if err := pm.db.PingContext(ctx); err != nil {
		return fmt.Errorf("database health check failed: %v", err)
	}

	// 检查缓存连接（如果启用）
	if pm.cacheManager != nil {
		if cache, err := pm.cacheManager.GetCache("users"); err == nil {
			if err := cache.Set(ctx, "health_check", "ok", time.Minute); err != nil {
				return fmt.Errorf("cache health check failed: %v", err)
			}
		}
	}

	return nil
}

// GetStats 获取统计信息
func (pm *PersistenceManager) GetStats(ctx context.Context) map[string]interface{} {
	stats := make(map[string]interface{})

	// 数据库统计
	dbStats := pm.db.Stats()
	stats["database"] = map[string]interface{}{
		"open_connections": dbStats.OpenConnections,
		"in_use":           dbStats.InUse,
		"idle":             dbStats.Idle,
		"max_open":         dbStats.MaxOpenConnections,
	}

	// 缓存统计
	if pm.cacheManager != nil {
		stats["cache"] = pm.cacheManager.GetStats()
	}

	// 配置信息
	stats["config"] = map[string]interface{}{
		"cache_enabled":      pm.config.CacheEnabled,
		"migration_enabled":  pm.config.MigrationEnabled,
		"validation_enabled": pm.config.ValidationEnabled,
		"initialized":        pm.initialized,
	}

	return stats
}

// Close 关闭持久化管理器
func (pm *PersistenceManager) Close() error {
	pm.mutex.Lock()
	defer pm.mutex.Unlock()

	logrus.Info("Closing persistence manager")

	var errors []error

	// 关闭数据库连接
	if pm.db != nil {
		if err := pm.db.Close(); err != nil {
			errors = append(errors, fmt.Errorf("failed to close database: %v", err))
		}
	}

	// 关闭缓存连接
	if pm.cacheManager != nil {
		for name := range cache.DefaultCacheConfigs {
			if cache, err := pm.cacheManager.GetCache(name); err == nil {
				if err := cache.Close(); err != nil {
					errors = append(errors, fmt.Errorf("failed to close cache %s: %v", name, err))
				}
			}
		}
	}

	pm.initialized = false

	if len(errors) > 0 {
		return fmt.Errorf("errors during close: %v", errors)
	}

	logrus.Info("Persistence manager closed successfully")
	return nil
}

// 全局持久化管理器
var (
	globalPersistenceManager *PersistenceManager
	persistenceManagerOnce   sync.Once
)

// GetGlobalPersistenceManager 获取全局持久化管理器
func GetGlobalPersistenceManager() *PersistenceManager {
	return globalPersistenceManager
}

// InitializeGlobalPersistenceManager 初始化全局持久化管理器
func InitializeGlobalPersistenceManager(db *sql.DB, redisClient *redis.Client, config *PersistenceConfig) *PersistenceManager {
	persistenceManagerOnce.Do(func() {
		globalPersistenceManager = NewPersistenceManager(db, redisClient, config)
	})
	return globalPersistenceManager
}
