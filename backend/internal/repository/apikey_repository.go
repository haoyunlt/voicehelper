package repository

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
	"voicehelper/backend/pkg/ratelimit"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	"golang.org/x/crypto/bcrypt"
)

// APIKey API密钥模型
type APIKey struct {
	ID          string                 `json:"id"`
	TenantID    string                 `json:"tenant_id"`
	Name        string                 `json:"name"`
	Key         string                 `json:"key"`         // API Key（前缀）
	SecretHash  string                 `json:"-"`           // Secret的哈希值
	Status      string                 `json:"status"`      // active, inactive, expired, revoked
	Permissions []string               `json:"permissions"` // 权限列表
	RateLimit   int                    `json:"rate_limit"`  // 速率限制（请求/小时）
	Metadata    map[string]interface{} `json:"metadata"`
	ExpiresAt   *time.Time             `json:"expires_at"`
	LastUsedAt  *time.Time             `json:"last_used_at"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// APIKeyCreateRequest 创建API Key请求
type APIKeyCreateRequest struct {
	Name        string                 `json:"name"`
	Permissions []string               `json:"permissions"`
	RateLimit   int                    `json:"rate_limit"`
	ExpiresIn   int                    `json:"expires_in"` // 过期时间（天）
	Metadata    map[string]interface{} `json:"metadata"`
}

// APIKeyCreateResponse 创建API Key响应
type APIKeyCreateResponse struct {
	ID        string     `json:"id"`
	Key       string     `json:"key"`
	Secret    string     `json:"secret"` // 仅在创建时返回一次
	ExpiresAt *time.Time `json:"expires_at"`
}

// APIKeyRepository API密钥仓库接口
type APIKeyRepository interface {
	Create(ctx context.Context, tenantID string, req *APIKeyCreateRequest) (*APIKeyCreateResponse, error)
	Validate(ctx context.Context, apiKey string) (*APIKey, error)
	ValidateWithSecret(ctx context.Context, apiKey, secret string) (*APIKey, error)
	Get(ctx context.Context, id string) (*APIKey, error)
	List(ctx context.Context, tenantID string, opts ListOptions) ([]*APIKey, int, error)
	Update(ctx context.Context, key *APIKey) error
	Revoke(ctx context.Context, id string) error
	UpdateLastUsed(ctx context.Context, id string) error
	CheckRateLimit(ctx context.Context, id string) (bool, error)
}

// PostgresAPIKeyRepository PostgreSQL实现
type PostgresAPIKeyRepository struct {
	db          *sql.DB
	rateLimiter *ratelimit.RateLimiter
}

// NewPostgresAPIKeyRepository 创建PostgreSQL API密钥仓库
func NewPostgresAPIKeyRepository(db *sql.DB) APIKeyRepository {
	return &PostgresAPIKeyRepository{db: db}
}

// NewPostgresAPIKeyRepositoryWithRateLimit 创建带速率限制的PostgreSQL API密钥仓库
func NewPostgresAPIKeyRepositoryWithRateLimit(db *sql.DB, rateLimiter *ratelimit.RateLimiter) APIKeyRepository {
	return &PostgresAPIKeyRepository{
		db:          db,
		rateLimiter: rateLimiter,
	}
}

// Create 创建API密钥
func (r *PostgresAPIKeyRepository) Create(ctx context.Context, tenantID string, req *APIKeyCreateRequest) (*APIKeyCreateResponse, error) {
	// 生成API Key和Secret
	apiKey, secret, err := generateAPIKeyPair()
	if err != nil {
		return nil, fmt.Errorf("failed to generate API key: %w", err)
	}

	// 哈希Secret
	secretHash, err := hashSecret(secret)
	if err != nil {
		return nil, fmt.Errorf("failed to hash secret: %w", err)
	}

	// 计算过期时间
	var expiresAt *time.Time
	if req.ExpiresIn > 0 {
		exp := time.Now().AddDate(0, 0, req.ExpiresIn)
		expiresAt = &exp
	}

	// 序列化权限和元数据
	permissionsJSON, _ := json.Marshal(req.Permissions)
	metadataJSON, _ := json.Marshal(req.Metadata)

	// 设置默认值
	if req.RateLimit == 0 {
		req.RateLimit = 1000 // 默认1000请求/小时
	}

	// 创建记录
	id := uuid.New().String()
	query := `
		INSERT INTO api_keys (
			id, tenant_id, name, key, secret_hash, status,
			permissions, rate_limit, metadata, expires_at,
			created_at, updated_at
		) VALUES (
			$1, $2, $3, $4, $5, 'active', $6, $7, $8, $9, NOW(), NOW()
		)
	`

	_, err = r.db.ExecContext(ctx, query,
		id, tenantID, req.Name, apiKey, secretHash,
		permissionsJSON, req.RateLimit, metadataJSON, expiresAt,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to create API key: %w", err)
	}

	return &APIKeyCreateResponse{
		ID:        id,
		Key:       apiKey,
		Secret:    secret,
		ExpiresAt: expiresAt,
	}, nil
}

// Validate 验证API密钥（仅Key）
func (r *PostgresAPIKeyRepository) Validate(ctx context.Context, apiKey string) (*APIKey, error) {
	query := `
		SELECT 
			id, tenant_id, name, key, secret_hash, status,
			permissions, rate_limit, metadata, expires_at,
			last_used_at, created_at, updated_at
		FROM api_keys
		WHERE key = $1 AND deleted_at IS NULL
	`

	key := &APIKey{}
	var permissionsJSON, metadataJSON []byte
	var expiresAt, lastUsedAt sql.NullTime

	err := r.db.QueryRowContext(ctx, query, apiKey).Scan(
		&key.ID, &key.TenantID, &key.Name, &key.Key, &key.SecretHash,
		&key.Status, &permissionsJSON, &key.RateLimit, &metadataJSON,
		&expiresAt, &lastUsedAt, &key.CreatedAt, &key.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("invalid API key")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to validate API key: %w", err)
	}

	// 解析JSON字段
	json.Unmarshal(permissionsJSON, &key.Permissions)
	json.Unmarshal(metadataJSON, &key.Metadata)

	if expiresAt.Valid {
		key.ExpiresAt = &expiresAt.Time
	}
	if lastUsedAt.Valid {
		key.LastUsedAt = &lastUsedAt.Time
	}

	// 检查状态
	if key.Status != "active" {
		return nil, fmt.Errorf("API key is %s", key.Status)
	}

	// 检查过期
	if key.ExpiresAt != nil && key.ExpiresAt.Before(time.Now()) {
		// 更新状态为过期
		r.updateStatus(ctx, key.ID, "expired")
		return nil, fmt.Errorf("API key has expired")
	}

	// 更新最后使用时间（异步）
	go r.UpdateLastUsed(context.Background(), key.ID)

	return key, nil
}

// ValidateWithSecret 验证API密钥（Key + Secret）
func (r *PostgresAPIKeyRepository) ValidateWithSecret(ctx context.Context, apiKey, secret string) (*APIKey, error) {
	// 先验证Key
	key, err := r.Validate(ctx, apiKey)
	if err != nil {
		return nil, err
	}

	// 验证Secret
	if !verifySecret(secret, key.SecretHash) {
		return nil, fmt.Errorf("invalid API secret")
	}

	return key, nil
}

// Get 获取API密钥详情
func (r *PostgresAPIKeyRepository) Get(ctx context.Context, id string) (*APIKey, error) {
	query := `
		SELECT 
			id, tenant_id, name, key, status,
			permissions, rate_limit, metadata, expires_at,
			last_used_at, created_at, updated_at
		FROM api_keys
		WHERE id = $1 AND deleted_at IS NULL
	`

	key := &APIKey{}
	var permissionsJSON, metadataJSON []byte
	var expiresAt, lastUsedAt sql.NullTime

	err := r.db.QueryRowContext(ctx, query, id).Scan(
		&key.ID, &key.TenantID, &key.Name, &key.Key,
		&key.Status, &permissionsJSON, &key.RateLimit, &metadataJSON,
		&expiresAt, &lastUsedAt, &key.CreatedAt, &key.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("API key not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get API key: %w", err)
	}

	// 解析JSON字段
	json.Unmarshal(permissionsJSON, &key.Permissions)
	json.Unmarshal(metadataJSON, &key.Metadata)

	if expiresAt.Valid {
		key.ExpiresAt = &expiresAt.Time
	}
	if lastUsedAt.Valid {
		key.LastUsedAt = &lastUsedAt.Time
	}

	return key, nil
}

// List 列出API密钥
func (r *PostgresAPIKeyRepository) List(ctx context.Context, tenantID string, opts ListOptions) ([]*APIKey, int, error) {
	// 设置默认值
	if opts.Limit == 0 {
		opts.Limit = 20
	}
	if opts.SortBy == "" {
		opts.SortBy = "created_at"
	}
	if opts.Order == "" {
		opts.Order = "desc"
	}

	// 计算总数
	countQuery := `
		SELECT COUNT(*) 
		FROM api_keys 
		WHERE tenant_id = $1 AND deleted_at IS NULL
	`
	var total int
	err := r.db.QueryRowContext(ctx, countQuery, tenantID).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count API keys: %w", err)
	}

	// 查询数据
	query := fmt.Sprintf(`
		SELECT 
			id, tenant_id, name, key, status,
			permissions, rate_limit, metadata, expires_at,
			last_used_at, created_at, updated_at
		FROM api_keys
		WHERE tenant_id = $1 AND deleted_at IS NULL
		ORDER BY %s %s
		LIMIT $2 OFFSET $3
	`, opts.SortBy, opts.Order)

	rows, err := r.db.QueryContext(ctx, query, tenantID, opts.Limit, opts.Offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list API keys: %w", err)
	}
	defer rows.Close()

	var keys []*APIKey
	for rows.Next() {
		key := &APIKey{}
		var permissionsJSON, metadataJSON []byte
		var expiresAt, lastUsedAt sql.NullTime

		err := rows.Scan(
			&key.ID, &key.TenantID, &key.Name, &key.Key,
			&key.Status, &permissionsJSON, &key.RateLimit, &metadataJSON,
			&expiresAt, &lastUsedAt, &key.CreatedAt, &key.UpdatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan API key: %w", err)
		}

		// 解析JSON字段
		json.Unmarshal(permissionsJSON, &key.Permissions)
		json.Unmarshal(metadataJSON, &key.Metadata)

		if expiresAt.Valid {
			key.ExpiresAt = &expiresAt.Time
		}
		if lastUsedAt.Valid {
			key.LastUsedAt = &lastUsedAt.Time
		}

		keys = append(keys, key)
	}

	return keys, total, nil
}

// Update 更新API密钥
func (r *PostgresAPIKeyRepository) Update(ctx context.Context, key *APIKey) error {
	permissionsJSON, _ := json.Marshal(key.Permissions)
	metadataJSON, _ := json.Marshal(key.Metadata)

	query := `
		UPDATE api_keys SET
			name = $2,
			status = $3,
			permissions = $4,
			rate_limit = $5,
			metadata = $6,
			expires_at = $7,
			updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query,
		key.ID, key.Name, key.Status, permissionsJSON,
		key.RateLimit, metadataJSON, key.ExpiresAt,
	)

	if err != nil {
		return fmt.Errorf("failed to update API key: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("API key not found")
	}

	return nil
}

// Revoke 吊销API密钥
func (r *PostgresAPIKeyRepository) Revoke(ctx context.Context, id string) error {
	return r.updateStatus(ctx, id, "revoked")
}

// UpdateLastUsed 更新最后使用时间
func (r *PostgresAPIKeyRepository) UpdateLastUsed(ctx context.Context, id string) error {
	query := `
		UPDATE api_keys 
		SET last_used_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	_, err := r.db.ExecContext(ctx, query, id)
	return err
}

// CheckRateLimit 检查速率限制
func (r *PostgresAPIKeyRepository) CheckRateLimit(ctx context.Context, id string) (bool, error) {
	// 获取API Key信息
	key, err := r.Get(ctx, id)
	if err != nil {
		return false, err
	}

	// 如果没有设置速率限制，则允许
	if key.RateLimit <= 0 {
		return true, nil
	}

	// 使用Redis速率限制器检查
	if r.rateLimiter != nil {
		config := ratelimit.RateLimitConfig{
			Limit:  key.RateLimit,
			Window: time.Minute, // 每分钟限制
		}

		result, err := r.rateLimiter.CheckFixed(ctx, fmt.Sprintf("apikey:%s", id), config)
		if err != nil {
			// 如果Redis出错，记录日志但允许请求通过
			logrus.WithError(err).Warn("Rate limiter check failed, allowing request")
			return true, nil
		}

		// 更新最后使用时间
		if result.Allowed {
			r.UpdateLastUsed(ctx, id)
		}

		return result.Allowed, nil
	}

	// 如果没有配置速率限制器，则允许
	return true, nil
}

// updateStatus 更新状态
func (r *PostgresAPIKeyRepository) updateStatus(ctx context.Context, id, status string) error {
	query := `
		UPDATE api_keys 
		SET status = $2, updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	_, err := r.db.ExecContext(ctx, query, id, status)
	if err != nil {
		return fmt.Errorf("failed to update status: %w", err)
	}

	return nil
}

// generateAPIKeyPair 生成API Key对
func generateAPIKeyPair() (string, string, error) {
	// 生成API Key（公开的）
	keyBytes := make([]byte, 16)
	if _, err := rand.Read(keyBytes); err != nil {
		return "", "", err
	}
	apiKey := "sk_" + hex.EncodeToString(keyBytes)

	// 生成Secret（私密的）
	secretBytes := make([]byte, 32)
	if _, err := rand.Read(secretBytes); err != nil {
		return "", "", err
	}
	secret := hex.EncodeToString(secretBytes)

	return apiKey, secret, nil
}

// hashSecret 哈希密钥
func hashSecret(secret string) (string, error) {
	hash, err := bcrypt.GenerateFromPassword([]byte(secret), bcrypt.DefaultCost)
	if err != nil {
		return "", err
	}
	return string(hash), nil
}

// verifySecret 验证密钥
func verifySecret(secret, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(secret))
	return err == nil
}

// GenerateSimpleAPIKey 生成简单的API Key（用于兼容旧系统）
func GenerateSimpleAPIKey() string {
	bytes := make([]byte, 32)
	rand.Read(bytes)
	hash := sha256.Sum256(bytes)
	return hex.EncodeToString(hash[:])
}

// ParseAPIKey 解析API Key（支持Bearer token格式）
func ParseAPIKey(authHeader string) string {
	// 支持多种格式：
	// 1. Bearer sk_xxx
	// 2. ApiKey sk_xxx
	// 3. sk_xxx

	authHeader = strings.TrimSpace(authHeader)

	if strings.HasPrefix(authHeader, "Bearer ") {
		return strings.TrimPrefix(authHeader, "Bearer ")
	}

	if strings.HasPrefix(authHeader, "ApiKey ") {
		return strings.TrimPrefix(authHeader, "ApiKey ")
	}

	if strings.HasPrefix(authHeader, "sk_") {
		return authHeader
	}

	return ""
}
