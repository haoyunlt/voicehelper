package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// Tenant 租户模型
type Tenant struct {
	ID        string                 `json:"id"`
	TenantID  string                 `json:"tenant_id"`
	Name      string                 `json:"name"`
	Plan      string                 `json:"plan"`   // free, basic, premium, enterprise
	Status    string                 `json:"status"` // active, suspended, deleted
	Config    map[string]interface{} `json:"config"`
	Quota     map[string]interface{} `json:"quota"`
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
}

// TenantRepository 租户仓库接口
type TenantRepository interface {
	Create(ctx context.Context, tenant *Tenant) error
	GetByID(ctx context.Context, tenantID string) (*Tenant, error)
	GetByTenantID(ctx context.Context, tenantID string) (*Tenant, error)
	Update(ctx context.Context, tenant *Tenant) error
	Delete(ctx context.Context, tenantID string) error
	List(ctx context.Context, limit, offset int) ([]*Tenant, error)
	GetByStatus(ctx context.Context, status string, limit, offset int) ([]*Tenant, error)
	UpdateQuota(ctx context.Context, tenantID string, quota map[string]interface{}) error
	GetStats(ctx context.Context, tenantID string) (*TenantStats, error)
}

// TenantStats 租户统计
type TenantStats struct {
	TenantID          string `json:"tenant_id"`
	UserCount         int    `json:"user_count"`
	ConversationCount int    `json:"conversation_count"`
	MessageCount      int    `json:"message_count"`
	DocumentCount     int    `json:"document_count"`
	VoiceSessionCount int    `json:"voice_session_count"`
	TokenUsage        int64  `json:"token_usage"`
	AudioMinutes      int    `json:"audio_minutes"`
}

// PostgresTenantRepository PostgreSQL租户仓库实现
type PostgresTenantRepository struct {
	db *sql.DB
}

// NewPostgresTenantRepository 创建PostgreSQL租户仓库
func NewPostgresTenantRepository(db *sql.DB) TenantRepository {
	return &PostgresTenantRepository{db: db}
}

// Create 创建租户
func (r *PostgresTenantRepository) Create(ctx context.Context, tenant *Tenant) error {
	if tenant.ID == "" {
		tenant.ID = uuid.New().String()
	}

	configJSON, err := json.Marshal(tenant.Config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	quotaJSON, err := json.Marshal(tenant.Quota)
	if err != nil {
		return fmt.Errorf("failed to marshal quota: %v", err)
	}

	query := `
		INSERT INTO tenants (id, tenant_id, name, plan, status, config, quota, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
	`

	_, err = r.db.ExecContext(ctx, query,
		tenant.ID, tenant.TenantID, tenant.Name, tenant.Plan,
		tenant.Status, configJSON, quotaJSON)
	if err != nil {
		return fmt.Errorf("failed to create tenant: %v", err)
	}

	logrus.WithField("tenant_id", tenant.TenantID).Info("Tenant created")
	return nil
}

// GetByID 根据ID获取租户
func (r *PostgresTenantRepository) GetByID(ctx context.Context, id string) (*Tenant, error) {
	query := `
		SELECT id, tenant_id, name, plan, status, config, quota, created_at, updated_at
		FROM tenants WHERE id = $1
	`

	return r.scanTenant(ctx, query, id)
}

// GetByTenantID 根据租户ID获取租户
func (r *PostgresTenantRepository) GetByTenantID(ctx context.Context, tenantID string) (*Tenant, error) {
	query := `
		SELECT id, tenant_id, name, plan, status, config, quota, created_at, updated_at
		FROM tenants WHERE tenant_id = $1
	`

	return r.scanTenant(ctx, query, tenantID)
}

// Update 更新租户
func (r *PostgresTenantRepository) Update(ctx context.Context, tenant *Tenant) error {
	configJSON, err := json.Marshal(tenant.Config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	quotaJSON, err := json.Marshal(tenant.Quota)
	if err != nil {
		return fmt.Errorf("failed to marshal quota: %v", err)
	}

	query := `
		UPDATE tenants 
		SET name = $2, plan = $3, status = $4, config = $5, quota = $6, updated_at = CURRENT_TIMESTAMP
		WHERE tenant_id = $1
	`

	result, err := r.db.ExecContext(ctx, query,
		tenant.TenantID, tenant.Name, tenant.Plan, tenant.Status, configJSON, quotaJSON)
	if err != nil {
		return fmt.Errorf("failed to update tenant: %v", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %v", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found: %s", tenant.TenantID)
	}

	logrus.WithField("tenant_id", tenant.TenantID).Info("Tenant updated")
	return nil
}

// Delete 删除租户
func (r *PostgresTenantRepository) Delete(ctx context.Context, tenantID string) error {
	query := `UPDATE tenants SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE tenant_id = $1`

	result, err := r.db.ExecContext(ctx, query, tenantID)
	if err != nil {
		return fmt.Errorf("failed to delete tenant: %v", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %v", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found: %s", tenantID)
	}

	logrus.WithField("tenant_id", tenantID).Info("Tenant deleted")
	return nil
}

// List 列出租户
func (r *PostgresTenantRepository) List(ctx context.Context, limit, offset int) ([]*Tenant, error) {
	query := `
		SELECT id, tenant_id, name, plan, status, config, quota, created_at, updated_at
		FROM tenants 
		WHERE status != 'deleted'
		ORDER BY created_at DESC
		LIMIT $1 OFFSET $2
	`

	rows, err := r.db.QueryContext(ctx, query, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to list tenants: %v", err)
	}
	defer rows.Close()

	var tenants []*Tenant
	for rows.Next() {
		tenant, err := r.scanTenantRow(rows)
		if err != nil {
			return nil, err
		}
		tenants = append(tenants, tenant)
	}

	return tenants, nil
}

// GetByStatus 根据状态获取租户
func (r *PostgresTenantRepository) GetByStatus(ctx context.Context, status string, limit, offset int) ([]*Tenant, error) {
	query := `
		SELECT id, tenant_id, name, plan, status, config, quota, created_at, updated_at
		FROM tenants 
		WHERE status = $1
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3
	`

	rows, err := r.db.QueryContext(ctx, query, status, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to get tenants by status: %v", err)
	}
	defer rows.Close()

	var tenants []*Tenant
	for rows.Next() {
		tenant, err := r.scanTenantRow(rows)
		if err != nil {
			return nil, err
		}
		tenants = append(tenants, tenant)
	}

	return tenants, nil
}

// UpdateQuota 更新租户配额
func (r *PostgresTenantRepository) UpdateQuota(ctx context.Context, tenantID string, quota map[string]interface{}) error {
	quotaJSON, err := json.Marshal(quota)
	if err != nil {
		return fmt.Errorf("failed to marshal quota: %v", err)
	}

	query := `
		UPDATE tenants 
		SET quota = $2, updated_at = CURRENT_TIMESTAMP
		WHERE tenant_id = $1
	`

	result, err := r.db.ExecContext(ctx, query, tenantID, quotaJSON)
	if err != nil {
		return fmt.Errorf("failed to update quota: %v", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %v", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("tenant not found: %s", tenantID)
	}

	logrus.WithField("tenant_id", tenantID).Info("Tenant quota updated")
	return nil
}

// GetStats 获取租户统计
func (r *PostgresTenantRepository) GetStats(ctx context.Context, tenantID string) (*TenantStats, error) {
	stats := &TenantStats{TenantID: tenantID}

	// 获取用户数量
	err := r.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM users WHERE tenant_id = $1 AND status = 'active'",
		tenantID).Scan(&stats.UserCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get user count: %v", err)
	}

	// 获取会话数量
	err = r.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM conversations WHERE tenant_id = $1 AND status = 'active'",
		tenantID).Scan(&stats.ConversationCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get conversation count: %v", err)
	}

	// 获取消息数量
	err = r.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM messages WHERE tenant_id = $1",
		tenantID).Scan(&stats.MessageCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get message count: %v", err)
	}

	// 获取文档数量
	err = r.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM documents WHERE tenant_id = $1 AND status = 'active'",
		tenantID).Scan(&stats.DocumentCount)
	if err != nil {
		// 如果文档表不存在，设为0
		stats.DocumentCount = 0
	}

	// 获取语音会话数量
	err = r.db.QueryRowContext(ctx,
		"SELECT COUNT(*) FROM voice_sessions WHERE tenant_id = $1",
		tenantID).Scan(&stats.VoiceSessionCount)
	if err != nil {
		// 如果语音会话表不存在，设为0
		stats.VoiceSessionCount = 0
	}

	return stats, nil
}

// scanTenant 扫描单个租户
func (r *PostgresTenantRepository) scanTenant(ctx context.Context, query string, args ...interface{}) (*Tenant, error) {
	row := r.db.QueryRowContext(ctx, query, args...)
	return r.scanTenantRow(row)
}

// scanTenantRow 扫描租户行
func (r *PostgresTenantRepository) scanTenantRow(scanner interface {
	Scan(dest ...interface{}) error
}) (*Tenant, error) {
	var tenant Tenant
	var configJSON, quotaJSON []byte

	err := scanner.Scan(
		&tenant.ID, &tenant.TenantID, &tenant.Name, &tenant.Plan, &tenant.Status,
		&configJSON, &quotaJSON, &tenant.CreatedAt, &tenant.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("tenant not found")
		}
		return nil, fmt.Errorf("failed to scan tenant: %v", err)
	}

	// 解析JSON字段
	if err := json.Unmarshal(configJSON, &tenant.Config); err != nil {
		tenant.Config = make(map[string]interface{})
	}

	if err := json.Unmarshal(quotaJSON, &tenant.Quota); err != nil {
		tenant.Quota = make(map[string]interface{})
	}

	return &tenant, nil
}
