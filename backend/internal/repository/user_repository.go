package repository

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
	_ "github.com/lib/pq"
)

// User 用户模型
type User struct {
	ID        string    `json:"id"`
	OpenID    string    `json:"open_id"`
	UnionID   string    `json:"union_id"`
	TenantID  string    `json:"tenant_id"`
	Username  string    `json:"username"`
	Nickname  string    `json:"nickname"`
	Avatar    string    `json:"avatar"`
	Email     string    `json:"email"`
	Phone     string    `json:"phone"`
	Role      string    `json:"role"`
	Status    string    `json:"status"`
	LastLogin time.Time `json:"last_login"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// UserRepository 用户仓库接口
type UserRepository interface {
	Create(ctx context.Context, user *User) error
	GetByID(ctx context.Context, id string) (*User, error)
	GetByOpenID(ctx context.Context, openID string) (*User, error)
	Update(ctx context.Context, user *User) error
	Delete(ctx context.Context, id string) error
	UpdateLastLogin(ctx context.Context, userID string) error
}

// PostgresUserRepository PostgreSQL实现
type PostgresUserRepository struct {
	db *sql.DB
}

// NewPostgresUserRepository 创建PostgreSQL用户仓库
func NewPostgresUserRepository(db *sql.DB) UserRepository {
	return &PostgresUserRepository{db: db}
}

// Create 创建用户
func (r *PostgresUserRepository) Create(ctx context.Context, user *User) error {
	if user.ID == "" {
		user.ID = uuid.New().String()
	}
	if user.TenantID == "" {
		user.TenantID = "default"
	}
	if user.Role == "" {
		user.Role = "user"
	}
	if user.Status == "" {
		user.Status = "active"
	}

	query := `
		INSERT INTO users (
			id, open_id, union_id, tenant_id, username, nickname, 
			avatar, email, phone, role, status, created_at, updated_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, NOW(), NOW()
		)
	`

	_, err := r.db.ExecContext(ctx, query,
		user.ID, user.OpenID, user.UnionID, user.TenantID,
		user.Username, user.Nickname, user.Avatar,
		user.Email, user.Phone, user.Role, user.Status,
	)

	if err != nil {
		return fmt.Errorf("failed to create user: %w", err)
	}

	return nil
}

// GetByID 根据ID获取用户
func (r *PostgresUserRepository) GetByID(ctx context.Context, id string) (*User, error) {
	query := `
		SELECT 
			id, open_id, union_id, tenant_id, username, nickname,
			avatar, email, phone, role, status, last_login, created_at, updated_at
		FROM users
		WHERE id = $1 AND deleted_at IS NULL
	`

	user := &User{}
	err := r.db.QueryRowContext(ctx, query, id).Scan(
		&user.ID, &user.OpenID, &user.UnionID, &user.TenantID,
		&user.Username, &user.Nickname, &user.Avatar,
		&user.Email, &user.Phone, &user.Role, &user.Status,
		&user.LastLogin, &user.CreatedAt, &user.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("user not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get user: %w", err)
	}

	return user, nil
}

// GetByOpenID 根据OpenID获取用户
func (r *PostgresUserRepository) GetByOpenID(ctx context.Context, openID string) (*User, error) {
	query := `
		SELECT 
			id, open_id, union_id, tenant_id, username, nickname,
			avatar, email, phone, role, status, last_login, created_at, updated_at
		FROM users
		WHERE open_id = $1 AND deleted_at IS NULL
		LIMIT 1
	`

	user := &User{}
	err := r.db.QueryRowContext(ctx, query, openID).Scan(
		&user.ID, &user.OpenID, &user.UnionID, &user.TenantID,
		&user.Username, &user.Nickname, &user.Avatar,
		&user.Email, &user.Phone, &user.Role, &user.Status,
		&user.LastLogin, &user.CreatedAt, &user.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, nil // 用户不存在，返回nil而不是错误
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get user by openid: %w", err)
	}

	return user, nil
}

// Update 更新用户信息
func (r *PostgresUserRepository) Update(ctx context.Context, user *User) error {
	query := `
		UPDATE users SET
			username = $2,
			nickname = $3,
			avatar = $4,
			email = $5,
			phone = $6,
			role = $7,
			status = $8,
			updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query,
		user.ID, user.Username, user.Nickname, user.Avatar,
		user.Email, user.Phone, user.Role, user.Status,
	)

	if err != nil {
		return fmt.Errorf("failed to update user: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("user not found")
	}

	return nil
}

// Delete 软删除用户
func (r *PostgresUserRepository) Delete(ctx context.Context, id string) error {
	query := `
		UPDATE users 
		SET deleted_at = NOW(), updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete user: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("user not found")
	}

	return nil
}

// UpdateLastLogin 更新最后登录时间
func (r *PostgresUserRepository) UpdateLastLogin(ctx context.Context, userID string) error {
	query := `
		UPDATE users 
		SET last_login = NOW(), updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	_, err := r.db.ExecContext(ctx, query, userID)
	if err != nil {
		return fmt.Errorf("failed to update last login: %w", err)
	}

	return nil
}

// GetOrCreateByOpenID 根据OpenID获取或创建用户
func (r *PostgresUserRepository) GetOrCreateByOpenID(ctx context.Context, openID, unionID string) (*User, error) {
	// 先尝试获取用户
	user, err := r.GetByOpenID(ctx, openID)
	if err != nil {
		return nil, err
	}

	// 如果用户存在，更新登录时间并返回
	if user != nil {
		_ = r.UpdateLastLogin(ctx, user.ID)
		return user, nil
	}

	// 用户不存在，创建新用户
	newUser := &User{
		ID:       uuid.New().String(),
		OpenID:   openID,
		UnionID:  unionID,
		TenantID: "default",
		Username: fmt.Sprintf("user_%s", openID[:8]),
		Nickname: "微信用户",
		Role:     "user",
		Status:   "active",
	}

	err = r.Create(ctx, newUser)
	if err != nil {
		return nil, err
	}

	return newUser, nil
}
