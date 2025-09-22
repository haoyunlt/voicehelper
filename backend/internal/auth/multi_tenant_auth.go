// 多租户身份认证和权限管理系统
// 支持SAML、OAuth2.0、OIDC等多种SSO协议
// 基于GitHub开源项目的最佳实践

package auth

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
	"golang.org/x/crypto/bcrypt"
)

// TenantType 租户类型
type TenantType string

const (
	TenantTypeEnterprise TenantType = "enterprise"
	TenantTypeTeam       TenantType = "team"
	TenantTypePersonal   TenantType = "personal"
)

// UserRole 用户角色
type UserRole string

const (
	RoleSuperAdmin  UserRole = "super_admin"
	RoleTenantAdmin UserRole = "tenant_admin"
	RoleManager     UserRole = "manager"
	RoleUser        UserRole = "user"
	RoleGuest       UserRole = "guest"
)

// Permission 权限定义
type Permission string

const (
	PermissionReadUsers      Permission = "read:users"
	PermissionWriteUsers     Permission = "write:users"
	PermissionDeleteUsers    Permission = "delete:users"
	PermissionReadTenants    Permission = "read:tenants"
	PermissionWriteTenants   Permission = "write:tenants"
	PermissionReadDialogs    Permission = "read:dialogs"
	PermissionWriteDialogs   Permission = "write:dialogs"
	PermissionReadModels     Permission = "read:models"
	PermissionWriteModels    Permission = "write:models"
	PermissionReadAnalytics  Permission = "read:analytics"
	PermissionWriteAnalytics Permission = "write:analytics"
)

// Tenant 租户信息
type Tenant struct {
	ID        string                 `json:"id" db:"id"`
	Name      string                 `json:"name" db:"name"`
	Type      TenantType             `json:"type" db:"type"`
	Domain    string                 `json:"domain" db:"domain"`
	Settings  map[string]interface{} `json:"settings" db:"settings"`
	SSOConfig *SSOConfig             `json:"sso_config,omitempty" db:"sso_config"`
	MaxUsers  int                    `json:"max_users" db:"max_users"`
	IsActive  bool                   `json:"is_active" db:"is_active"`
	CreatedAt time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt time.Time              `json:"updated_at" db:"updated_at"`
}

// User 用户信息
type User struct {
	ID           string                 `json:"id" db:"id"`
	TenantID     string                 `json:"tenant_id" db:"tenant_id"`
	Email        string                 `json:"email" db:"email"`
	Username     string                 `json:"username" db:"username"`
	FullName     string                 `json:"full_name" db:"full_name"`
	PasswordHash string                 `json:"-" db:"password_hash"`
	Role         UserRole               `json:"role" db:"role"`
	Permissions  []Permission           `json:"permissions" db:"permissions"`
	Metadata     map[string]interface{} `json:"metadata" db:"metadata"`
	IsActive     bool                   `json:"is_active" db:"is_active"`
	LastLoginAt  *time.Time             `json:"last_login_at" db:"last_login_at"`
	CreatedAt    time.Time              `json:"created_at" db:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at" db:"updated_at"`
}

// SSOConfig SSO配置
type SSOConfig struct {
	Provider     string                 `json:"provider"` // saml, oauth2, oidc
	EntityID     string                 `json:"entity_id"`
	SSOURL       string                 `json:"sso_url"`
	Certificate  string                 `json:"certificate"`
	ClientID     string                 `json:"client_id"`
	ClientSecret string                 `json:"client_secret"`
	RedirectURL  string                 `json:"redirect_url"`
	Scopes       []string               `json:"scopes"`
	Metadata     map[string]interface{} `json:"metadata"`
}

// AuthToken 认证令牌
type AuthToken struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	TokenType    string    `json:"token_type"`
	ExpiresIn    int       `json:"expires_in"`
	ExpiresAt    time.Time `json:"expires_at"`
	Scope        string    `json:"scope"`
}

// Claims JWT声明
type Claims struct {
	UserID      string       `json:"user_id"`
	TenantID    string       `json:"tenant_id"`
	Email       string       `json:"email"`
	Role        UserRole     `json:"role"`
	Permissions []Permission `json:"permissions"`
	jwt.RegisteredClaims
}

// MultiTenantAuthService 多租户认证服务
type MultiTenantAuthService struct {
	tenantRepo    TenantRepository
	userRepo      UserRepository
	sessionRepo   SessionRepository
	jwtSecret     []byte
	tokenExpiry   time.Duration
	refreshExpiry time.Duration
	logger        *logrus.Logger
}

// TenantRepository 租户存储接口
type TenantRepository interface {
	CreateTenant(ctx context.Context, tenant *Tenant) error
	GetTenant(ctx context.Context, id string) (*Tenant, error)
	GetTenantByDomain(ctx context.Context, domain string) (*Tenant, error)
	UpdateTenant(ctx context.Context, tenant *Tenant) error
	DeleteTenant(ctx context.Context, id string) error
	ListTenants(ctx context.Context, limit, offset int) ([]*Tenant, error)
}

// UserRepository 用户存储接口
type UserRepository interface {
	CreateUser(ctx context.Context, user *User) error
	GetUser(ctx context.Context, id string) (*User, error)
	GetUserByEmail(ctx context.Context, email string) (*User, error)
	GetUsersByTenant(ctx context.Context, tenantID string, limit, offset int) ([]*User, error)
	UpdateUser(ctx context.Context, user *User) error
	DeleteUser(ctx context.Context, id string) error
	UpdateLastLogin(ctx context.Context, userID string) error
}

// SessionRepository 会话存储接口
type SessionRepository interface {
	CreateSession(ctx context.Context, session *Session) error
	GetSession(ctx context.Context, sessionID string) (*Session, error)
	UpdateSession(ctx context.Context, session *Session) error
	DeleteSession(ctx context.Context, sessionID string) error
	DeleteUserSessions(ctx context.Context, userID string) error
}

// Session 用户会话
type Session struct {
	ID        string    `json:"id" db:"id"`
	UserID    string    `json:"user_id" db:"user_id"`
	TenantID  string    `json:"tenant_id" db:"tenant_id"`
	Token     string    `json:"token" db:"token"`
	ExpiresAt time.Time `json:"expires_at" db:"expires_at"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

// NewMultiTenantAuthService 创建多租户认证服务
func NewMultiTenantAuthService(
	tenantRepo TenantRepository,
	userRepo UserRepository,
	sessionRepo SessionRepository,
	jwtSecret []byte,
	logger *logrus.Logger,
) *MultiTenantAuthService {
	return &MultiTenantAuthService{
		tenantRepo:    tenantRepo,
		userRepo:      userRepo,
		sessionRepo:   sessionRepo,
		jwtSecret:     jwtSecret,
		tokenExpiry:   24 * time.Hour,
		refreshExpiry: 7 * 24 * time.Hour,
		logger:        logger,
	}
}

// CreateTenant 创建租户
func (s *MultiTenantAuthService) CreateTenant(ctx context.Context, req *CreateTenantRequest) (*Tenant, error) {
	// 验证域名唯一性
	existing, err := s.tenantRepo.GetTenantByDomain(ctx, req.Domain)
	if err == nil && existing != nil {
		return nil, fmt.Errorf("domain already exists: %s", req.Domain)
	}

	tenant := &Tenant{
		ID:        uuid.New().String(),
		Name:      req.Name,
		Type:      req.Type,
		Domain:    req.Domain,
		Settings:  req.Settings,
		MaxUsers:  req.MaxUsers,
		IsActive:  true,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.tenantRepo.CreateTenant(ctx, tenant); err != nil {
		s.logger.WithError(err).Error("Failed to create tenant")
		return nil, fmt.Errorf("failed to create tenant: %w", err)
	}

	// 创建租户管理员用户
	if req.AdminUser != nil {
		adminUser := &User{
			ID:       uuid.New().String(),
			TenantID: tenant.ID,
			Email:    req.AdminUser.Email,
			Username: req.AdminUser.Username,
			FullName: req.AdminUser.FullName,
			Role:     RoleTenantAdmin,
			Permissions: []Permission{
				PermissionReadUsers, PermissionWriteUsers, PermissionDeleteUsers,
				PermissionReadDialogs, PermissionWriteDialogs,
				PermissionReadModels, PermissionWriteModels,
				PermissionReadAnalytics,
			},
			IsActive:  true,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
		}

		// 设置密码
		if req.AdminUser.Password != "" {
			hashedPassword, err := s.hashPassword(req.AdminUser.Password)
			if err != nil {
				return nil, fmt.Errorf("failed to hash password: %w", err)
			}
			adminUser.PasswordHash = hashedPassword
		}

		if err := s.userRepo.CreateUser(ctx, adminUser); err != nil {
			s.logger.WithError(err).Error("Failed to create admin user")
			// 回滚租户创建
			s.tenantRepo.DeleteTenant(ctx, tenant.ID)
			return nil, fmt.Errorf("failed to create admin user: %w", err)
		}
	}

	s.logger.WithFields(logrus.Fields{
		"tenant_id": tenant.ID,
		"domain":    tenant.Domain,
	}).Info("Tenant created successfully")

	return tenant, nil
}

// CreateTenantRequest 创建租户请求
type CreateTenantRequest struct {
	Name      string                 `json:"name"`
	Type      TenantType             `json:"type"`
	Domain    string                 `json:"domain"`
	Settings  map[string]interface{} `json:"settings"`
	MaxUsers  int                    `json:"max_users"`
	AdminUser *CreateUserRequest     `json:"admin_user,omitempty"`
}

// CreateUserRequest 创建用户请求
type CreateUserRequest struct {
	Email    string `json:"email"`
	Username string `json:"username"`
	FullName string `json:"full_name"`
	Password string `json:"password"`
}

// Authenticate 用户认证
func (s *MultiTenantAuthService) Authenticate(ctx context.Context, email, password, domain string) (*AuthToken, *User, error) {
	// 根据域名获取租户
	tenant, err := s.tenantRepo.GetTenantByDomain(ctx, domain)
	if err != nil {
		return nil, nil, fmt.Errorf("tenant not found for domain: %s", domain)
	}

	if !tenant.IsActive {
		return nil, nil, fmt.Errorf("tenant is inactive")
	}

	// 获取用户
	user, err := s.userRepo.GetUserByEmail(ctx, email)
	if err != nil {
		return nil, nil, fmt.Errorf("user not found")
	}

	if user.TenantID != tenant.ID {
		return nil, nil, fmt.Errorf("user does not belong to this tenant")
	}

	if !user.IsActive {
		return nil, nil, fmt.Errorf("user is inactive")
	}

	// 验证密码
	if !s.verifyPassword(password, user.PasswordHash) {
		return nil, nil, fmt.Errorf("invalid credentials")
	}

	// 生成令牌
	token, err := s.generateToken(user)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate token: %w", err)
	}

	// 更新最后登录时间
	if err := s.userRepo.UpdateLastLogin(ctx, user.ID); err != nil {
		s.logger.WithError(err).Warn("Failed to update last login time")
	}

	// 创建会话
	session := &Session{
		ID:        uuid.New().String(),
		UserID:    user.ID,
		TenantID:  user.TenantID,
		Token:     token.AccessToken,
		ExpiresAt: token.ExpiresAt,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	if err := s.sessionRepo.CreateSession(ctx, session); err != nil {
		s.logger.WithError(err).Warn("Failed to create session")
	}

	s.logger.WithFields(logrus.Fields{
		"user_id":   user.ID,
		"tenant_id": user.TenantID,
		"email":     user.Email,
	}).Info("User authenticated successfully")

	return token, user, nil
}

// ValidateToken 验证令牌
func (s *MultiTenantAuthService) ValidateToken(tokenString string) (*Claims, error) {
	token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
		}
		return s.jwtSecret, nil
	})

	if err != nil {
		return nil, fmt.Errorf("invalid token: %w", err)
	}

	if claims, ok := token.Claims.(*Claims); ok && token.Valid {
		return claims, nil
	}

	return nil, fmt.Errorf("invalid token claims")
}

// RefreshToken 刷新令牌
func (s *MultiTenantAuthService) RefreshToken(ctx context.Context, refreshToken string) (*AuthToken, error) {
	// 验证刷新令牌
	claims, err := s.ValidateToken(refreshToken)
	if err != nil {
		return nil, fmt.Errorf("invalid refresh token: %w", err)
	}

	// 获取用户信息
	user, err := s.userRepo.GetUser(ctx, claims.UserID)
	if err != nil {
		return nil, fmt.Errorf("user not found: %w", err)
	}

	if !user.IsActive {
		return nil, fmt.Errorf("user is inactive")
	}

	// 生成新令牌
	newToken, err := s.generateToken(user)
	if err != nil {
		return nil, fmt.Errorf("failed to generate new token: %w", err)
	}

	return newToken, nil
}

// Logout 用户登出
func (s *MultiTenantAuthService) Logout(ctx context.Context, tokenString string) error {
	claims, err := s.ValidateToken(tokenString)
	if err != nil {
		return fmt.Errorf("invalid token: %w", err)
	}

	// 删除会话
	if err := s.sessionRepo.DeleteUserSessions(ctx, claims.UserID); err != nil {
		s.logger.WithError(err).Warn("Failed to delete user sessions")
	}

	s.logger.WithField("user_id", claims.UserID).Info("User logged out")
	return nil
}

// CheckPermission 检查权限
func (s *MultiTenantAuthService) CheckPermission(user *User, permission Permission) bool {
	// 超级管理员拥有所有权限
	if user.Role == RoleSuperAdmin {
		return true
	}

	// 检查用户权限列表
	for _, p := range user.Permissions {
		if p == permission {
			return true
		}
	}

	// 检查角色默认权限
	return s.hasRolePermission(user.Role, permission)
}

// hasRolePermission 检查角色默认权限
func (s *MultiTenantAuthService) hasRolePermission(role UserRole, permission Permission) bool {
	rolePermissions := map[UserRole][]Permission{
		RoleTenantAdmin: {
			PermissionReadUsers, PermissionWriteUsers, PermissionDeleteUsers,
			PermissionReadDialogs, PermissionWriteDialogs,
			PermissionReadModels, PermissionWriteModels,
			PermissionReadAnalytics,
		},
		RoleManager: {
			PermissionReadUsers, PermissionWriteUsers,
			PermissionReadDialogs, PermissionWriteDialogs,
			PermissionReadModels, PermissionReadAnalytics,
		},
		RoleUser: {
			PermissionReadDialogs, PermissionWriteDialogs,
			PermissionReadModels,
		},
		RoleGuest: {
			PermissionReadDialogs,
		},
	}

	permissions, exists := rolePermissions[role]
	if !exists {
		return false
	}

	for _, p := range permissions {
		if p == permission {
			return true
		}
	}

	return false
}

// generateToken 生成JWT令牌
func (s *MultiTenantAuthService) generateToken(user *User) (*AuthToken, error) {
	now := time.Now()
	expiresAt := now.Add(s.tokenExpiry)

	claims := &Claims{
		UserID:      user.ID,
		TenantID:    user.TenantID,
		Email:       user.Email,
		Role:        user.Role,
		Permissions: user.Permissions,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(expiresAt),
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			Issuer:    "voicehelper",
			Subject:   user.ID,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	accessToken, err := token.SignedString(s.jwtSecret)
	if err != nil {
		return nil, fmt.Errorf("failed to sign access token: %w", err)
	}

	// 生成刷新令牌
	refreshClaims := &Claims{
		UserID:   user.ID,
		TenantID: user.TenantID,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(now.Add(s.refreshExpiry)),
			IssuedAt:  jwt.NewNumericDate(now),
			NotBefore: jwt.NewNumericDate(now),
			Issuer:    "voicehelper",
			Subject:   user.ID,
		},
	}

	refreshTokenJWT := jwt.NewWithClaims(jwt.SigningMethodHS256, refreshClaims)
	refreshToken, err := refreshTokenJWT.SignedString(s.jwtSecret)
	if err != nil {
		return nil, fmt.Errorf("failed to sign refresh token: %w", err)
	}

	return &AuthToken{
		AccessToken:  accessToken,
		RefreshToken: refreshToken,
		TokenType:    "Bearer",
		ExpiresIn:    int(s.tokenExpiry.Seconds()),
		ExpiresAt:    expiresAt,
		Scope:        strings.Join(permissionsToStrings(user.Permissions), " "),
	}, nil
}

// hashPassword 哈希密码
func (s *MultiTenantAuthService) hashPassword(password string) (string, error) {
	bytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	return string(bytes), err
}

// verifyPassword 验证密码
func (s *MultiTenantAuthService) verifyPassword(password, hash string) bool {
	err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password))
	return err == nil
}

// generateRandomString 生成随机字符串
func (s *MultiTenantAuthService) generateRandomString(length int) (string, error) {
	bytes := make([]byte, length)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(bytes)[:length], nil
}

// permissionsToStrings 权限转字符串数组
func permissionsToStrings(permissions []Permission) []string {
	result := make([]string, len(permissions))
	for i, p := range permissions {
		result[i] = string(p)
	}
	return result
}
