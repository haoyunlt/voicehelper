package handlers

import (
	"database/sql"
	"fmt"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/sirupsen/logrus"
	"voicehelper/backend/pkg/wechat"
)

// WechatUser 微信用户模型
type WechatUser struct {
	ID         string    `json:"id" db:"id"`
	OpenID     string    `json:"openid" db:"openid"`
	UnionID    string    `json:"unionid" db:"unionid"`
	Nickname   string    `json:"nickname" db:"nickname"`
	Avatar     string    `json:"avatar" db:"avatar"`
	Gender     int       `json:"gender" db:"gender"`
	City       string    `json:"city" db:"city"`
	Province   string    `json:"province" db:"province"`
	Country    string    `json:"country" db:"country"`
	Language   string    `json:"language" db:"language"`
	TenantID   string    `json:"tenant_id" db:"tenant_id"`
	CreatedAt  time.Time `json:"created_at" db:"created_at"`
	UpdatedAt  time.Time `json:"updated_at" db:"updated_at"`
	LastLogin  time.Time `json:"last_login" db:"last_login"`
	IsActive   bool      `json:"is_active" db:"is_active"`
}

// JWTClaims JWT声明
type JWTClaims struct {
	UserID   string `json:"user_id"`
	OpenID   string `json:"openid"`
	TenantID string `json:"tenant_id"`
	jwt.RegisteredClaims
}

// findOrCreateWechatUser 查找或创建微信用户
func (h *APIHandler) findOrCreateWechatUser(sessionResp *wechat.SessionResponse, userInfo *wechat.UserInfo, tenantID string) (*WechatUser, error) {
	// 首先尝试查找现有用户
	user, err := h.findWechatUserByOpenID(sessionResp.OpenID)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("查询用户失败: %w", err)
	}

	now := time.Now()

	if user != nil {
		// 用户已存在，更新最后登录时间和用户信息
		user.LastLogin = now
		user.UpdatedAt = now
		
		// 如果有新的用户信息，更新用户资料
		if userInfo != nil {
			user.Nickname = userInfo.NickName
			user.Avatar = userInfo.AvatarURL
			user.Gender = userInfo.Gender
			user.City = userInfo.City
			user.Province = userInfo.Province
			user.Country = userInfo.Country
			user.Language = userInfo.Language
			if userInfo.UnionID != "" {
				user.UnionID = userInfo.UnionID
			}
		}

		if err := h.updateWechatUser(user); err != nil {
			return nil, fmt.Errorf("更新用户失败: %w", err)
		}

		logrus.WithFields(logrus.Fields{
			"user_id": user.ID,
			"openid":  sessionResp.OpenID[:8] + "...",
		}).Info("微信用户登录")

		return user, nil
	}

	// 用户不存在，创建新用户
	user = &WechatUser{
		ID:        generateUserID(),
		OpenID:    sessionResp.OpenID,
		UnionID:   sessionResp.UnionID,
		TenantID:  tenantID,
		CreatedAt: now,
		UpdatedAt: now,
		LastLogin: now,
		IsActive:  true,
	}

	// 设置用户信息
	if userInfo != nil {
		user.Nickname = userInfo.NickName
		user.Avatar = userInfo.AvatarURL
		user.Gender = userInfo.Gender
		user.City = userInfo.City
		user.Province = userInfo.Province
		user.Country = userInfo.Country
		user.Language = userInfo.Language
		if userInfo.UnionID != "" {
			user.UnionID = userInfo.UnionID
		}
	} else {
		// 设置默认值
		user.Nickname = "微信用户"
		user.Language = "zh_CN"
	}

	if err := h.createWechatUser(user); err != nil {
		return nil, fmt.Errorf("创建用户失败: %w", err)
	}

	logrus.WithFields(logrus.Fields{
		"user_id":  user.ID,
		"openid":   sessionResp.OpenID[:8] + "...",
		"nickname": user.Nickname,
	}).Info("创建新微信用户")

	return user, nil
}

// findWechatUserByOpenID 根据OpenID查找用户
func (h *APIHandler) findWechatUserByOpenID(openID string) (*WechatUser, error) {
	query := `
		SELECT id, openid, unionid, nickname, avatar, gender, city, province, 
		       country, language, tenant_id, created_at, updated_at, last_login, is_active
		FROM wechat_users 
		WHERE openid = $1 AND is_active = true
	`

	var user WechatUser
	err := h.db.QueryRow(query, openID).Scan(
		&user.ID, &user.OpenID, &user.UnionID, &user.Nickname, &user.Avatar,
		&user.Gender, &user.City, &user.Province, &user.Country, &user.Language,
		&user.TenantID, &user.CreatedAt, &user.UpdatedAt, &user.LastLogin, &user.IsActive,
	)

	if err != nil {
		return nil, err
	}

	return &user, nil
}

// createWechatUser 创建微信用户
func (h *APIHandler) createWechatUser(user *WechatUser) error {
	query := `
		INSERT INTO wechat_users (
			id, openid, unionid, nickname, avatar, gender, city, province,
			country, language, tenant_id, created_at, updated_at, last_login, is_active
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
		)
	`

	_, err := h.db.Exec(query,
		user.ID, user.OpenID, user.UnionID, user.Nickname, user.Avatar,
		user.Gender, user.City, user.Province, user.Country, user.Language,
		user.TenantID, user.CreatedAt, user.UpdatedAt, user.LastLogin, user.IsActive,
	)

	return err
}

// updateWechatUser 更新微信用户
func (h *APIHandler) updateWechatUser(user *WechatUser) error {
	query := `
		UPDATE wechat_users SET
			unionid = $2, nickname = $3, avatar = $4, gender = $5,
			city = $6, province = $7, country = $8, language = $9,
			updated_at = $10, last_login = $11
		WHERE id = $1
	`

	_, err := h.db.Exec(query,
		user.ID, user.UnionID, user.Nickname, user.Avatar, user.Gender,
		user.City, user.Province, user.Country, user.Language,
		user.UpdatedAt, user.LastLogin,
	)

	return err
}

// generateJWTToken 生成JWT token
func (h *APIHandler) generateJWTToken(user *WechatUser) (string, error) {
	claims := JWTClaims{
		UserID:   user.ID,
		OpenID:   user.OpenID,
		TenantID: user.TenantID,
		RegisteredClaims: jwt.RegisteredClaims{
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(2 * time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now()),
			NotBefore: jwt.NewNumericDate(time.Now()),
			Issuer:    "voicehelper",
			Subject:   user.ID,
		},
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	
	// 从环境变量获取JWT密钥
	jwtSecret := h.authMiddleware.GetJWTSecret()
	if jwtSecret == "" {
		return "", fmt.Errorf("JWT密钥未配置")
	}

	tokenString, err := token.SignedString([]byte(jwtSecret))
	if err != nil {
		return "", fmt.Errorf("签名token失败: %w", err)
	}

	return tokenString, nil
}

// generateUserID 生成用户ID
func generateUserID() string {
	// 使用时间戳 + 随机数生成用户ID
	return fmt.Sprintf("wx_%d_%d", time.Now().UnixNano(), time.Now().Nanosecond()%10000)
}
