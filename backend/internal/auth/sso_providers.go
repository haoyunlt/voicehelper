// SSO提供商集成
// 支持SAML、OAuth2.0、OIDC等多种SSO协议

package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/coreos/go-oidc/v3/oidc"
	"github.com/crewjam/saml/samlsp"
	"github.com/sirupsen/logrus"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"golang.org/x/oauth2/microsoft"
)

// SSOProvider SSO提供商接口
type SSOProvider interface {
	GetAuthURL(state string) string
	HandleCallback(ctx context.Context, code, state string) (*SSOUserInfo, error)
	ValidateToken(ctx context.Context, token string) (*SSOUserInfo, error)
}

// SSOUserInfo SSO用户信息
type SSOUserInfo struct {
	ID       string            `json:"id"`
	Email    string            `json:"email"`
	Name     string            `json:"name"`
	Username string            `json:"username"`
	Avatar   string            `json:"avatar"`
	Metadata map[string]string `json:"metadata"`
}

// OIDCProvider OIDC提供商
type OIDCProvider struct {
	config   *oauth2.Config
	verifier *oidc.IDTokenVerifier
	logger   *logrus.Logger
}

// NewOIDCProvider 创建OIDC提供商
func NewOIDCProvider(ctx context.Context, config *SSOConfig, logger *logrus.Logger) (*OIDCProvider, error) {
	provider, err := oidc.NewProvider(ctx, config.SSOURL)
	if err != nil {
		return nil, fmt.Errorf("failed to create OIDC provider: %w", err)
	}

	oauth2Config := &oauth2.Config{
		ClientID:     config.ClientID,
		ClientSecret: config.ClientSecret,
		RedirectURL:  config.RedirectURL,
		Endpoint:     provider.Endpoint(),
		Scopes:       config.Scopes,
	}

	verifier := provider.Verifier(&oidc.Config{
		ClientID: config.ClientID,
	})

	return &OIDCProvider{
		config:   oauth2Config,
		verifier: verifier,
		logger:   logger,
	}, nil
}

// GetAuthURL 获取认证URL
func (p *OIDCProvider) GetAuthURL(state string) string {
	return p.config.AuthCodeURL(state, oauth2.AccessTypeOffline)
}

// HandleCallback 处理回调
func (p *OIDCProvider) HandleCallback(ctx context.Context, code, state string) (*SSOUserInfo, error) {
	token, err := p.config.Exchange(ctx, code)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}

	rawIDToken, ok := token.Extra("id_token").(string)
	if !ok {
		return nil, fmt.Errorf("no id_token in token response")
	}

	idToken, err := p.verifier.Verify(ctx, rawIDToken)
	if err != nil {
		return nil, fmt.Errorf("failed to verify ID token: %w", err)
	}

	var claims struct {
		Sub      string `json:"sub"`
		Email    string `json:"email"`
		Name     string `json:"name"`
		Username string `json:"preferred_username"`
		Picture  string `json:"picture"`
	}

	if err := idToken.Claims(&claims); err != nil {
		return nil, fmt.Errorf("failed to parse claims: %w", err)
	}

	return &SSOUserInfo{
		ID:       claims.Sub,
		Email:    claims.Email,
		Name:     claims.Name,
		Username: claims.Username,
		Avatar:   claims.Picture,
		Metadata: map[string]string{
			"provider": "oidc",
		},
	}, nil
}

// ValidateToken 验证令牌
func (p *OIDCProvider) ValidateToken(ctx context.Context, tokenString string) (*SSOUserInfo, error) {
	idToken, err := p.verifier.Verify(ctx, tokenString)
	if err != nil {
		return nil, fmt.Errorf("failed to verify token: %w", err)
	}

	var claims struct {
		Sub      string `json:"sub"`
		Email    string `json:"email"`
		Name     string `json:"name"`
		Username string `json:"preferred_username"`
		Picture  string `json:"picture"`
	}

	if err := idToken.Claims(&claims); err != nil {
		return nil, fmt.Errorf("failed to parse claims: %w", err)
	}

	return &SSOUserInfo{
		ID:       claims.Sub,
		Email:    claims.Email,
		Name:     claims.Name,
		Username: claims.Username,
		Avatar:   claims.Picture,
		Metadata: map[string]string{
			"provider": "oidc",
		},
	}, nil
}

// GoogleOAuthProvider Google OAuth提供商
type GoogleOAuthProvider struct {
	config *oauth2.Config
	logger *logrus.Logger
}

// NewGoogleOAuthProvider 创建Google OAuth提供商
func NewGoogleOAuthProvider(config *SSOConfig, logger *logrus.Logger) *GoogleOAuthProvider {
	oauth2Config := &oauth2.Config{
		ClientID:     config.ClientID,
		ClientSecret: config.ClientSecret,
		RedirectURL:  config.RedirectURL,
		Scopes:       config.Scopes,
		Endpoint:     google.Endpoint,
	}

	return &GoogleOAuthProvider{
		config: oauth2Config,
		logger: logger,
	}
}

// GetAuthURL 获取认证URL
func (p *GoogleOAuthProvider) GetAuthURL(state string) string {
	return p.config.AuthCodeURL(state, oauth2.AccessTypeOffline)
}

// HandleCallback 处理回调
func (p *GoogleOAuthProvider) HandleCallback(ctx context.Context, code, state string) (*SSOUserInfo, error) {
	token, err := p.config.Exchange(ctx, code)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}

	client := p.config.Client(ctx, token)
	resp, err := client.Get("https://www.googleapis.com/oauth2/v2/userinfo")
	if err != nil {
		return nil, fmt.Errorf("failed to get user info: %w", err)
	}
	defer resp.Body.Close()

	var userInfo struct {
		ID      string `json:"id"`
		Email   string `json:"email"`
		Name    string `json:"name"`
		Picture string `json:"picture"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, fmt.Errorf("failed to decode user info: %w", err)
	}

	return &SSOUserInfo{
		ID:       userInfo.ID,
		Email:    userInfo.Email,
		Name:     userInfo.Name,
		Username: strings.Split(userInfo.Email, "@")[0],
		Avatar:   userInfo.Picture,
		Metadata: map[string]string{
			"provider": "google",
		},
	}, nil
}

// ValidateToken 验证令牌
func (p *GoogleOAuthProvider) ValidateToken(ctx context.Context, tokenString string) (*SSOUserInfo, error) {
	// Google OAuth token验证需要调用tokeninfo端点
	resp, err := http.Get(fmt.Sprintf("https://www.googleapis.com/oauth2/v1/tokeninfo?access_token=%s", tokenString))
	if err != nil {
		return nil, fmt.Errorf("failed to validate token: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("invalid token")
	}

	var tokenInfo struct {
		UserID string `json:"user_id"`
		Email  string `json:"email"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&tokenInfo); err != nil {
		return nil, fmt.Errorf("failed to decode token info: %w", err)
	}

	return &SSOUserInfo{
		ID:    tokenInfo.UserID,
		Email: tokenInfo.Email,
		Metadata: map[string]string{
			"provider": "google",
		},
	}, nil
}

// MicrosoftOAuthProvider Microsoft OAuth提供商
type MicrosoftOAuthProvider struct {
	config *oauth2.Config
	logger *logrus.Logger
}

// NewMicrosoftOAuthProvider 创建Microsoft OAuth提供商
func NewMicrosoftOAuthProvider(config *SSOConfig, logger *logrus.Logger) *MicrosoftOAuthProvider {
	oauth2Config := &oauth2.Config{
		ClientID:     config.ClientID,
		ClientSecret: config.ClientSecret,
		RedirectURL:  config.RedirectURL,
		Scopes:       config.Scopes,
		Endpoint:     microsoft.AzureADEndpoint("common"),
	}

	return &MicrosoftOAuthProvider{
		config: oauth2Config,
		logger: logger,
	}
}

// GetAuthURL 获取认证URL
func (p *MicrosoftOAuthProvider) GetAuthURL(state string) string {
	return p.config.AuthCodeURL(state)
}

// HandleCallback 处理回调
func (p *MicrosoftOAuthProvider) HandleCallback(ctx context.Context, code, state string) (*SSOUserInfo, error) {
	token, err := p.config.Exchange(ctx, code)
	if err != nil {
		return nil, fmt.Errorf("failed to exchange code: %w", err)
	}

	client := p.config.Client(ctx, token)
	resp, err := client.Get("https://graph.microsoft.com/v1.0/me")
	if err != nil {
		return nil, fmt.Errorf("failed to get user info: %w", err)
	}
	defer resp.Body.Close()

	var userInfo struct {
		ID                string `json:"id"`
		Mail              string `json:"mail"`
		UserPrincipalName string `json:"userPrincipalName"`
		DisplayName       string `json:"displayName"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, fmt.Errorf("failed to decode user info: %w", err)
	}

	email := userInfo.Mail
	if email == "" {
		email = userInfo.UserPrincipalName
	}

	return &SSOUserInfo{
		ID:       userInfo.ID,
		Email:    email,
		Name:     userInfo.DisplayName,
		Username: strings.Split(email, "@")[0],
		Metadata: map[string]string{
			"provider": "microsoft",
		},
	}, nil
}

// ValidateToken 验证令牌
func (p *MicrosoftOAuthProvider) ValidateToken(ctx context.Context, tokenString string) (*SSOUserInfo, error) {
	// 创建带有Bearer token的请求
	req, err := http.NewRequestWithContext(ctx, "GET", "https://graph.microsoft.com/v1.0/me", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+tokenString)

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to validate token: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("invalid token")
	}

	var userInfo struct {
		ID                string `json:"id"`
		Mail              string `json:"mail"`
		UserPrincipalName string `json:"userPrincipalName"`
		DisplayName       string `json:"displayName"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&userInfo); err != nil {
		return nil, fmt.Errorf("failed to decode user info: %w", err)
	}

	email := userInfo.Mail
	if email == "" {
		email = userInfo.UserPrincipalName
	}

	return &SSOUserInfo{
		ID:       userInfo.ID,
		Email:    email,
		Name:     userInfo.DisplayName,
		Username: strings.Split(email, "@")[0],
		Metadata: map[string]string{
			"provider": "microsoft",
		},
	}, nil
}

// SAMLProvider SAML提供商
type SAMLProvider struct {
	middleware *samlsp.Middleware
	logger     *logrus.Logger
}

// NewSAMLProvider 创建SAML提供商
func NewSAMLProvider(config *SSOConfig, logger *logrus.Logger) (*SAMLProvider, error) {
	keyPair, err := samlsp.DefaultSessionCookieStore()
	if err != nil {
		return nil, fmt.Errorf("failed to create session store: %w", err)
	}

	idpMetadataURL, err := url.Parse(config.SSOURL)
	if err != nil {
		return nil, fmt.Errorf("invalid IDP metadata URL: %w", err)
	}

	rootURL, err := url.Parse(config.RedirectURL)
	if err != nil {
		return nil, fmt.Errorf("invalid root URL: %w", err)
	}

	samlSP, err := samlsp.New(samlsp.Options{
		URL:            *rootURL,
		Key:            keyPair.PrivateKey,
		Certificate:    keyPair.Certificate,
		IDPMetadataURL: idpMetadataURL,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to create SAML SP: %w", err)
	}

	return &SAMLProvider{
		middleware: samlSP,
		logger:     logger,
	}, nil
}

// GetAuthURL 获取认证URL
func (p *SAMLProvider) GetAuthURL(state string) string {
	// SAML使用不同的认证流程，这里返回SAML SSO端点
	return "/saml/login"
}

// HandleCallback 处理回调
func (p *SAMLProvider) HandleCallback(ctx context.Context, code, state string) (*SSOUserInfo, error) {
	// SAML回调处理通常在中间件中完成
	// 这里需要从SAML断言中提取用户信息
	return nil, fmt.Errorf("SAML callback should be handled by middleware")
}

// ValidateToken 验证令牌
func (p *SAMLProvider) ValidateToken(ctx context.Context, token string) (*SSOUserInfo, error) {
	// SAML通常使用会话而不是token
	return nil, fmt.Errorf("SAML uses sessions, not tokens")
}

// SSOManager SSO管理器
type SSOManager struct {
	providers map[string]SSOProvider
	logger    *logrus.Logger
}

// NewSSOManager 创建SSO管理器
func NewSSOManager(logger *logrus.Logger) *SSOManager {
	return &SSOManager{
		providers: make(map[string]SSOProvider),
		logger:    logger,
	}
}

// RegisterProvider 注册SSO提供商
func (m *SSOManager) RegisterProvider(name string, provider SSOProvider) {
	m.providers[name] = provider
	m.logger.WithField("provider", name).Info("SSO provider registered")
}

// GetProvider 获取SSO提供商
func (m *SSOManager) GetProvider(name string) (SSOProvider, error) {
	provider, exists := m.providers[name]
	if !exists {
		return nil, fmt.Errorf("SSO provider not found: %s", name)
	}
	return provider, nil
}

// InitiateSSO 发起SSO认证
func (m *SSOManager) InitiateSSO(providerName, state string) (string, error) {
	provider, err := m.GetProvider(providerName)
	if err != nil {
		return "", err
	}

	authURL := provider.GetAuthURL(state)
	m.logger.WithFields(logrus.Fields{
		"provider": providerName,
		"state":    state,
	}).Info("SSO authentication initiated")

	return authURL, nil
}

// HandleSSOCallback 处理SSO回调
func (m *SSOManager) HandleSSOCallback(ctx context.Context, providerName, code, state string) (*SSOUserInfo, error) {
	provider, err := m.GetProvider(providerName)
	if err != nil {
		return nil, err
	}

	userInfo, err := provider.HandleCallback(ctx, code, state)
	if err != nil {
		m.logger.WithError(err).WithField("provider", providerName).Error("SSO callback failed")
		return nil, err
	}

	m.logger.WithFields(logrus.Fields{
		"provider": providerName,
		"user_id":  userInfo.ID,
		"email":    userInfo.Email,
	}).Info("SSO callback handled successfully")

	return userInfo, nil
}
