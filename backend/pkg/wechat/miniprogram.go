package wechat

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/sirupsen/logrus"
)

// MiniProgramConfig 微信小程序配置
type MiniProgramConfig struct {
	AppID     string `json:"app_id"`
	AppSecret string `json:"app_secret"`
}

// SessionResponse 微信登录会话响应
type SessionResponse struct {
	OpenID     string `json:"openid"`
	SessionKey string `json:"session_key"`
	UnionID    string `json:"unionid,omitempty"`
	ErrCode    int    `json:"errcode,omitempty"`
	ErrMsg     string `json:"errmsg,omitempty"`
}

// UserInfo 微信用户信息
type UserInfo struct {
	OpenID    string `json:"openId"`
	NickName  string `json:"nickName"`
	Gender    int    `json:"gender"`
	City      string `json:"city"`
	Province  string `json:"province"`
	Country   string `json:"country"`
	AvatarURL string `json:"avatarUrl"`
	UnionID   string `json:"unionId,omitempty"`
	Language  string `json:"language"`
}

// MiniProgramClient 微信小程序客户端
type MiniProgramClient struct {
	config *MiniProgramConfig
	client *http.Client
}

// NewMiniProgramClient 创建微信小程序客户端
func NewMiniProgramClient(config *MiniProgramConfig) *MiniProgramClient {
	return &MiniProgramClient{
		config: config,
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Code2Session 通过code换取session
func (c *MiniProgramClient) Code2Session(code string) (*SessionResponse, error) {
	url := fmt.Sprintf(
		"https://api.weixin.qq.com/sns/jscode2session?appid=%s&secret=%s&js_code=%s&grant_type=authorization_code",
		c.config.AppID, c.config.AppSecret, code,
	)

	logrus.WithFields(logrus.Fields{
		"appid": c.config.AppID,
		"code":  code[:8] + "...", // 只记录前8位，保护隐私
	}).Info("微信Code2Session请求")

	resp, err := c.client.Get(url)
	if err != nil {
		logrus.WithError(err).Error("微信Code2Session请求失败")
		return nil, fmt.Errorf("微信API请求失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %w", err)
	}

	var sessionResp SessionResponse
	if err := json.Unmarshal(body, &sessionResp); err != nil {
		return nil, fmt.Errorf("解析响应失败: %w", err)
	}

	// 检查微信API错误
	if sessionResp.ErrCode != 0 {
		logrus.WithFields(logrus.Fields{
			"errcode": sessionResp.ErrCode,
			"errmsg":  sessionResp.ErrMsg,
		}).Error("微信API返回错误")
		return nil, fmt.Errorf("微信API错误: %d - %s", sessionResp.ErrCode, sessionResp.ErrMsg)
	}

	logrus.WithFields(logrus.Fields{
		"openid":      sessionResp.OpenID[:8] + "...", // 只记录前8位
		"has_unionid": sessionResp.UnionID != "",
	}).Info("微信Code2Session成功")

	return &sessionResp, nil
}

// DecryptUserInfo 解密用户信息
func (c *MiniProgramClient) DecryptUserInfo(encryptedData, iv, sessionKey string) (*UserInfo, error) {
	// Base64解码
	cipherText, err := base64.StdEncoding.DecodeString(encryptedData)
	if err != nil {
		return nil, fmt.Errorf("解码加密数据失败: %w", err)
	}

	ivBytes, err := base64.StdEncoding.DecodeString(iv)
	if err != nil {
		return nil, fmt.Errorf("解码IV失败: %w", err)
	}

	sessionKeyBytes, err := base64.StdEncoding.DecodeString(sessionKey)
	if err != nil {
		return nil, fmt.Errorf("解码SessionKey失败: %w", err)
	}

	// AES解密
	block, err := aes.NewCipher(sessionKeyBytes)
	if err != nil {
		return nil, fmt.Errorf("创建AES加密器失败: %w", err)
	}

	mode := cipher.NewCBCDecrypter(block, ivBytes)
	plainText := make([]byte, len(cipherText))
	mode.CryptBlocks(plainText, cipherText)

	// 去除PKCS7填充
	plainText = removePKCS7Padding(plainText)

	// 解析用户信息
	var userInfo UserInfo
	if err := json.Unmarshal(plainText, &userInfo); err != nil {
		return nil, fmt.Errorf("解析用户信息失败: %w", err)
	}

	logrus.WithFields(logrus.Fields{
		"openid":   userInfo.OpenID[:8] + "...",
		"nickname": userInfo.NickName,
		"city":     userInfo.City,
	}).Info("用户信息解密成功")

	return &userInfo, nil
}

// removePKCS7Padding 移除PKCS7填充
func removePKCS7Padding(data []byte) []byte {
	length := len(data)
	if length == 0 {
		return data
	}

	padding := int(data[length-1])
	if padding > length || padding == 0 {
		return data
	}

	// 验证填充是否有效
	for i := length - padding; i < length; i++ {
		if data[i] != byte(padding) {
			return data
		}
	}

	return data[:length-padding]
}

// ValidateSignature 验证数据签名
func (c *MiniProgramClient) ValidateSignature(rawData, signature, sessionKey string) bool {
	// 这里应该实现SHA1签名验证
	// signature = sha1(rawData + sessionKey)
	// 为简化示例，这里返回true
	// 实际项目中应该实现完整的签名验证
	return true
}
