// 模型路由服务 - Go语言实现
// 与Python模型路由系统集成，提供HTTP API接口

package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// ModelType 模型类型枚举
type ModelType string

const (
	TextGeneration  ModelType = "text_generation"
	ChatCompletion  ModelType = "chat_completion"
	Embedding       ModelType = "embedding"
	ImageGeneration ModelType = "image_generation"
	SpeechToText    ModelType = "speech_to_text"
	TextToSpeech    ModelType = "text_to_speech"
	Vision          ModelType = "vision"
	CodeGeneration  ModelType = "code_generation"
)

// RoutingRequest 路由请求结构
type RoutingRequest struct {
	ModelType   ModelType              `json:"model_type"`
	Prompt      string                 `json:"prompt"`
	MaxTokens   *int                   `json:"max_tokens,omitempty"`
	Temperature *float64               `json:"temperature,omitempty"`
	UserID      *string                `json:"user_id,omitempty"`
	SessionID   *string                `json:"session_id,omitempty"`
	Priority    int                    `json:"priority"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// RoutingResponse 路由响应结构
type RoutingResponse struct {
	ModelID      string                 `json:"model_id"`
	Response     interface{}            `json:"response"`
	ResponseTime float64                `json:"response_time"`
	Cost         float64                `json:"cost"`
	TokensUsed   int                    `json:"tokens_used"`
	Success      bool                   `json:"success"`
	Error        *string                `json:"error,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// ModelMetrics 模型指标
type ModelMetrics struct {
	ModelID            string  `json:"model_id"`
	TotalRequests      int     `json:"total_requests"`
	SuccessfulRequests int     `json:"successful_requests"`
	FailedRequests     int     `json:"failed_requests"`
	AvgResponseTime    float64 `json:"avg_response_time"`
	AvgCost            float64 `json:"avg_cost"`
	LastRequestTime    float64 `json:"last_request_time"`
	CurrentLoad        int     `json:"current_load"`
	ErrorRate          float64 `json:"error_rate"`
	Availability       float64 `json:"availability"`
}

// ModelStatus 模型状态
type ModelStatus struct {
	Name            string  `json:"name"`
	Provider        string  `json:"provider"`
	Type            string  `json:"type"`
	Enabled         bool    `json:"enabled"`
	Priority        int     `json:"priority"`
	CurrentLoad     int     `json:"current_load"`
	Availability    float64 `json:"availability"`
	AvgResponseTime float64 `json:"avg_response_time"`
	ErrorRate       float64 `json:"error_rate"`
	TotalRequests   int     `json:"total_requests"`
}

// ModelRouterService 模型路由服务
type ModelRouterService struct {
	pythonServiceURL string
	httpClient       *http.Client
	metrics          map[string]*ModelMetrics
	metricsMutex     sync.RWMutex
	logger           *logrus.Logger
}

// NewModelRouterService 创建模型路由服务实例
func NewModelRouterService(pythonServiceURL string, logger *logrus.Logger) *ModelRouterService {
	return &ModelRouterService{
		pythonServiceURL: pythonServiceURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
		metrics: make(map[string]*ModelMetrics),
		logger:  logger,
	}
}

// RouteRequest 路由请求到最适合的模型
func (s *ModelRouterService) RouteRequest(ctx context.Context, req *RoutingRequest) (*RoutingResponse, error) {
	startTime := time.Now()

	// 序列化请求
	reqBody, err := json.Marshal(req)
	if err != nil {
		s.logger.WithError(err).Error("Failed to marshal routing request")
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// 创建HTTP请求
	httpReq, err := http.NewRequestWithContext(ctx, "POST", s.pythonServiceURL+"/route", bytes.NewBuffer(reqBody))
	if err != nil {
		s.logger.WithError(err).Error("Failed to create HTTP request")
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// 发送请求
	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		s.logger.WithError(err).Error("Failed to send routing request")
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	// 读取响应
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		s.logger.WithError(err).Error("Failed to read response body")
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		s.logger.WithField("status", resp.StatusCode).WithField("body", string(body)).Error("Routing request failed")
		return nil, fmt.Errorf("routing request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// 解析响应
	var routingResp RoutingResponse
	if err := json.Unmarshal(body, &routingResp); err != nil {
		s.logger.WithError(err).Error("Failed to unmarshal routing response")
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	// 记录指标
	responseTime := time.Since(startTime).Seconds()
	s.updateMetrics(routingResp.ModelID, routingResp.Success, responseTime, routingResp.Cost)

	s.logger.WithFields(logrus.Fields{
		"model_id":      routingResp.ModelID,
		"success":       routingResp.Success,
		"response_time": responseTime,
		"cost":          routingResp.Cost,
		"tokens_used":   routingResp.TokensUsed,
	}).Info("Routing request completed")

	return &routingResp, nil
}

// GetModelMetrics 获取模型指标
func (s *ModelRouterService) GetModelMetrics(ctx context.Context) (map[string]*ModelMetrics, error) {
	// 从Python服务获取最新指标
	httpReq, err := http.NewRequestWithContext(ctx, "GET", s.pythonServiceURL+"/metrics", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create metrics request: %w", err)
	}

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		s.logger.WithError(err).Error("Failed to get model metrics")
		return nil, fmt.Errorf("failed to get metrics: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("metrics request failed with status %d", resp.StatusCode)
	}

	var metrics map[string]*ModelMetrics
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		return nil, fmt.Errorf("failed to decode metrics: %w", err)
	}

	// 更新本地缓存
	s.metricsMutex.Lock()
	s.metrics = metrics
	s.metricsMutex.Unlock()

	return metrics, nil
}

// GetModelStatus 获取模型状态
func (s *ModelRouterService) GetModelStatus(ctx context.Context) (map[string]*ModelStatus, error) {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", s.pythonServiceURL+"/status", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create status request: %w", err)
	}

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		s.logger.WithError(err).Error("Failed to get model status")
		return nil, fmt.Errorf("failed to get status: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("status request failed with status %d", resp.StatusCode)
	}

	var status map[string]*ModelStatus
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, fmt.Errorf("failed to decode status: %w", err)
	}

	return status, nil
}

// HealthCheck 健康检查
func (s *ModelRouterService) HealthCheck(ctx context.Context) error {
	httpReq, err := http.NewRequestWithContext(ctx, "GET", s.pythonServiceURL+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("health check request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check failed with status %d", resp.StatusCode)
	}

	return nil
}

// updateMetrics 更新本地指标缓存
func (s *ModelRouterService) updateMetrics(modelID string, success bool, responseTime, cost float64) {
	s.metricsMutex.Lock()
	defer s.metricsMutex.Unlock()

	metrics, exists := s.metrics[modelID]
	if !exists {
		metrics = &ModelMetrics{
			ModelID: modelID,
		}
		s.metrics[modelID] = metrics
	}

	metrics.TotalRequests++
	metrics.LastRequestTime = float64(time.Now().Unix())

	if success {
		metrics.SuccessfulRequests++
	} else {
		metrics.FailedRequests++
	}

	// 更新平均响应时间和成本
	if metrics.TotalRequests == 1 {
		metrics.AvgResponseTime = responseTime
		metrics.AvgCost = cost
	} else {
		alpha := 0.1 // 指数移动平均
		metrics.AvgResponseTime = (1-alpha)*metrics.AvgResponseTime + alpha*responseTime
		metrics.AvgCost = (1-alpha)*metrics.AvgCost + alpha*cost
	}

	// 更新错误率和可用性
	metrics.ErrorRate = float64(metrics.FailedRequests) / float64(metrics.TotalRequests)
	metrics.Availability = float64(metrics.SuccessfulRequests) / float64(metrics.TotalRequests)
}

// GetCachedMetrics 获取缓存的指标
func (s *ModelRouterService) GetCachedMetrics() map[string]*ModelMetrics {
	s.metricsMutex.RLock()
	defer s.metricsMutex.RUnlock()

	// 创建副本
	result := make(map[string]*ModelMetrics)
	for k, v := range s.metrics {
		result[k] = &ModelMetrics{
			ModelID:            v.ModelID,
			TotalRequests:      v.TotalRequests,
			SuccessfulRequests: v.SuccessfulRequests,
			FailedRequests:     v.FailedRequests,
			AvgResponseTime:    v.AvgResponseTime,
			AvgCost:            v.AvgCost,
			LastRequestTime:    v.LastRequestTime,
			CurrentLoad:        v.CurrentLoad,
			ErrorRate:          v.ErrorRate,
			Availability:       v.Availability,
		}
	}

	return result
}

// ChatCompletion 聊天完成接口（便捷方法）
func (s *ModelRouterService) ChatCompletion(ctx context.Context, prompt string, options ...func(*RoutingRequest)) (*RoutingResponse, error) {
	req := &RoutingRequest{
		ModelType: ChatCompletion,
		Prompt:    prompt,
		Priority:  5, // 默认优先级
	}

	// 应用选项
	for _, option := range options {
		option(req)
	}

	return s.RouteRequest(ctx, req)
}

// WithMaxTokens 设置最大token数
func WithMaxTokens(maxTokens int) func(*RoutingRequest) {
	return func(req *RoutingRequest) {
		req.MaxTokens = &maxTokens
	}
}

// WithTemperature 设置温度参数
func WithTemperature(temperature float64) func(*RoutingRequest) {
	return func(req *RoutingRequest) {
		req.Temperature = &temperature
	}
}

// WithUserID 设置用户ID
func WithUserID(userID string) func(*RoutingRequest) {
	return func(req *RoutingRequest) {
		req.UserID = &userID
	}
}

// WithSessionID 设置会话ID
func WithSessionID(sessionID string) func(*RoutingRequest) {
	return func(req *RoutingRequest) {
		req.SessionID = &sessionID
	}
}

// WithPriority 设置优先级
func WithPriority(priority int) func(*RoutingRequest) {
	return func(req *RoutingRequest) {
		req.Priority = priority
	}
}

// WithMetadata 设置元数据
func WithMetadata(metadata map[string]interface{}) func(*RoutingRequest) {
	return func(req *RoutingRequest) {
		req.Metadata = metadata
	}
}
