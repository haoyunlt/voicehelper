package health

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/redis/go-redis/v9"
	"github.com/sirupsen/logrus"
)

// HealthStatus 健康状态枚举
type HealthStatus string

const (
	StatusHealthy   HealthStatus = "healthy"
	StatusUnhealthy HealthStatus = "unhealthy"
	StatusDegraded  HealthStatus = "degraded"
	StatusUnknown   HealthStatus = "unknown"
)

// ComponentHealth 组件健康状态
type ComponentHealth struct {
	Name         string                 `json:"name"`
	Status       HealthStatus           `json:"status"`
	Message      string                 `json:"message,omitempty"`
	LastChecked  time.Time              `json:"last_checked"`
	ResponseTime time.Duration          `json:"response_time"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// SystemHealth 系统整体健康状态
type SystemHealth struct {
	Status     HealthStatus               `json:"status"`
	Timestamp  time.Time                  `json:"timestamp"`
	Version    string                     `json:"version"`
	Uptime     time.Duration              `json:"uptime"`
	Components map[string]ComponentHealth `json:"components"`
	Summary    HealthSummary              `json:"summary"`
}

// HealthSummary 健康状态摘要
type HealthSummary struct {
	TotalComponents     int `json:"total_components"`
	HealthyComponents   int `json:"healthy_components"`
	DegradedComponents  int `json:"degraded_components"`
	UnhealthyComponents int `json:"unhealthy_components"`
}

// HealthChecker 健康检查器接口
type HealthChecker interface {
	Name() string
	Check(ctx context.Context) ComponentHealth
}

// HealthManager 健康管理器
type HealthManager struct {
	checkers   map[string]HealthChecker
	cache      map[string]ComponentHealth
	cacheMutex sync.RWMutex
	startTime  time.Time
	version    string

	// Prometheus指标
	healthCheckDuration *prometheus.HistogramVec
	healthCheckStatus   *prometheus.GaugeVec
	healthCheckTotal    *prometheus.CounterVec
}

// NewHealthManager 创建健康管理器
func NewHealthManager(version string) *HealthManager {
	return &HealthManager{
		checkers:  make(map[string]HealthChecker),
		cache:     make(map[string]ComponentHealth),
		startTime: time.Now(),
		version:   version,

		healthCheckDuration: promauto.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "voicehelper_health_check_duration_seconds",
				Help:    "Health check duration in seconds",
				Buckets: []float64{0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5},
			},
			[]string{"component", "status"},
		),

		healthCheckStatus: promauto.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "voicehelper_health_check_status",
				Help: "Health check status (1=healthy, 0.5=degraded, 0=unhealthy)",
			},
			[]string{"component"},
		),

		healthCheckTotal: promauto.NewCounterVec(
			prometheus.CounterOpts{
				Name: "voicehelper_health_check_total",
				Help: "Total number of health checks",
			},
			[]string{"component", "status"},
		),
	}
}

// RegisterChecker 注册健康检查器
func (hm *HealthManager) RegisterChecker(checker HealthChecker) {
	hm.checkers[checker.Name()] = checker
	logrus.WithField("component", checker.Name()).Info("Registered health checker")
}

// CheckAll 检查所有组件健康状态
func (hm *HealthManager) CheckAll(ctx context.Context) SystemHealth {
	start := time.Now()
	components := make(map[string]ComponentHealth)

	// 并发检查所有组件
	var wg sync.WaitGroup
	resultChan := make(chan ComponentHealth, len(hm.checkers))

	for _, checker := range hm.checkers {
		wg.Add(1)
		go func(c HealthChecker) {
			defer wg.Done()
			health := hm.checkComponent(ctx, c)
			resultChan <- health
		}(checker)
	}

	// 等待所有检查完成
	go func() {
		wg.Wait()
		close(resultChan)
	}()

	// 收集结果
	for health := range resultChan {
		components[health.Name] = health
		hm.updateCache(health)
	}

	// 计算整体状态
	overallStatus := hm.calculateOverallStatus(components)
	summary := hm.calculateSummary(components)

	systemHealth := SystemHealth{
		Status:     overallStatus,
		Timestamp:  time.Now(),
		Version:    hm.version,
		Uptime:     time.Since(hm.startTime),
		Components: components,
		Summary:    summary,
	}

	logrus.WithFields(logrus.Fields{
		"status":           overallStatus,
		"check_duration":   time.Since(start),
		"total_components": len(components),
		"healthy":          summary.HealthyComponents,
		"degraded":         summary.DegradedComponents,
		"unhealthy":        summary.UnhealthyComponents,
	}).Info("Health check completed")

	return systemHealth
}

// checkComponent 检查单个组件
func (hm *HealthManager) checkComponent(ctx context.Context, checker HealthChecker) ComponentHealth {
	start := time.Now()

	// 设置超时
	checkCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	health := checker.Check(checkCtx)
	duration := time.Since(start)
	health.ResponseTime = duration

	// 记录指标
	statusValue := hm.statusToFloat(health.Status)
	hm.healthCheckDuration.WithLabelValues(health.Name, string(health.Status)).Observe(duration.Seconds())
	hm.healthCheckStatus.WithLabelValues(health.Name).Set(statusValue)
	hm.healthCheckTotal.WithLabelValues(health.Name, string(health.Status)).Inc()

	return health
}

// updateCache 更新缓存
func (hm *HealthManager) updateCache(health ComponentHealth) {
	hm.cacheMutex.Lock()
	defer hm.cacheMutex.Unlock()
	hm.cache[health.Name] = health
}

// GetCachedHealth 获取缓存的健康状态
func (hm *HealthManager) GetCachedHealth() SystemHealth {
	hm.cacheMutex.RLock()
	defer hm.cacheMutex.RUnlock()

	components := make(map[string]ComponentHealth)
	for name, health := range hm.cache {
		components[name] = health
	}

	overallStatus := hm.calculateOverallStatus(components)
	summary := hm.calculateSummary(components)

	return SystemHealth{
		Status:     overallStatus,
		Timestamp:  time.Now(),
		Version:    hm.version,
		Uptime:     time.Since(hm.startTime),
		Components: components,
		Summary:    summary,
	}
}

// calculateOverallStatus 计算整体状态
func (hm *HealthManager) calculateOverallStatus(components map[string]ComponentHealth) HealthStatus {
	if len(components) == 0 {
		return StatusUnknown
	}

	hasUnhealthy := false
	hasDegraded := false

	for _, health := range components {
		switch health.Status {
		case StatusUnhealthy:
			hasUnhealthy = true
		case StatusDegraded:
			hasDegraded = true
		}
	}

	if hasUnhealthy {
		return StatusUnhealthy
	}
	if hasDegraded {
		return StatusDegraded
	}
	return StatusHealthy
}

// calculateSummary 计算摘要
func (hm *HealthManager) calculateSummary(components map[string]ComponentHealth) HealthSummary {
	summary := HealthSummary{
		TotalComponents: len(components),
	}

	for _, health := range components {
		switch health.Status {
		case StatusHealthy:
			summary.HealthyComponents++
		case StatusDegraded:
			summary.DegradedComponents++
		case StatusUnhealthy:
			summary.UnhealthyComponents++
		}
	}

	return summary
}

// statusToFloat 将状态转换为浮点数
func (hm *HealthManager) statusToFloat(status HealthStatus) float64 {
	switch status {
	case StatusHealthy:
		return 1.0
	case StatusDegraded:
		return 0.5
	case StatusUnhealthy:
		return 0.0
	default:
		return -1.0
	}
}

// StartPeriodicChecks 启动定期健康检查
func (hm *HealthManager) StartPeriodicChecks(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	logrus.WithField("interval", interval).Info("Starting periodic health checks")

	// 立即执行一次检查
	hm.CheckAll(ctx)

	for {
		select {
		case <-ticker.C:
			hm.CheckAll(ctx)
		case <-ctx.Done():
			logrus.Info("Stopping periodic health checks")
			return
		}
	}
}

// DatabaseHealthChecker 数据库健康检查器
type DatabaseHealthChecker struct {
	name string
	db   *sql.DB
}

// NewDatabaseHealthChecker 创建数据库健康检查器
func NewDatabaseHealthChecker(name string, db *sql.DB) *DatabaseHealthChecker {
	return &DatabaseHealthChecker{
		name: name,
		db:   db,
	}
}

func (d *DatabaseHealthChecker) Name() string {
	return d.name
}

func (d *DatabaseHealthChecker) Check(ctx context.Context) ComponentHealth {
	start := time.Now()
	health := ComponentHealth{
		Name:        d.name,
		LastChecked: start,
		Metadata:    make(map[string]interface{}),
	}

	// 检查数据库连接
	if err := d.db.PingContext(ctx); err != nil {
		health.Status = StatusUnhealthy
		health.Message = fmt.Sprintf("Database ping failed: %v", err)
		return health
	}

	// 获取数据库统计信息
	stats := d.db.Stats()
	health.Metadata["open_connections"] = stats.OpenConnections
	health.Metadata["in_use"] = stats.InUse
	health.Metadata["idle"] = stats.Idle
	health.Metadata["max_open_connections"] = stats.MaxOpenConnections

	// 检查连接池状态
	if stats.OpenConnections >= stats.MaxOpenConnections {
		health.Status = StatusDegraded
		health.Message = "Connection pool is at maximum capacity"
	} else {
		health.Status = StatusHealthy
		health.Message = "Database is healthy"
	}

	return health
}

// RedisHealthChecker Redis健康检查器
type RedisHealthChecker struct {
	name   string
	client *redis.Client
}

// NewRedisHealthChecker 创建Redis健康检查器
func NewRedisHealthChecker(name string, client *redis.Client) *RedisHealthChecker {
	return &RedisHealthChecker{
		name:   name,
		client: client,
	}
}

func (r *RedisHealthChecker) Name() string {
	return r.name
}

func (r *RedisHealthChecker) Check(ctx context.Context) ComponentHealth {
	start := time.Now()
	health := ComponentHealth{
		Name:        r.name,
		LastChecked: start,
		Metadata:    make(map[string]interface{}),
	}

	// 检查Redis连接
	if err := r.client.Ping(ctx).Err(); err != nil {
		health.Status = StatusUnhealthy
		health.Message = fmt.Sprintf("Redis ping failed: %v", err)
		return health
	}

	// 获取Redis信息
	info, err := r.client.Info(ctx, "memory", "clients").Result()
	if err != nil {
		health.Status = StatusDegraded
		health.Message = fmt.Sprintf("Failed to get Redis info: %v", err)
		return health
	}

	health.Metadata["info"] = info
	health.Status = StatusHealthy
	health.Message = "Redis is healthy"

	return health
}

// HTTPServiceHealthChecker HTTP服务健康检查器
type HTTPServiceHealthChecker struct {
	name   string
	url    string
	client *http.Client
}

// NewHTTPServiceHealthChecker 创建HTTP服务健康检查器
func NewHTTPServiceHealthChecker(name, url string) *HTTPServiceHealthChecker {
	return &HTTPServiceHealthChecker{
		name: name,
		url:  url,
		client: &http.Client{
			Timeout: 5 * time.Second,
		},
	}
}

func (h *HTTPServiceHealthChecker) Name() string {
	return h.name
}

func (h *HTTPServiceHealthChecker) Check(ctx context.Context) ComponentHealth {
	start := time.Now()
	health := ComponentHealth{
		Name:        h.name,
		LastChecked: start,
		Metadata:    make(map[string]interface{}),
	}

	req, err := http.NewRequestWithContext(ctx, "GET", h.url, nil)
	if err != nil {
		health.Status = StatusUnhealthy
		health.Message = fmt.Sprintf("Failed to create request: %v", err)
		return health
	}

	resp, err := h.client.Do(req)
	if err != nil {
		health.Status = StatusUnhealthy
		health.Message = fmt.Sprintf("HTTP request failed: %v", err)
		return health
	}
	defer resp.Body.Close()

	health.Metadata["status_code"] = resp.StatusCode
	health.Metadata["url"] = h.url

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		health.Status = StatusHealthy
		health.Message = "HTTP service is healthy"
	} else if resp.StatusCode >= 500 {
		health.Status = StatusUnhealthy
		health.Message = fmt.Sprintf("HTTP service returned %d", resp.StatusCode)
	} else {
		health.Status = StatusDegraded
		health.Message = fmt.Sprintf("HTTP service returned %d", resp.StatusCode)
	}

	return health
}

// HealthHandler HTTP处理器
type HealthHandler struct {
	manager *HealthManager
}

// NewHealthHandler 创建健康检查HTTP处理器
func NewHealthHandler(manager *HealthManager) *HealthHandler {
	return &HealthHandler{manager: manager}
}

// HandleHealth 处理健康检查请求
func (h *HealthHandler) HandleHealth(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// 检查是否需要强制刷新
	forceRefresh := r.URL.Query().Get("refresh") == "true"

	var health SystemHealth
	if forceRefresh {
		health = h.manager.CheckAll(ctx)
	} else {
		health = h.manager.GetCachedHealth()
	}

	// 设置HTTP状态码
	statusCode := http.StatusOK
	switch health.Status {
	case StatusUnhealthy:
		statusCode = http.StatusServiceUnavailable
	case StatusDegraded:
		statusCode = http.StatusPartialContent
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)

	if err := json.NewEncoder(w).Encode(health); err != nil {
		logrus.WithError(err).Error("Failed to encode health response")
	}
}

// HandleLiveness 处理存活性检查
func (h *HealthHandler) HandleLiveness(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "alive",
		"timestamp": time.Now(),
	})
}

// HandleReadiness 处理就绪性检查
func (h *HealthHandler) HandleReadiness(w http.ResponseWriter, r *http.Request) {
	health := h.manager.GetCachedHealth()

	statusCode := http.StatusOK
	if health.Status == StatusUnhealthy {
		statusCode = http.StatusServiceUnavailable
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    health.Status,
		"ready":     health.Status != StatusUnhealthy,
		"timestamp": time.Now(),
	})
}
