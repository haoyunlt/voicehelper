package discovery

import (
	"context"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// ServiceRegistry manages discovered and registered services
type ServiceRegistry struct {
	services  map[string]*RegisteredService
	discovery *ServiceDiscovery
	mu        sync.RWMutex
	logger    *logrus.Logger

	// Health check configuration
	healthCheckInterval time.Duration
	healthCheckTimeout  time.Duration
	healthCheckEnabled  bool

	// Background workers
	healthCheckTicker *time.Ticker
	stopChan          chan struct{}
	wg                sync.WaitGroup
}

// RegisteredService represents a service in the registry
type RegisteredService struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	URL             string                 `json:"url"`
	Category        string                 `json:"category"`
	Status          ServiceStatus          `json:"status"`
	DiscoveryResult *DiscoveryResult       `json:"discovery_result,omitempty"`
	HealthStatus    HealthStatus           `json:"health_status"`
	LastHealthCheck time.Time              `json:"last_health_check"`
	RegisteredAt    time.Time              `json:"registered_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
	Metadata        map[string]interface{} `json:"metadata"`
	Tags            []string               `json:"tags"`

	// Load balancing and routing
	Weight   int `json:"weight"`
	Priority int `json:"priority"`

	// Statistics
	RequestCount    int64         `json:"request_count"`
	ErrorCount      int64         `json:"error_count"`
	LastRequestTime time.Time     `json:"last_request_time"`
	AverageLatency  time.Duration `json:"average_latency"`
}

// ServiceStatus represents the operational status of a service
type ServiceStatus string

const (
	StatusActive      ServiceStatus = "active"
	StatusInactive    ServiceStatus = "inactive"
	StatusError       ServiceStatus = "error"
	StatusMaintenance ServiceStatus = "maintenance"
	StatusDiscovering ServiceStatus = "discovering"
)

// HealthStatus represents the health status of a service
type HealthStatus string

const (
	HealthHealthy   HealthStatus = "healthy"
	HealthUnhealthy HealthStatus = "unhealthy"
	HealthUnknown   HealthStatus = "unknown"
	HealthTimeout   HealthStatus = "timeout"
)

// RegistryConfig contains configuration for the service registry
type RegistryConfig struct {
	HealthCheckInterval time.Duration
	HealthCheckTimeout  time.Duration
	HealthCheckEnabled  bool
	AutoDiscovery       bool
	DiscoveryMethods    []string
}

// NewServiceRegistry creates a new service registry
func NewServiceRegistry(config *RegistryConfig) *ServiceRegistry {
	if config == nil {
		config = &RegistryConfig{
			HealthCheckInterval: 30 * time.Second,
			HealthCheckTimeout:  10 * time.Second,
			HealthCheckEnabled:  true,
			AutoDiscovery:       true,
			DiscoveryMethods:    []string{"openapi", "swagger", "health", "info"},
		}
	}

	registry := &ServiceRegistry{
		services:            make(map[string]*RegisteredService),
		discovery:           NewServiceDiscovery(),
		logger:              logrus.New(),
		healthCheckInterval: config.HealthCheckInterval,
		healthCheckTimeout:  config.HealthCheckTimeout,
		healthCheckEnabled:  config.HealthCheckEnabled,
		stopChan:            make(chan struct{}),
	}

	// Start background health checks if enabled
	if config.HealthCheckEnabled {
		registry.startHealthChecker()
	}

	return registry
}

// RegisterService registers a new service in the registry
func (sr *ServiceRegistry) RegisterService(ctx context.Context, service *RegisteredService) error {
	if service.ID == "" {
		return fmt.Errorf("service ID cannot be empty")
	}
	if service.URL == "" {
		return fmt.Errorf("service URL cannot be empty")
	}

	sr.mu.Lock()
	defer sr.mu.Unlock()

	// Set default values
	if service.Name == "" {
		service.Name = service.ID
	}
	if service.Status == "" {
		service.Status = StatusDiscovering
	}
	if service.HealthStatus == "" {
		service.HealthStatus = HealthUnknown
	}
	if service.Metadata == nil {
		service.Metadata = make(map[string]interface{})
	}
	if service.Weight == 0 {
		service.Weight = 100
	}

	service.RegisteredAt = time.Now()
	service.UpdatedAt = time.Now()

	sr.services[service.ID] = service
	sr.logger.Infof("Registered service: %s (%s)", service.Name, service.ID)

	// Perform service discovery in background
	go sr.discoverServiceCapabilities(ctx, service.ID)

	return nil
}

// DiscoverAndRegisterService discovers a service and registers it
func (sr *ServiceRegistry) DiscoverAndRegisterService(ctx context.Context, serviceURL, serviceID string) (*RegisteredService, error) {
	if serviceID == "" {
		serviceID = fmt.Sprintf("service_%d", time.Now().Unix())
	}

	// Perform discovery
	discoveryRequest := &DiscoveryRequest{
		URL:              serviceURL,
		Timeout:          30 * time.Second,
		FollowRedirects:  true,
		DiscoveryMethods: []string{"openapi", "swagger", "health", "info", "root"},
	}

	discoveryResult, err := sr.discovery.DiscoverService(ctx, discoveryRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to discover service: %w", err)
	}

	// Create registered service from discovery result
	service := &RegisteredService{
		ID:              serviceID,
		Name:            discoveryResult.ServiceName,
		URL:             serviceURL,
		Status:          StatusActive,
		DiscoveryResult: discoveryResult,
		HealthStatus:    HealthUnknown,
		Metadata:        discoveryResult.Metadata,
		Weight:          100,
		Priority:        1,
	}

	// Infer category from capabilities or endpoints
	service.Category = sr.inferServiceCategory(discoveryResult)

	// Register the service
	if err := sr.RegisterService(ctx, service); err != nil {
		return nil, err
	}

	return service, nil
}

// GetService retrieves a service by ID
func (sr *ServiceRegistry) GetService(serviceID string) (*RegisteredService, error) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	service, exists := sr.services[serviceID]
	if !exists {
		return nil, fmt.Errorf("service not found: %s", serviceID)
	}

	return service, nil
}

// ListServices returns all registered services
func (sr *ServiceRegistry) ListServices() []*RegisteredService {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	services := make([]*RegisteredService, 0, len(sr.services))
	for _, service := range sr.services {
		services = append(services, service)
	}

	return services
}

// ListServicesByCategory returns services filtered by category
func (sr *ServiceRegistry) ListServicesByCategory(category string) []*RegisteredService {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	var services []*RegisteredService
	for _, service := range sr.services {
		if service.Category == category {
			services = append(services, service)
		}
	}

	return services
}

// ListHealthyServices returns only healthy services
func (sr *ServiceRegistry) ListHealthyServices() []*RegisteredService {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	var services []*RegisteredService
	for _, service := range sr.services {
		if service.HealthStatus == HealthHealthy && service.Status == StatusActive {
			services = append(services, service)
		}
	}

	return services
}

// UpdateService updates an existing service
func (sr *ServiceRegistry) UpdateService(serviceID string, updates map[string]interface{}) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	service, exists := sr.services[serviceID]
	if !exists {
		return fmt.Errorf("service not found: %s", serviceID)
	}

	// Apply updates
	for key, value := range updates {
		switch key {
		case "name":
			if name, ok := value.(string); ok {
				service.Name = name
			}
		case "status":
			if status, ok := value.(ServiceStatus); ok {
				service.Status = status
			}
		case "category":
			if category, ok := value.(string); ok {
				service.Category = category
			}
		case "weight":
			if weight, ok := value.(int); ok {
				service.Weight = weight
			}
		case "priority":
			if priority, ok := value.(int); ok {
				service.Priority = priority
			}
		case "tags":
			if tags, ok := value.([]string); ok {
				service.Tags = tags
			}
		default:
			// Store in metadata
			service.Metadata[key] = value
		}
	}

	service.UpdatedAt = time.Now()
	sr.logger.Infof("Updated service: %s", serviceID)

	return nil
}

// RemoveService removes a service from the registry
func (sr *ServiceRegistry) RemoveService(serviceID string) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	if _, exists := sr.services[serviceID]; !exists {
		return fmt.Errorf("service not found: %s", serviceID)
	}

	delete(sr.services, serviceID)
	sr.logger.Infof("Removed service: %s", serviceID)

	return nil
}

// CheckServiceHealth performs a health check on a specific service
func (sr *ServiceRegistry) CheckServiceHealth(ctx context.Context, serviceID string) (HealthStatus, error) {
	service, err := sr.GetService(serviceID)
	if err != nil {
		return HealthUnknown, err
	}

	health := sr.performHealthCheck(ctx, service)

	// Update service health status
	sr.mu.Lock()
	service.HealthStatus = health
	service.LastHealthCheck = time.Now()
	sr.mu.Unlock()

	return health, nil
}

// GetServiceStats returns statistics for a service
func (sr *ServiceRegistry) GetServiceStats(serviceID string) (map[string]interface{}, error) {
	service, err := sr.GetService(serviceID)
	if err != nil {
		return nil, err
	}

	sr.mu.RLock()
	defer sr.mu.RUnlock()

	stats := map[string]interface{}{
		"request_count":     service.RequestCount,
		"error_count":       service.ErrorCount,
		"last_request_time": service.LastRequestTime,
		"average_latency":   service.AverageLatency,
		"health_status":     service.HealthStatus,
		"last_health_check": service.LastHealthCheck,
		"uptime":            time.Since(service.RegisteredAt),
		"error_rate":        float64(service.ErrorCount) / float64(service.RequestCount),
	}

	return stats, nil
}

// RecordServiceRequest records a request to a service for statistics
func (sr *ServiceRegistry) RecordServiceRequest(serviceID string, latency time.Duration, success bool) {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	service, exists := sr.services[serviceID]
	if !exists {
		return
	}

	service.RequestCount++
	service.LastRequestTime = time.Now()

	if !success {
		service.ErrorCount++
	}

	// Update average latency (simple moving average)
	if service.AverageLatency == 0 {
		service.AverageLatency = latency
	} else {
		service.AverageLatency = (service.AverageLatency + latency) / 2
	}
}

// startHealthChecker starts the background health checker
func (sr *ServiceRegistry) startHealthChecker() {
	sr.healthCheckTicker = time.NewTicker(sr.healthCheckInterval)

	sr.wg.Add(1)
	go func() {
		defer sr.wg.Done()

		for {
			select {
			case <-sr.healthCheckTicker.C:
				sr.performHealthChecks()
			case <-sr.stopChan:
				return
			}
		}
	}()
}

// performHealthChecks performs health checks on all active services
func (sr *ServiceRegistry) performHealthChecks() {
	sr.mu.RLock()
	services := make([]*RegisteredService, 0, len(sr.services))
	for _, service := range sr.services {
		if service.Status == StatusActive {
			services = append(services, service)
		}
	}
	sr.mu.RUnlock()

	// Perform health checks concurrently
	var wg sync.WaitGroup
	for _, service := range services {
		wg.Add(1)
		go func(svc *RegisteredService) {
			defer wg.Done()

			ctx, cancel := context.WithTimeout(context.Background(), sr.healthCheckTimeout)
			defer cancel()

			health := sr.performHealthCheck(ctx, svc)

			sr.mu.Lock()
			svc.HealthStatus = health
			svc.LastHealthCheck = time.Now()
			sr.mu.Unlock()
		}(service)
	}

	wg.Wait()
}

// performHealthCheck performs a health check on a single service
func (sr *ServiceRegistry) performHealthCheck(ctx context.Context, service *RegisteredService) HealthStatus {
	// Use discovered health check endpoint if available
	healthPath := "/health"
	if service.DiscoveryResult != nil && service.DiscoveryResult.HealthCheck != nil {
		healthPath = service.DiscoveryResult.HealthCheck.Path
	}

	healthURL := service.URL + healthPath

	client := &http.Client{Timeout: sr.healthCheckTimeout}
	req, err := http.NewRequestWithContext(ctx, "GET", healthURL, nil)
	if err != nil {
		sr.logger.Errorf("Failed to create health check request for %s: %v", service.ID, err)
		return HealthUnhealthy
	}

	resp, err := client.Do(req)
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return HealthTimeout
		}
		return HealthUnhealthy
	}
	defer resp.Body.Close()

	// Consider 2xx status codes as healthy
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return HealthHealthy
	}

	return HealthUnhealthy
}

// discoverServiceCapabilities performs service discovery for a registered service
func (sr *ServiceRegistry) discoverServiceCapabilities(ctx context.Context, serviceID string) {
	service, err := sr.GetService(serviceID)
	if err != nil {
		sr.logger.Errorf("Failed to get service for discovery: %v", err)
		return
	}

	discoveryRequest := &DiscoveryRequest{
		URL:              service.URL,
		Timeout:          30 * time.Second,
		FollowRedirects:  true,
		DiscoveryMethods: []string{"openapi", "swagger", "health", "info", "root"},
	}

	discoveryResult, err := sr.discovery.DiscoverService(ctx, discoveryRequest)
	if err != nil {
		sr.logger.Errorf("Failed to discover service capabilities for %s: %v", serviceID, err)
		sr.mu.Lock()
		service.Status = StatusError
		sr.mu.Unlock()
		return
	}

	// Update service with discovery results
	sr.mu.Lock()
	service.DiscoveryResult = discoveryResult
	service.Status = StatusActive
	if service.Name == "" || service.Name == serviceID {
		service.Name = discoveryResult.ServiceName
	}
	if service.Category == "" {
		service.Category = sr.inferServiceCategory(discoveryResult)
	}
	// Merge metadata
	for key, value := range discoveryResult.Metadata {
		service.Metadata[key] = value
	}
	service.UpdatedAt = time.Now()
	sr.mu.Unlock()

	sr.logger.Infof("Discovered capabilities for service %s: %d endpoints", serviceID, len(discoveryResult.Endpoints))
}

// inferServiceCategory infers service category from discovery results
func (sr *ServiceRegistry) inferServiceCategory(result *DiscoveryResult) string {
	// Check capabilities
	for _, capability := range result.Capabilities {
		switch capability {
		case "openapi", "swagger":
			return "api"
		}
	}

	// Check endpoints for patterns
	for _, endpoint := range result.Endpoints {
		path := endpoint.Path
		if strings.Contains(path, "/auth") || strings.Contains(path, "/login") {
			return "auth"
		}
		if strings.Contains(path, "/payment") || strings.Contains(path, "/billing") {
			return "payment"
		}
		if strings.Contains(path, "/storage") || strings.Contains(path, "/file") {
			return "storage"
		}
		if strings.Contains(path, "/message") || strings.Contains(path, "/notification") {
			return "messaging"
		}
	}

	// Check service name for hints
	name := strings.ToLower(result.ServiceName)
	if strings.Contains(name, "auth") {
		return "auth"
	}
	if strings.Contains(name, "payment") || strings.Contains(name, "billing") {
		return "payment"
	}
	if strings.Contains(name, "storage") || strings.Contains(name, "file") {
		return "storage"
	}

	return "api" // Default category
}

// Stop stops the service registry and background workers
func (sr *ServiceRegistry) Stop() {
	close(sr.stopChan)

	if sr.healthCheckTicker != nil {
		sr.healthCheckTicker.Stop()
	}

	sr.wg.Wait()
	sr.logger.Info("Service registry stopped")
}
