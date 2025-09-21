package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// ServiceCategory represents different categories of third-party services
type ServiceCategory string

const (
	CategoryAPI       ServiceCategory = "api"
	CategoryDatabase  ServiceCategory = "database"
	CategoryMessaging ServiceCategory = "messaging"
	CategoryPayment   ServiceCategory = "payment"
	CategoryAnalytics ServiceCategory = "analytics"
	CategoryStorage   ServiceCategory = "storage"
	CategoryAuth      ServiceCategory = "auth"
	CategoryML        ServiceCategory = "ml"
	CategorySocial    ServiceCategory = "social"
	CategoryEcommerce ServiceCategory = "ecommerce"
)

// ServiceStatus represents the status of a service
type ServiceStatus string

const (
	StatusActive      ServiceStatus = "active"
	StatusInactive    ServiceStatus = "inactive"
	StatusError       ServiceStatus = "error"
	StatusMaintenance ServiceStatus = "maintenance"
)

// AuthType represents different authentication methods
type AuthType string

const (
	AuthNone   AuthType = "none"
	AuthAPIKey AuthType = "api_key"
	AuthOAuth2 AuthType = "oauth2"
	AuthJWT    AuthType = "jwt"
	AuthBasic  AuthType = "basic"
)

// ServiceConfig represents the configuration for a third-party service
type ServiceConfig struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Description string            `json:"description"`
	Category    ServiceCategory   `json:"category"`
	BaseURL     string            `json:"base_url"`
	Version     string            `json:"version"`
	AuthType    AuthType          `json:"auth_type"`
	AuthConfig  map[string]string `json:"auth_config"`
	Headers     map[string]string `json:"headers"`
	Timeout     time.Duration     `json:"timeout"`
	RateLimit   RateLimit         `json:"rate_limit"`
	Endpoints   []Endpoint        `json:"endpoints"`
	Status      ServiceStatus     `json:"status"`
	CreatedAt   time.Time         `json:"created_at"`
	UpdatedAt   time.Time         `json:"updated_at"`
}

// RateLimit represents rate limiting configuration
type RateLimit struct {
	RequestsPerSecond int           `json:"requests_per_second"`
	BurstSize         int           `json:"burst_size"`
	WindowSize        time.Duration `json:"window_size"`
}

// Endpoint represents a service endpoint
type Endpoint struct {
	Name        string            `json:"name"`
	Path        string            `json:"path"`
	Method      string            `json:"method"`
	Description string            `json:"description"`
	Parameters  []Parameter       `json:"parameters"`
	Headers     map[string]string `json:"headers"`
	Response    ResponseSchema    `json:"response"`
}

// Parameter represents an endpoint parameter
type Parameter struct {
	Name        string      `json:"name"`
	Type        string      `json:"type"`
	Required    bool        `json:"required"`
	Description string      `json:"description"`
	Default     interface{} `json:"default,omitempty"`
	Validation  string      `json:"validation,omitempty"`
}

// ResponseSchema represents the expected response structure
type ResponseSchema struct {
	ContentType string                 `json:"content_type"`
	Schema      map[string]interface{} `json:"schema"`
	Examples    []interface{}          `json:"examples"`
}

// ServiceRegistry manages third-party service integrations
type ServiceRegistry struct {
	services map[string]*ServiceConfig
	mu       sync.RWMutex
	logger   *logrus.Logger
}

// NewServiceRegistry creates a new service registry
func NewServiceRegistry() *ServiceRegistry {
	return &ServiceRegistry{
		services: make(map[string]*ServiceConfig),
		logger:   logrus.New(),
	}
}

// RegisterService registers a new third-party service
func (sr *ServiceRegistry) RegisterService(config *ServiceConfig) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	if config.ID == "" {
		return fmt.Errorf("service ID cannot be empty")
	}

	if config.Name == "" {
		return fmt.Errorf("service name cannot be empty")
	}

	if config.BaseURL == "" {
		return fmt.Errorf("service base URL cannot be empty")
	}

	// Set default values
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}

	if config.Status == "" {
		config.Status = StatusActive
	}

	config.CreatedAt = time.Now()
	config.UpdatedAt = time.Now()

	sr.services[config.ID] = config
	sr.logger.Infof("Registered service: %s (%s)", config.Name, config.ID)

	return nil
}

// GetService retrieves a service configuration by ID
func (sr *ServiceRegistry) GetService(serviceID string) (*ServiceConfig, error) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	service, exists := sr.services[serviceID]
	if !exists {
		return nil, fmt.Errorf("service not found: %s", serviceID)
	}

	return service, nil
}

// ListServices returns all registered services
func (sr *ServiceRegistry) ListServices() []*ServiceConfig {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	services := make([]*ServiceConfig, 0, len(sr.services))
	for _, service := range sr.services {
		services = append(services, service)
	}

	return services
}

// ListServicesByCategory returns services filtered by category
func (sr *ServiceRegistry) ListServicesByCategory(category ServiceCategory) []*ServiceConfig {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	var services []*ServiceConfig
	for _, service := range sr.services {
		if service.Category == category {
			services = append(services, service)
		}
	}

	return services
}

// UpdateService updates an existing service configuration
func (sr *ServiceRegistry) UpdateService(serviceID string, updates map[string]interface{}) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	service, exists := sr.services[serviceID]
	if !exists {
		return fmt.Errorf("service not found: %s", serviceID)
	}

	// Apply updates
	if name, ok := updates["name"].(string); ok {
		service.Name = name
	}
	if description, ok := updates["description"].(string); ok {
		service.Description = description
	}
	if baseURL, ok := updates["base_url"].(string); ok {
		service.BaseURL = baseURL
	}
	if status, ok := updates["status"].(ServiceStatus); ok {
		service.Status = status
	}

	service.UpdatedAt = time.Now()
	sr.logger.Infof("Updated service: %s (%s)", service.Name, serviceID)

	return nil
}

// RemoveService removes a service from the registry
func (sr *ServiceRegistry) RemoveService(serviceID string) error {
	sr.mu.Lock()
	defer sr.mu.Unlock()

	service, exists := sr.services[serviceID]
	if !exists {
		return fmt.Errorf("service not found: %s", serviceID)
	}

	delete(sr.services, serviceID)
	sr.logger.Infof("Removed service: %s (%s)", service.Name, serviceID)

	return nil
}

// HealthCheck performs health checks on all active services
func (sr *ServiceRegistry) HealthCheck(ctx context.Context) map[string]bool {
	sr.mu.RLock()
	services := make([]*ServiceConfig, 0, len(sr.services))
	for _, service := range sr.services {
		if service.Status == StatusActive {
			services = append(services, service)
		}
	}
	sr.mu.RUnlock()

	results := make(map[string]bool)
	var wg sync.WaitGroup

	for _, service := range services {
		wg.Add(1)
		go func(svc *ServiceConfig) {
			defer wg.Done()
			healthy := sr.checkServiceHealth(ctx, svc)
			results[svc.ID] = healthy
		}(service)
	}

	wg.Wait()
	return results
}

// checkServiceHealth performs a health check on a single service
func (sr *ServiceRegistry) checkServiceHealth(ctx context.Context, service *ServiceConfig) bool {
	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	// Try to make a simple GET request to the base URL
	req, err := http.NewRequestWithContext(ctx, "GET", service.BaseURL, nil)
	if err != nil {
		sr.logger.Errorf("Failed to create health check request for %s: %v", service.ID, err)
		return false
	}

	// Add authentication headers if needed
	sr.addAuthHeaders(req, service)

	resp, err := client.Do(req)
	if err != nil {
		sr.logger.Errorf("Health check failed for %s: %v", service.ID, err)
		return false
	}
	defer resp.Body.Close()

	// Consider 2xx and 3xx status codes as healthy
	return resp.StatusCode < 400
}

// addAuthHeaders adds authentication headers to the request
func (sr *ServiceRegistry) addAuthHeaders(req *http.Request, service *ServiceConfig) {
	switch service.AuthType {
	case AuthAPIKey:
		if apiKey, ok := service.AuthConfig["api_key"]; ok {
			if header, ok := service.AuthConfig["header"]; ok {
				req.Header.Set(header, apiKey)
			} else {
				req.Header.Set("X-API-Key", apiKey)
			}
		}
	case AuthBasic:
		if username, ok := service.AuthConfig["username"]; ok {
			if password, ok := service.AuthConfig["password"]; ok {
				req.SetBasicAuth(username, password)
			}
		}
	case AuthJWT:
		if token, ok := service.AuthConfig["token"]; ok {
			req.Header.Set("Authorization", "Bearer "+token)
		}
	}

	// Add custom headers
	for key, value := range service.Headers {
		req.Header.Set(key, value)
	}
}

// GetServiceStats returns statistics about registered services
func (sr *ServiceRegistry) GetServiceStats() map[string]interface{} {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	stats := map[string]interface{}{
		"total_services": len(sr.services),
		"by_category":    make(map[ServiceCategory]int),
		"by_status":      make(map[ServiceStatus]int),
	}

	categoryStats := stats["by_category"].(map[ServiceCategory]int)
	statusStats := stats["by_status"].(map[ServiceStatus]int)

	for _, service := range sr.services {
		categoryStats[service.Category]++
		statusStats[service.Status]++
	}

	return stats
}

// ExportServices exports all service configurations to JSON
func (sr *ServiceRegistry) ExportServices() ([]byte, error) {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	return json.MarshalIndent(sr.services, "", "  ")
}

// ImportServices imports service configurations from JSON
func (sr *ServiceRegistry) ImportServices(data []byte) error {
	var services map[string]*ServiceConfig
	if err := json.Unmarshal(data, &services); err != nil {
		return fmt.Errorf("failed to unmarshal services: %w", err)
	}

	sr.mu.Lock()
	defer sr.mu.Unlock()

	for id, service := range services {
		service.ID = id // Ensure ID matches the key
		sr.services[id] = service
	}

	sr.logger.Infof("Imported %d services", len(services))
	return nil
}

// LoadDefaultServices loads a set of popular third-party services
func (sr *ServiceRegistry) LoadDefaultServices() error {
	defaultServices := []*ServiceConfig{
		// API Services
		{
			ID:          "openai",
			Name:        "OpenAI API",
			Description: "OpenAI GPT and other AI models",
			Category:    CategoryML,
			BaseURL:     "https://api.openai.com/v1",
			Version:     "v1",
			AuthType:    AuthAPIKey,
			AuthConfig: map[string]string{
				"header": "Authorization",
				"prefix": "Bearer ",
			},
			Timeout: 60 * time.Second,
			RateLimit: RateLimit{
				RequestsPerSecond: 10,
				BurstSize:         20,
				WindowSize:        time.Minute,
			},
			Endpoints: []Endpoint{
				{
					Name:        "chat_completions",
					Path:        "/chat/completions",
					Method:      "POST",
					Description: "Create a chat completion",
				},
				{
					Name:        "embeddings",
					Path:        "/embeddings",
					Method:      "POST",
					Description: "Create embeddings",
				},
			},
			Status: StatusActive,
		},
		{
			ID:          "github",
			Name:        "GitHub API",
			Description: "GitHub REST API for repository management",
			Category:    CategoryAPI,
			BaseURL:     "https://api.github.com",
			Version:     "v3",
			AuthType:    AuthAPIKey,
			AuthConfig: map[string]string{
				"header": "Authorization",
				"prefix": "token ",
			},
			Timeout: 30 * time.Second,
			RateLimit: RateLimit{
				RequestsPerSecond: 50,
				BurstSize:         100,
				WindowSize:        time.Hour,
			},
			Status: StatusActive,
		},
		{
			ID:          "slack",
			Name:        "Slack API",
			Description: "Slack Web API for messaging and workspace management",
			Category:    CategoryMessaging,
			BaseURL:     "https://slack.com/api",
			Version:     "v1",
			AuthType:    AuthOAuth2,
			Timeout:     30 * time.Second,
			Status:      StatusActive,
		},
		{
			ID:          "stripe",
			Name:        "Stripe API",
			Description: "Stripe payment processing API",
			Category:    CategoryPayment,
			BaseURL:     "https://api.stripe.com/v1",
			Version:     "v1",
			AuthType:    AuthAPIKey,
			AuthConfig: map[string]string{
				"header": "Authorization",
				"prefix": "Bearer ",
			},
			Timeout: 30 * time.Second,
			Status:  StatusActive,
		},
		{
			ID:          "google_analytics",
			Name:        "Google Analytics API",
			Description: "Google Analytics Reporting API",
			Category:    CategoryAnalytics,
			BaseURL:     "https://analyticsreporting.googleapis.com/v4",
			Version:     "v4",
			AuthType:    AuthOAuth2,
			Timeout:     30 * time.Second,
			Status:      StatusActive,
		},
		{
			ID:          "aws_s3",
			Name:        "Amazon S3 API",
			Description: "Amazon Simple Storage Service API",
			Category:    CategoryStorage,
			BaseURL:     "https://s3.amazonaws.com",
			Version:     "v1",
			AuthType:    AuthAPIKey,
			Timeout:     60 * time.Second,
			Status:      StatusActive,
		},
		{
			ID:          "auth0",
			Name:        "Auth0 Management API",
			Description: "Auth0 identity and access management",
			Category:    CategoryAuth,
			BaseURL:     "https://YOUR_DOMAIN.auth0.com/api/v2",
			Version:     "v2",
			AuthType:    AuthJWT,
			Timeout:     30 * time.Second,
			Status:      StatusActive,
		},
		{
			ID:          "twitter",
			Name:        "Twitter API",
			Description: "Twitter API v2 for social media integration",
			Category:    CategorySocial,
			BaseURL:     "https://api.twitter.com/2",
			Version:     "v2",
			AuthType:    AuthOAuth2,
			Timeout:     30 * time.Second,
			Status:      StatusActive,
		},
		{
			ID:          "shopify",
			Name:        "Shopify API",
			Description: "Shopify Admin API for e-commerce",
			Category:    CategoryEcommerce,
			BaseURL:     "https://YOUR_SHOP.myshopify.com/admin/api/2023-01",
			Version:     "2023-01",
			AuthType:    AuthAPIKey,
			Timeout:     30 * time.Second,
			Status:      StatusActive,
		},
		{
			ID:          "sendgrid",
			Name:        "SendGrid API",
			Description: "SendGrid email delivery service",
			Category:    CategoryMessaging,
			BaseURL:     "https://api.sendgrid.com/v3",
			Version:     "v3",
			AuthType:    AuthAPIKey,
			AuthConfig: map[string]string{
				"header": "Authorization",
				"prefix": "Bearer ",
			},
			Timeout: 30 * time.Second,
			Status:  StatusActive,
		},
	}

	for _, service := range defaultServices {
		if err := sr.RegisterService(service); err != nil {
			sr.logger.Errorf("Failed to register default service %s: %v", service.ID, err)
		}
	}

	return nil
}

// SearchServices searches for services by name or description
func (sr *ServiceRegistry) SearchServices(query string) []*ServiceConfig {
	sr.mu.RLock()
	defer sr.mu.RUnlock()

	var results []*ServiceConfig
	for _, service := range sr.services {
		if sr.matchesQuery(service, query) {
			results = append(results, service)
		}
	}

	return results
}

// matchesQuery checks if a service matches the search query
func (sr *ServiceRegistry) matchesQuery(service *ServiceConfig, query string) bool {
	// Simple case-insensitive substring matching
	// In a real implementation, you might want to use more sophisticated search
	return contains(service.Name, query) ||
		contains(service.Description, query) ||
		contains(string(service.Category), query)
}

// contains performs case-insensitive substring matching
func contains(s, substr string) bool {
	return len(s) >= len(substr) &&
		(s == substr ||
			(len(s) > len(substr) &&
				(s[:len(substr)] == substr ||
					s[len(s)-len(substr):] == substr ||
					containsSubstring(s, substr))))
}

// containsSubstring checks if substr is contained in s
func containsSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
