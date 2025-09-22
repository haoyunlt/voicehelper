package discovery

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// ServiceDiscovery handles automatic discovery of service capabilities
type ServiceDiscovery struct {
	client *http.Client
	logger *logrus.Logger
	mu     sync.RWMutex
	cache  map[string]*DiscoveryResult
}

// DiscoveryResult contains the discovered service information
type DiscoveryResult struct {
	ServiceURL   string                 `json:"service_url"`
	ServiceName  string                 `json:"service_name"`
	Version      string                 `json:"version"`
	Description  string                 `json:"description"`
	Capabilities []string               `json:"capabilities"`
	Endpoints    []EndpointInfo         `json:"endpoints"`
	HealthCheck  *HealthCheckInfo       `json:"health_check,omitempty"`
	Metadata     map[string]interface{} `json:"metadata"`
	DiscoveredAt time.Time              `json:"discovered_at"`
	LastChecked  time.Time              `json:"last_checked"`
	Status       string                 `json:"status"`
}

// EndpointInfo contains information about a discovered endpoint
type EndpointInfo struct {
	Path        string                 `json:"path"`
	Method      string                 `json:"method"`
	Description string                 `json:"description"`
	Parameters  []ParameterInfo        `json:"parameters,omitempty"`
	Responses   []ResponseInfo         `json:"responses,omitempty"`
	Tags        []string               `json:"tags,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ParameterInfo contains parameter information
type ParameterInfo struct {
	Name        string      `json:"name"`
	Type        string      `json:"type"`
	Required    bool        `json:"required"`
	Description string      `json:"description"`
	Default     interface{} `json:"default,omitempty"`
}

// ResponseInfo contains response information
type ResponseInfo struct {
	StatusCode  int                    `json:"status_code"`
	Description string                 `json:"description"`
	ContentType string                 `json:"content_type"`
	Schema      map[string]interface{} `json:"schema,omitempty"`
}

// HealthCheckInfo contains health check configuration
type HealthCheckInfo struct {
	Path     string        `json:"path"`
	Method   string        `json:"method"`
	Interval time.Duration `json:"interval"`
	Timeout  time.Duration `json:"timeout"`
}

// DiscoveryRequest represents a service discovery request
type DiscoveryRequest struct {
	URL                string            `json:"url"`
	Headers            map[string]string `json:"headers,omitempty"`
	Timeout            time.Duration     `json:"timeout,omitempty"`
	FollowRedirects    bool              `json:"follow_redirects,omitempty"`
	DiscoveryMethods   []string          `json:"discovery_methods,omitempty"`
	AuthToken          string            `json:"auth_token,omitempty"`
	CustomDiscoveryURL string            `json:"custom_discovery_url,omitempty"`
}

// NewServiceDiscovery creates a new service discovery instance
func NewServiceDiscovery() *ServiceDiscovery {
	return &ServiceDiscovery{
		client: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		logger: logrus.New(),
		cache:  make(map[string]*DiscoveryResult),
	}
}

// DiscoverService discovers service capabilities from a given URL
func (sd *ServiceDiscovery) DiscoverService(ctx context.Context, request *DiscoveryRequest) (*DiscoveryResult, error) {
	if request.URL == "" {
		return nil, fmt.Errorf("service URL is required")
	}

	// Check cache first
	sd.mu.RLock()
	if cached, exists := sd.cache[request.URL]; exists {
		// Return cached result if it's still fresh (within 5 minutes)
		if time.Since(cached.LastChecked) < 5*time.Minute {
			sd.mu.RUnlock()
			return cached, nil
		}
	}
	sd.mu.RUnlock()

	// Perform discovery
	result, err := sd.performDiscovery(ctx, request)
	if err != nil {
		return nil, err
	}

	// Cache the result
	sd.mu.Lock()
	sd.cache[request.URL] = result
	sd.mu.Unlock()

	return result, nil
}

// performDiscovery performs the actual service discovery
func (sd *ServiceDiscovery) performDiscovery(ctx context.Context, request *DiscoveryRequest) (*DiscoveryResult, error) {
	result := &DiscoveryResult{
		ServiceURL:   request.URL,
		Metadata:     make(map[string]interface{}),
		DiscoveredAt: time.Now(),
		LastChecked:  time.Now(),
		Status:       "unknown",
	}

	// Set default discovery methods if not specified
	methods := request.DiscoveryMethods
	if len(methods) == 0 {
		methods = []string{"openapi", "swagger", "health", "info", "root"}
	}

	// Try different discovery methods
	for _, method := range methods {
		switch method {
		case "openapi":
			if err := sd.discoverOpenAPI(ctx, request, result); err != nil {
				sd.logger.Debugf("OpenAPI discovery failed for %s: %v", request.URL, err)
			}
		case "swagger":
			if err := sd.discoverSwagger(ctx, request, result); err != nil {
				sd.logger.Debugf("Swagger discovery failed for %s: %v", request.URL, err)
			}
		case "health":
			if err := sd.discoverHealthCheck(ctx, request, result); err != nil {
				sd.logger.Debugf("Health check discovery failed for %s: %v", request.URL, err)
			}
		case "info":
			if err := sd.discoverServiceInfo(ctx, request, result); err != nil {
				sd.logger.Debugf("Service info discovery failed for %s: %v", request.URL, err)
			}
		case "root":
			if err := sd.discoverFromRoot(ctx, request, result); err != nil {
				sd.logger.Debugf("Root discovery failed for %s: %v", request.URL, err)
			}
		}
	}

	// Determine overall status
	if len(result.Endpoints) > 0 || result.ServiceName != "" {
		result.Status = "discovered"
	} else {
		result.Status = "limited_info"
	}

	return result, nil
}

// discoverOpenAPI tries to discover service info from OpenAPI spec
func (sd *ServiceDiscovery) discoverOpenAPI(ctx context.Context, request *DiscoveryRequest, result *DiscoveryResult) error {
	paths := []string{"/openapi.json", "/openapi.yaml", "/v3/api-docs", "/api-docs"}

	for _, path := range paths {
		specURL := strings.TrimRight(request.URL, "/") + path

		resp, err := sd.makeRequest(ctx, "GET", specURL, request.Headers, request.AuthToken)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				continue
			}

			var spec map[string]interface{}
			if err := json.Unmarshal(body, &spec); err != nil {
				continue
			}

			sd.parseOpenAPISpec(spec, result)
			return nil
		}
	}

	return fmt.Errorf("no OpenAPI specification found")
}

// discoverSwagger tries to discover service info from Swagger spec
func (sd *ServiceDiscovery) discoverSwagger(ctx context.Context, request *DiscoveryRequest, result *DiscoveryResult) error {
	paths := []string{"/swagger.json", "/swagger.yaml", "/v2/api-docs", "/api/swagger.json"}

	for _, path := range paths {
		specURL := strings.TrimRight(request.URL, "/") + path

		resp, err := sd.makeRequest(ctx, "GET", specURL, request.Headers, request.AuthToken)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				continue
			}

			var spec map[string]interface{}
			if err := json.Unmarshal(body, &spec); err != nil {
				continue
			}

			sd.parseSwaggerSpec(spec, result)
			return nil
		}
	}

	return fmt.Errorf("no Swagger specification found")
}

// discoverHealthCheck tries to discover health check endpoint
func (sd *ServiceDiscovery) discoverHealthCheck(ctx context.Context, request *DiscoveryRequest, result *DiscoveryResult) error {
	healthPaths := []string{"/health", "/healthz", "/health/check", "/api/health", "/status"}

	for _, path := range healthPaths {
		healthURL := strings.TrimRight(request.URL, "/") + path

		resp, err := sd.makeRequest(ctx, "GET", healthURL, request.Headers, request.AuthToken)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			result.HealthCheck = &HealthCheckInfo{
				Path:     path,
				Method:   "GET",
				Interval: 30 * time.Second,
				Timeout:  10 * time.Second,
			}

			// Try to parse health response
			body, err := io.ReadAll(resp.Body)
			if err == nil {
				var healthData map[string]interface{}
				if json.Unmarshal(body, &healthData) == nil {
					result.Metadata["health_info"] = healthData
				}
			}

			return nil
		}
	}

	return fmt.Errorf("no health check endpoint found")
}

// discoverServiceInfo tries to discover service info from common info endpoints
func (sd *ServiceDiscovery) discoverServiceInfo(ctx context.Context, request *DiscoveryRequest, result *DiscoveryResult) error {
	infoPaths := []string{"/info", "/version", "/api/info", "/api/version", "/actuator/info"}

	for _, path := range infoPaths {
		infoURL := strings.TrimRight(request.URL, "/") + path

		resp, err := sd.makeRequest(ctx, "GET", infoURL, request.Headers, request.AuthToken)
		if err != nil {
			continue
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusOK {
			body, err := io.ReadAll(resp.Body)
			if err != nil {
				continue
			}

			var info map[string]interface{}
			if err := json.Unmarshal(body, &info); err != nil {
				continue
			}

			// Extract service information
			if name, ok := info["name"].(string); ok {
				result.ServiceName = name
			}
			if version, ok := info["version"].(string); ok {
				result.Version = version
			}
			if description, ok := info["description"].(string); ok {
				result.Description = description
			}

			result.Metadata["service_info"] = info
			return nil
		}
	}

	return fmt.Errorf("no service info endpoint found")
}

// discoverFromRoot tries to discover basic service info from root endpoint
func (sd *ServiceDiscovery) discoverFromRoot(ctx context.Context, request *DiscoveryRequest, result *DiscoveryResult) error {
	resp, err := sd.makeRequest(ctx, "GET", request.URL, request.Headers, request.AuthToken)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		// Extract basic info from response headers
		if server := resp.Header.Get("Server"); server != "" {
			result.Metadata["server"] = server
		}

		// Try to parse JSON response
		body, err := io.ReadAll(resp.Body)
		if err == nil {
			var rootData map[string]interface{}
			if json.Unmarshal(body, &rootData) == nil {
				result.Metadata["root_response"] = rootData

				// Try to extract service name from common fields
				for _, field := range []string{"name", "service", "title", "application"} {
					if name, ok := rootData[field].(string); ok && result.ServiceName == "" {
						result.ServiceName = name
						break
					}
				}
			}
		}

		return nil
	}

	return fmt.Errorf("root endpoint returned status %d", resp.StatusCode)
}

// parseOpenAPISpec parses OpenAPI specification and extracts service info
func (sd *ServiceDiscovery) parseOpenAPISpec(spec map[string]interface{}, result *DiscoveryResult) {
	// Extract basic info
	if info, ok := spec["info"].(map[string]interface{}); ok {
		if title, ok := info["title"].(string); ok {
			result.ServiceName = title
		}
		if version, ok := info["version"].(string); ok {
			result.Version = version
		}
		if description, ok := info["description"].(string); ok {
			result.Description = description
		}
	}

	// Extract paths/endpoints
	if paths, ok := spec["paths"].(map[string]interface{}); ok {
		for path, pathInfo := range paths {
			if pathData, ok := pathInfo.(map[string]interface{}); ok {
				for method, methodInfo := range pathData {
					if methodData, ok := methodInfo.(map[string]interface{}); ok {
						endpoint := EndpointInfo{
							Path:   path,
							Method: strings.ToUpper(method),
						}

						if summary, ok := methodData["summary"].(string); ok {
							endpoint.Description = summary
						}

						if tags, ok := methodData["tags"].([]interface{}); ok {
							for _, tag := range tags {
								if tagStr, ok := tag.(string); ok {
									endpoint.Tags = append(endpoint.Tags, tagStr)
								}
							}
						}

						result.Endpoints = append(result.Endpoints, endpoint)
					}
				}
			}
		}
	}

	result.Capabilities = append(result.Capabilities, "openapi")
}

// parseSwaggerSpec parses Swagger specification and extracts service info
func (sd *ServiceDiscovery) parseSwaggerSpec(spec map[string]interface{}, result *DiscoveryResult) {
	// Similar to OpenAPI but for Swagger 2.0
	if info, ok := spec["info"].(map[string]interface{}); ok {
		if title, ok := info["title"].(string); ok {
			result.ServiceName = title
		}
		if version, ok := info["version"].(string); ok {
			result.Version = version
		}
		if description, ok := info["description"].(string); ok {
			result.Description = description
		}
	}

	// Extract paths
	if paths, ok := spec["paths"].(map[string]interface{}); ok {
		for path, pathInfo := range paths {
			if pathData, ok := pathInfo.(map[string]interface{}); ok {
				for method, methodInfo := range pathData {
					if methodData, ok := methodInfo.(map[string]interface{}); ok {
						endpoint := EndpointInfo{
							Path:   path,
							Method: strings.ToUpper(method),
						}

						if summary, ok := methodData["summary"].(string); ok {
							endpoint.Description = summary
						}

						result.Endpoints = append(result.Endpoints, endpoint)
					}
				}
			}
		}
	}

	result.Capabilities = append(result.Capabilities, "swagger")
}

// makeRequest makes an HTTP request with proper headers and authentication
func (sd *ServiceDiscovery) makeRequest(ctx context.Context, method, url string, headers map[string]string, authToken string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, url, nil)
	if err != nil {
		return nil, err
	}

	// Add custom headers
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	// Add authentication
	if authToken != "" {
		req.Header.Set("Authorization", "Bearer "+authToken)
	}

	// Set default headers
	req.Header.Set("User-Agent", "VoiceHelper-ServiceDiscovery/1.0")
	req.Header.Set("Accept", "application/json, text/plain, */*")

	return sd.client.Do(req)
}

// GetCachedResult returns a cached discovery result
func (sd *ServiceDiscovery) GetCachedResult(serviceURL string) (*DiscoveryResult, bool) {
	sd.mu.RLock()
	defer sd.mu.RUnlock()

	result, exists := sd.cache[serviceURL]
	return result, exists
}

// ClearCache clears the discovery cache
func (sd *ServiceDiscovery) ClearCache() {
	sd.mu.Lock()
	defer sd.mu.Unlock()

	sd.cache = make(map[string]*DiscoveryResult)
}

// ListCachedServices returns all cached service URLs
func (sd *ServiceDiscovery) ListCachedServices() []string {
	sd.mu.RLock()
	defer sd.mu.RUnlock()

	urls := make([]string, 0, len(sd.cache))
	for url := range sd.cache {
		urls = append(urls, url)
	}

	return urls
}
