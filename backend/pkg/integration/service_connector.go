package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// RequestOptions represents options for making API requests
type RequestOptions struct {
	Method      string            `json:"method"`
	Endpoint    string            `json:"endpoint"`
	Headers     map[string]string `json:"headers"`
	QueryParams map[string]string `json:"query_params"`
	Body        interface{}       `json:"body"`
	Timeout     time.Duration     `json:"timeout"`
}

// Response represents the response from a third-party service
type Response struct {
	StatusCode int                    `json:"status_code"`
	Headers    map[string][]string    `json:"headers"`
	Body       map[string]interface{} `json:"body"`
	RawBody    []byte                 `json:"raw_body"`
	Duration   time.Duration          `json:"duration"`
	Error      string                 `json:"error,omitempty"`
}

// ServiceConnector handles connections and requests to third-party services
type ServiceConnector struct {
	registry     *ServiceRegistry
	httpClient   *http.Client
	rateLimiters map[string]interface{}
	mu           sync.RWMutex
	logger       *logrus.Logger
	metrics      *ConnectorMetrics
}

// ConnectorMetrics tracks metrics for service connections
type ConnectorMetrics struct {
	TotalRequests   int64                    `json:"total_requests"`
	SuccessRequests int64                    `json:"success_requests"`
	ErrorRequests   int64                    `json:"error_requests"`
	ServiceMetrics  map[string]*ServiceStats `json:"service_metrics"`
	mu              sync.RWMutex
}

// ServiceStats tracks statistics for individual services
type ServiceStats struct {
	RequestCount    int64         `json:"request_count"`
	SuccessCount    int64         `json:"success_count"`
	ErrorCount      int64         `json:"error_count"`
	AverageLatency  time.Duration `json:"average_latency"`
	LastRequestTime time.Time     `json:"last_request_time"`
	TotalLatency    time.Duration `json:"total_latency"`
}

// NewServiceConnector creates a new service connector
func NewServiceConnector(registry *ServiceRegistry) *ServiceConnector {
	return &ServiceConnector{
		registry: registry,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
		rateLimiters: make(map[string]interface{}),
		logger:       logrus.New(),
		metrics: &ConnectorMetrics{
			ServiceMetrics: make(map[string]*ServiceStats),
		},
	}
}

// MakeRequest makes a request to a third-party service
func (sc *ServiceConnector) MakeRequest(ctx context.Context, serviceID string, options RequestOptions) (*Response, error) {
	startTime := time.Now()

	// Get service configuration
	service, err := sc.registry.GetService(serviceID)
	if err != nil {
		return nil, fmt.Errorf("service not found: %w", err)
	}

	// Check service status
	if service.Status != StatusActive {
		return nil, fmt.Errorf("service %s is not active (status: %s)", serviceID, service.Status)
	}

	// Apply rate limiting
	if err := sc.applyRateLimit(serviceID, service); err != nil {
		return nil, fmt.Errorf("rate limit exceeded: %w", err)
	}

	// Build request URL
	requestURL, err := sc.buildRequestURL(service, options)
	if err != nil {
		return nil, fmt.Errorf("failed to build request URL: %w", err)
	}

	// Prepare request body
	var bodyReader io.Reader
	if options.Body != nil {
		bodyBytes, err := json.Marshal(options.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		bodyReader = bytes.NewReader(bodyBytes)
	}

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, options.Method, requestURL, bodyReader)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	sc.setRequestHeaders(req, service, options)

	// Set timeout
	timeout := options.Timeout
	if timeout == 0 {
		timeout = service.Timeout
	}

	client := &http.Client{
		Timeout:   timeout,
		Transport: sc.httpClient.Transport,
	}

	// Make the request
	resp, err := client.Do(req)
	if err != nil {
		sc.recordMetrics(serviceID, false, time.Since(startTime))
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	// Read response body
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		sc.recordMetrics(serviceID, false, time.Since(startTime))
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Parse response
	response := &Response{
		StatusCode: resp.StatusCode,
		Headers:    resp.Header,
		RawBody:    bodyBytes,
		Duration:   time.Since(startTime),
	}

	// Try to parse JSON body
	if len(bodyBytes) > 0 {
		var jsonBody map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &jsonBody); err == nil {
			response.Body = jsonBody
		}
	}

	// Check for errors
	success := resp.StatusCode >= 200 && resp.StatusCode < 300
	if !success {
		response.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyBytes))
	}

	sc.recordMetrics(serviceID, success, response.Duration)
	return response, nil
}

// buildRequestURL builds the complete request URL
func (sc *ServiceConnector) buildRequestURL(service *ServiceConfig, options RequestOptions) (string, error) {
	baseURL := strings.TrimRight(service.BaseURL, "/")
	endpoint := strings.TrimLeft(options.Endpoint, "/")

	requestURL := fmt.Sprintf("%s/%s", baseURL, endpoint)

	// Add query parameters
	if len(options.QueryParams) > 0 {
		u, err := url.Parse(requestURL)
		if err != nil {
			return "", err
		}

		q := u.Query()
		for key, value := range options.QueryParams {
			q.Set(key, value)
		}
		u.RawQuery = q.Encode()
		requestURL = u.String()
	}

	return requestURL, nil
}

// setRequestHeaders sets the appropriate headers for the request
func (sc *ServiceConnector) setRequestHeaders(req *http.Request, service *ServiceConfig, options RequestOptions) {
	// Set default headers
	req.Header.Set("User-Agent", "Chatbot-Integration/1.0")
	req.Header.Set("Accept", "application/json")

	if options.Body != nil {
		req.Header.Set("Content-Type", "application/json")
	}

	// Set service-specific headers
	for key, value := range service.Headers {
		req.Header.Set(key, value)
	}

	// Set request-specific headers
	for key, value := range options.Headers {
		req.Header.Set(key, value)
	}

	// Set authentication headers
	sc.setAuthHeaders(req, service)
}

// setAuthHeaders sets authentication headers based on service configuration
func (sc *ServiceConnector) setAuthHeaders(req *http.Request, service *ServiceConfig) {
	switch service.AuthType {
	case AuthAPIKey:
		if apiKey, ok := service.AuthConfig["api_key"]; ok {
			header := service.AuthConfig["header"]
			if header == "" {
				header = "X-API-Key"
			}

			prefix := service.AuthConfig["prefix"]
			value := prefix + apiKey
			req.Header.Set(header, value)
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
	case AuthOAuth2:
		if accessToken, ok := service.AuthConfig["access_token"]; ok {
			req.Header.Set("Authorization", "Bearer "+accessToken)
		}
	}
}

// applyRateLimit applies rate limiting for the service
func (sc *ServiceConnector) applyRateLimit(serviceID string, serviceConfig *ServiceConfig) error {
	sc.mu.Lock()
	_, exists := sc.rateLimiters[serviceID]
	if !exists {
		// Simple rate limiting implementation without external dependencies
		// In production, you would use a proper rate limiter
		sc.rateLimiters[serviceID] = nil // Placeholder
	}
	sc.mu.Unlock()

	// Simple delay-based rate limiting
	rps := serviceConfig.RateLimit.RequestsPerSecond
	if rps > 0 {
		delay := time.Second / time.Duration(rps)
		time.Sleep(delay)
	}

	return nil
}

// recordMetrics records metrics for the request
func (sc *ServiceConnector) recordMetrics(serviceID string, success bool, duration time.Duration) {
	sc.metrics.mu.Lock()
	defer sc.metrics.mu.Unlock()

	// Update global metrics
	sc.metrics.TotalRequests++
	if success {
		sc.metrics.SuccessRequests++
	} else {
		sc.metrics.ErrorRequests++
	}

	// Update service-specific metrics
	stats, exists := sc.metrics.ServiceMetrics[serviceID]
	if !exists {
		stats = &ServiceStats{}
		sc.metrics.ServiceMetrics[serviceID] = stats
	}

	stats.RequestCount++
	stats.LastRequestTime = time.Now()
	stats.TotalLatency += duration
	stats.AverageLatency = stats.TotalLatency / time.Duration(stats.RequestCount)

	if success {
		stats.SuccessCount++
	} else {
		stats.ErrorCount++
	}
}

// GetMetrics returns connector metrics
func (sc *ServiceConnector) GetMetrics() *ConnectorMetrics {
	sc.metrics.mu.RLock()
	defer sc.metrics.mu.RUnlock()

	// Create a copy to avoid race conditions
	metrics := &ConnectorMetrics{
		TotalRequests:   sc.metrics.TotalRequests,
		SuccessRequests: sc.metrics.SuccessRequests,
		ErrorRequests:   sc.metrics.ErrorRequests,
		ServiceMetrics:  make(map[string]*ServiceStats),
	}

	for serviceID, stats := range sc.metrics.ServiceMetrics {
		metrics.ServiceMetrics[serviceID] = &ServiceStats{
			RequestCount:    stats.RequestCount,
			SuccessCount:    stats.SuccessCount,
			ErrorCount:      stats.ErrorCount,
			AverageLatency:  stats.AverageLatency,
			LastRequestTime: stats.LastRequestTime,
			TotalLatency:    stats.TotalLatency,
		}
	}

	return metrics
}

// BatchRequest represents a batch of requests to potentially different services
type BatchRequest struct {
	ID        string         `json:"id"`
	ServiceID string         `json:"service_id"`
	Options   RequestOptions `json:"options"`
}

// BatchResponse represents the response from a batch request
type BatchResponse struct {
	ID       string    `json:"id"`
	Response *Response `json:"response"`
	Error    string    `json:"error,omitempty"`
}

// MakeBatchRequests makes multiple requests concurrently
func (sc *ServiceConnector) MakeBatchRequests(ctx context.Context, requests []BatchRequest) []BatchResponse {
	responses := make([]BatchResponse, len(requests))
	var wg sync.WaitGroup

	for i, req := range requests {
		wg.Add(1)
		go func(index int, batchReq BatchRequest) {
			defer wg.Done()

			resp, err := sc.MakeRequest(ctx, batchReq.ServiceID, batchReq.Options)

			batchResp := BatchResponse{
				ID:       batchReq.ID,
				Response: resp,
			}

			if err != nil {
				batchResp.Error = err.Error()
			}

			responses[index] = batchResp
		}(i, req)
	}

	wg.Wait()
	return responses
}

// TestConnection tests the connection to a service
func (sc *ServiceConnector) TestConnection(ctx context.Context, serviceID string) error {
	_, err := sc.registry.GetService(serviceID)
	if err != nil {
		return err
	}

	// Make a simple GET request to test connectivity
	options := RequestOptions{
		Method:   "GET",
		Endpoint: "/",
		Timeout:  10 * time.Second,
	}

	_, err = sc.MakeRequest(ctx, serviceID, options)
	return err
}

// GetServiceHealth returns health status for all services
func (sc *ServiceConnector) GetServiceHealth(ctx context.Context) map[string]bool {
	services := sc.registry.ListServices()
	health := make(map[string]bool)
	var wg sync.WaitGroup

	for _, service := range services {
		if service.Status != StatusActive {
			health[service.ID] = false
			continue
		}

		wg.Add(1)
		go func(svc *ServiceConfig) {
			defer wg.Done()

			err := sc.TestConnection(ctx, svc.ID)
			health[svc.ID] = err == nil
		}(service)
	}

	wg.Wait()
	return health
}

// ClearMetrics clears all recorded metrics
func (sc *ServiceConnector) ClearMetrics() {
	sc.metrics.mu.Lock()
	defer sc.metrics.mu.Unlock()

	sc.metrics.TotalRequests = 0
	sc.metrics.SuccessRequests = 0
	sc.metrics.ErrorRequests = 0
	sc.metrics.ServiceMetrics = make(map[string]*ServiceStats)
}

// GetServiceEndpoints returns available endpoints for a service
func (sc *ServiceConnector) GetServiceEndpoints(serviceID string) ([]Endpoint, error) {
	service, err := sc.registry.GetService(serviceID)
	if err != nil {
		return nil, err
	}

	return service.Endpoints, nil
}

// ValidateRequest validates a request against service configuration
func (sc *ServiceConnector) ValidateRequest(serviceID string, options RequestOptions) error {
	service, err := sc.registry.GetService(serviceID)
	if err != nil {
		return err
	}

	// Check if endpoint exists
	var endpoint *Endpoint
	for _, ep := range service.Endpoints {
		if ep.Path == options.Endpoint || ep.Name == options.Endpoint {
			endpoint = &ep
			break
		}
	}

	if endpoint == nil {
		return fmt.Errorf("endpoint not found: %s", options.Endpoint)
	}

	// Validate HTTP method
	if endpoint.Method != "" && endpoint.Method != options.Method {
		return fmt.Errorf("invalid method %s for endpoint %s, expected %s",
			options.Method, options.Endpoint, endpoint.Method)
	}

	// Validate required parameters (basic validation)
	if options.Body != nil {
		bodyMap, ok := options.Body.(map[string]interface{})
		if ok {
			for _, param := range endpoint.Parameters {
				if param.Required {
					if _, exists := bodyMap[param.Name]; !exists {
						return fmt.Errorf("required parameter missing: %s", param.Name)
					}
				}
			}
		}
	}

	return nil
}
