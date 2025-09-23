package tests

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"voicehelper/backend/pkg/discovery"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestServiceDiscovery tests the service discovery functionality
func TestServiceDiscovery(t *testing.T) {
	// Create a mock service with OpenAPI spec
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/openapi.json":
			spec := map[string]interface{}{
				"openapi": "3.0.0",
				"info": map[string]interface{}{
					"title":       "Test Service",
					"version":     "1.0.0",
					"description": "A test service for discovery",
				},
				"paths": map[string]interface{}{
					"/users": map[string]interface{}{
						"get": map[string]interface{}{
							"summary": "Get users",
							"tags":    []string{"users"},
						},
					},
					"/health": map[string]interface{}{
						"get": map[string]interface{}{
							"summary": "Health check",
							"tags":    []string{"system"},
						},
					},
				},
			}
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(spec)
		case "/health":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
		case "/info":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"name":        "Test Service",
				"version":     "1.0.0",
				"description": "A test service",
			})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer mockServer.Close()

	// Test service discovery
	serviceDiscovery := discovery.NewServiceDiscovery()

	request := &discovery.DiscoveryRequest{
		URL:              mockServer.URL,
		Timeout:          10 * time.Second,
		FollowRedirects:  true,
		DiscoveryMethods: []string{"openapi", "health", "info"},
	}

	ctx := context.Background()
	result, err := serviceDiscovery.DiscoverService(ctx, request)

	require.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "Test Service", result.ServiceName)
	assert.Equal(t, "1.0.0", result.Version)
	assert.Contains(t, []string{"A test service for discovery", "A test service"}, result.Description)
	assert.Contains(t, result.Capabilities, "openapi")
	assert.Len(t, result.Endpoints, 2)
	assert.NotNil(t, result.HealthCheck)
	assert.Equal(t, "/health", result.HealthCheck.Path)
}

// TestServiceRegistry tests the service registry functionality
func TestServiceRegistry(t *testing.T) {
	registry := discovery.NewServiceRegistry(nil)

	// Test service registration
	service := &discovery.RegisteredService{
		ID:       "test-service-1",
		Name:     "Test Service",
		URL:      "http://localhost:8080",
		Category: "api",
		Status:   discovery.StatusActive,
		Weight:   100,
		Priority: 1,
	}

	ctx := context.Background()
	err := registry.RegisterService(ctx, service)
	require.NoError(t, err)

	// Test service retrieval
	retrievedService, err := registry.GetService("test-service-1")
	require.NoError(t, err)
	assert.Equal(t, "Test Service", retrievedService.Name)
	assert.Equal(t, "api", retrievedService.Category)

	// Test service listing
	services := registry.ListServices()
	assert.Len(t, services, 1)
	assert.Equal(t, "test-service-1", services[0].ID)

	// Test service listing by category
	apiServices := registry.ListServicesByCategory("api")
	assert.Len(t, apiServices, 1)

	dbServices := registry.ListServicesByCategory("database")
	assert.Len(t, dbServices, 0)

	// Test service update
	updates := map[string]interface{}{
		"name":   "Updated Test Service",
		"weight": 200,
	}
	err = registry.UpdateService("test-service-1", updates)
	require.NoError(t, err)

	updatedService, err := registry.GetService("test-service-1")
	require.NoError(t, err)
	assert.Equal(t, "Updated Test Service", updatedService.Name)
	assert.Equal(t, 200, updatedService.Weight)

	// Test service removal
	err = registry.RemoveService("test-service-1")
	require.NoError(t, err)

	_, err = registry.GetService("test-service-1")
	assert.Error(t, err)
}

// TestLoadBalancer tests the load balancer functionality
func TestLoadBalancer(t *testing.T) {
	registry := discovery.NewServiceRegistry(nil)

	// Register multiple services
	services := []*discovery.RegisteredService{
		{
			ID:           "service-1",
			Name:         "Service 1",
			URL:          "http://service1:8080",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthHealthy,
			Weight:       100,
			Priority:     1,
		},
		{
			ID:           "service-2",
			Name:         "Service 2",
			URL:          "http://service2:8080",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthHealthy,
			Weight:       200,
			Priority:     2,
		},
		{
			ID:           "service-3",
			Name:         "Service 3",
			URL:          "http://service3:8080",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthUnhealthy,
			Weight:       150,
			Priority:     1,
		},
	}

	ctx := context.Background()
	for _, service := range services {
		err := registry.RegisterService(ctx, service)
		require.NoError(t, err)
	}

	// Test load balancer with healthy first strategy
	config := &discovery.LoadBalancerConfig{
		Strategy:             discovery.StrategyHealthyFirst,
		EnableCircuitBreaker: true,
	}
	loadBalancer := discovery.NewLoadBalancer(registry, config)

	// Test service selection
	selection, err := loadBalancer.SelectService("api", nil)
	require.NoError(t, err)
	assert.NotNil(t, selection)
	assert.Contains(t, []string{"service-1", "service-2"}, selection.Service.ID) // Should select healthy service

	// Test multiple service selection
	selections, err := loadBalancer.SelectMultipleServices("api", 2, nil)
	require.NoError(t, err)
	assert.Len(t, selections, 2)

	// Test circuit breaker
	loadBalancer.RecordFailure("service-1")
	loadBalancer.RecordFailure("service-1")
	loadBalancer.RecordFailure("service-1")
	loadBalancer.RecordFailure("service-1")
	loadBalancer.RecordFailure("service-1") // Should open circuit

	status := loadBalancer.GetCircuitBreakerStatus()
	assert.Equal(t, discovery.CircuitOpen, status["service-1"])

	// Test service selection with circuit breaker open
	selection, err = loadBalancer.SelectService("api", nil)
	require.NoError(t, err)
	assert.Equal(t, "service-2", selection.Service.ID) // Should avoid service-1

	// Test circuit breaker reset
	loadBalancer.ResetCircuitBreaker("service-1")
	status = loadBalancer.GetCircuitBreakerStatus()
	assert.Equal(t, discovery.CircuitClosed, status["service-1"])
}

// TestDiscoveryIntegration tests the integration between discovery and registry
func TestDiscoveryIntegration(t *testing.T) {
	// Create a mock service
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/health":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
		case "/info":
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]interface{}{
				"name":        "Auto Discovered Service",
				"version":     "2.0.0",
				"description": "Automatically discovered service",
			})
		default:
			w.WriteHeader(http.StatusOK)
			w.Write([]byte("OK"))
		}
	}))
	defer mockServer.Close()

	// Test automatic discovery and registration
	registry := discovery.NewServiceRegistry(nil)

	ctx := context.Background()
	service, err := registry.DiscoverAndRegisterService(ctx, mockServer.URL, "auto-service")
	require.NoError(t, err)
	assert.NotNil(t, service)
	assert.Equal(t, "auto-service", service.ID)
	assert.Equal(t, "Auto Discovered Service", service.Name)

	// Verify service is registered
	retrievedService, err := registry.GetService("auto-service")
	require.NoError(t, err)
	assert.Equal(t, "Auto Discovered Service", retrievedService.Name)
	assert.NotNil(t, retrievedService.DiscoveryResult)
}

// TestHealthChecks tests the health check functionality
func TestHealthChecks(t *testing.T) {
	// Create a mock service that responds to health checks
	healthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/health" {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer healthyServer.Close()

	// Create a mock service that fails health checks
	unhealthyServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer unhealthyServer.Close()

	registry := discovery.NewServiceRegistry(&discovery.RegistryConfig{
		HealthCheckEnabled:  true,
		HealthCheckInterval: 100 * time.Millisecond,
		HealthCheckTimeout:  1 * time.Second,
	})

	// Register services
	ctx := context.Background()
	healthyService := &discovery.RegisteredService{
		ID:     "healthy-service",
		Name:   "Healthy Service",
		URL:    healthyServer.URL,
		Status: discovery.StatusActive,
	}
	err := registry.RegisterService(ctx, healthyService)
	require.NoError(t, err)

	unhealthyService := &discovery.RegisteredService{
		ID:     "unhealthy-service",
		Name:   "Unhealthy Service",
		URL:    unhealthyServer.URL,
		Status: discovery.StatusActive,
	}
	err = registry.RegisterService(ctx, unhealthyService)
	require.NoError(t, err)

	// Wait for health checks to run
	time.Sleep(200 * time.Millisecond)

	// Check health status
	healthyServices := registry.ListHealthyServices()
	assert.Len(t, healthyServices, 1)
	assert.Equal(t, "healthy-service", healthyServices[0].ID)

	// Test manual health check
	health, err := registry.CheckServiceHealth(ctx, "healthy-service")
	require.NoError(t, err)
	assert.Equal(t, discovery.HealthHealthy, health)

	health, err = registry.CheckServiceHealth(ctx, "unhealthy-service")
	require.NoError(t, err)
	assert.Equal(t, discovery.HealthUnhealthy, health)
}

// BenchmarkServiceSelection benchmarks service selection performance
func BenchmarkServiceSelection(b *testing.B) {
	registry := discovery.NewServiceRegistry(nil)

	// Register many services
	ctx := context.Background()
	for i := 0; i < 100; i++ {
		service := &discovery.RegisteredService{
			ID:           fmt.Sprintf("service-%d", i),
			Name:         fmt.Sprintf("Service %d", i),
			URL:          fmt.Sprintf("http://service%d:8080", i),
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthHealthy,
			Weight:       100 + i,
			Priority:     1,
		}
		registry.RegisterService(ctx, service)
	}

	loadBalancer := discovery.NewLoadBalancer(registry, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := loadBalancer.SelectService("api", nil)
		if err != nil {
			b.Fatal(err)
		}
	}
}
