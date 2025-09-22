package discovery

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// LoadBalancer handles load balancing across discovered services
type LoadBalancer struct {
	registry *ServiceRegistry
	strategy LoadBalancingStrategy
	logger   *logrus.Logger
	mu       sync.RWMutex

	// Circuit breaker state
	circuitBreakers map[string]*CircuitBreaker
}

// LoadBalancingStrategy defines different load balancing strategies
type LoadBalancingStrategy string

const (
	StrategyRoundRobin     LoadBalancingStrategy = "round_robin"
	StrategyWeightedRandom LoadBalancingStrategy = "weighted_random"
	StrategyLeastLatency   LoadBalancingStrategy = "least_latency"
	StrategyHealthyFirst   LoadBalancingStrategy = "healthy_first"
	StrategyPriority       LoadBalancingStrategy = "priority"
)

// CircuitBreaker implements circuit breaker pattern for services
type CircuitBreaker struct {
	ServiceID        string
	FailureCount     int
	LastFailureTime  time.Time
	State            CircuitState
	FailureThreshold int
	RecoveryTimeout  time.Duration
	mu               sync.RWMutex
}

// CircuitState represents the state of a circuit breaker
type CircuitState string

const (
	CircuitClosed   CircuitState = "closed"    // Normal operation
	CircuitOpen     CircuitState = "open"      // Failing, reject requests
	CircuitHalfOpen CircuitState = "half_open" // Testing recovery
)

// ServiceSelection represents a selected service for load balancing
type ServiceSelection struct {
	Service *RegisteredService `json:"service"`
	Reason  string             `json:"reason"`
	Score   float64            `json:"score,omitempty"`
}

// LoadBalancerConfig contains configuration for the load balancer
type LoadBalancerConfig struct {
	Strategy             LoadBalancingStrategy
	FailureThreshold     int
	RecoveryTimeout      time.Duration
	EnableCircuitBreaker bool
}

// NewLoadBalancer creates a new load balancer
func NewLoadBalancer(registry *ServiceRegistry, config *LoadBalancerConfig) *LoadBalancer {
	if config == nil {
		config = &LoadBalancerConfig{
			Strategy:             StrategyHealthyFirst,
			FailureThreshold:     5,
			RecoveryTimeout:      30 * time.Second,
			EnableCircuitBreaker: true,
		}
	}

	return &LoadBalancer{
		registry:        registry,
		strategy:        config.Strategy,
		logger:          logrus.New(),
		circuitBreakers: make(map[string]*CircuitBreaker),
	}
}

// SelectService selects a service based on the load balancing strategy
func (lb *LoadBalancer) SelectService(category string, excludeServices []string) (*ServiceSelection, error) {
	// Get available services
	var services []*RegisteredService
	if category != "" {
		services = lb.registry.ListServicesByCategory(category)
	} else {
		services = lb.registry.ListServices()
	}

	if len(services) == 0 {
		return nil, fmt.Errorf("no services available")
	}

	// Filter out excluded services and unhealthy services
	availableServices := lb.filterAvailableServices(services, excludeServices)
	if len(availableServices) == 0 {
		return nil, fmt.Errorf("no available services after filtering")
	}

	// Apply load balancing strategy
	switch lb.strategy {
	case StrategyRoundRobin:
		return lb.selectRoundRobin(availableServices)
	case StrategyWeightedRandom:
		return lb.selectWeightedRandom(availableServices)
	case StrategyLeastLatency:
		return lb.selectLeastLatency(availableServices)
	case StrategyHealthyFirst:
		return lb.selectHealthyFirst(availableServices)
	case StrategyPriority:
		return lb.selectByPriority(availableServices)
	default:
		return lb.selectHealthyFirst(availableServices)
	}
}

// SelectMultipleServices selects multiple services for redundancy
func (lb *LoadBalancer) SelectMultipleServices(category string, count int, excludeServices []string) ([]*ServiceSelection, error) {
	if count <= 0 {
		return nil, fmt.Errorf("count must be positive")
	}

	var selections []*ServiceSelection
	excluded := make([]string, len(excludeServices))
	copy(excluded, excludeServices)

	for i := 0; i < count; i++ {
		selection, err := lb.SelectService(category, excluded)
		if err != nil {
			if len(selections) == 0 {
				return nil, err
			}
			break // Return what we have
		}

		selections = append(selections, selection)
		excluded = append(excluded, selection.Service.ID)
	}

	return selections, nil
}

// filterAvailableServices filters services based on health and circuit breaker state
func (lb *LoadBalancer) filterAvailableServices(services []*RegisteredService, excludeServices []string) []*RegisteredService {
	var available []*RegisteredService

	excludeMap := make(map[string]bool)
	for _, id := range excludeServices {
		excludeMap[id] = true
	}

	for _, service := range services {
		// Skip excluded services
		if excludeMap[service.ID] {
			continue
		}

		// Skip inactive services
		if service.Status != StatusActive {
			continue
		}

		// Check circuit breaker
		if lb.isCircuitOpen(service.ID) {
			continue
		}

		available = append(available, service)
	}

	return available
}

// selectRoundRobin implements round-robin selection
func (lb *LoadBalancer) selectRoundRobin(services []*RegisteredService) (*ServiceSelection, error) {
	if len(services) == 0 {
		return nil, fmt.Errorf("no services available")
	}

	// Simple round-robin based on current time
	index := int(time.Now().UnixNano()) % len(services)

	return &ServiceSelection{
		Service: services[index],
		Reason:  "round_robin",
	}, nil
}

// selectWeightedRandom implements weighted random selection
func (lb *LoadBalancer) selectWeightedRandom(services []*RegisteredService) (*ServiceSelection, error) {
	if len(services) == 0 {
		return nil, fmt.Errorf("no services available")
	}

	// Calculate total weight
	totalWeight := 0
	for _, service := range services {
		weight := service.Weight
		if weight <= 0 {
			weight = 1
		}
		totalWeight += weight
	}

	// Select random point
	randomPoint := rand.Intn(totalWeight)
	currentWeight := 0

	for _, service := range services {
		weight := service.Weight
		if weight <= 0 {
			weight = 1
		}
		currentWeight += weight

		if randomPoint < currentWeight {
			return &ServiceSelection{
				Service: service,
				Reason:  "weighted_random",
				Score:   float64(weight) / float64(totalWeight),
			}, nil
		}
	}

	// Fallback to first service
	return &ServiceSelection{
		Service: services[0],
		Reason:  "weighted_random_fallback",
	}, nil
}

// selectLeastLatency selects service with lowest average latency
func (lb *LoadBalancer) selectLeastLatency(services []*RegisteredService) (*ServiceSelection, error) {
	if len(services) == 0 {
		return nil, fmt.Errorf("no services available")
	}

	var bestService *RegisteredService
	var bestLatency time.Duration = time.Hour // Start with high value

	for _, service := range services {
		latency := service.AverageLatency
		if latency == 0 {
			latency = 100 * time.Millisecond // Default for new services
		}

		if latency < bestLatency {
			bestLatency = latency
			bestService = service
		}
	}

	if bestService == nil {
		bestService = services[0]
	}

	return &ServiceSelection{
		Service: bestService,
		Reason:  "least_latency",
		Score:   float64(bestLatency.Milliseconds()),
	}, nil
}

// selectHealthyFirst prioritizes healthy services
func (lb *LoadBalancer) selectHealthyFirst(services []*RegisteredService) (*ServiceSelection, error) {
	if len(services) == 0 {
		return nil, fmt.Errorf("no services available")
	}

	// Separate healthy and unhealthy services
	var healthyServices []*RegisteredService
	var unhealthyServices []*RegisteredService

	for _, service := range services {
		if service.HealthStatus == HealthHealthy {
			healthyServices = append(healthyServices, service)
		} else {
			unhealthyServices = append(unhealthyServices, service)
		}
	}

	// Prefer healthy services
	if len(healthyServices) > 0 {
		// Use weighted random among healthy services
		selection, err := lb.selectWeightedRandom(healthyServices)
		if err == nil {
			selection.Reason = "healthy_first"
			return selection, nil
		}
	}

	// Fallback to unhealthy services if no healthy ones
	if len(unhealthyServices) > 0 {
		selection, err := lb.selectWeightedRandom(unhealthyServices)
		if err == nil {
			selection.Reason = "healthy_first_fallback"
			return selection, nil
		}
	}

	return nil, fmt.Errorf("no suitable services found")
}

// selectByPriority selects service based on priority
func (lb *LoadBalancer) selectByPriority(services []*RegisteredService) (*ServiceSelection, error) {
	if len(services) == 0 {
		return nil, fmt.Errorf("no services available")
	}

	// Sort by priority (higher priority first)
	sort.Slice(services, func(i, j int) bool {
		return services[i].Priority > services[j].Priority
	})

	// Group services by priority
	highestPriority := services[0].Priority
	var highPriorityServices []*RegisteredService

	for _, service := range services {
		if service.Priority == highestPriority {
			highPriorityServices = append(highPriorityServices, service)
		} else {
			break
		}
	}

	// Use weighted random among highest priority services
	selection, err := lb.selectWeightedRandom(highPriorityServices)
	if err == nil {
		selection.Reason = "priority"
		return selection, nil
	}

	return nil, fmt.Errorf("no suitable services found")
}

// RecordSuccess records a successful request for circuit breaker
func (lb *LoadBalancer) RecordSuccess(serviceID string) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	if cb, exists := lb.circuitBreakers[serviceID]; exists {
		cb.mu.Lock()
		cb.FailureCount = 0
		if cb.State == CircuitHalfOpen {
			cb.State = CircuitClosed
		}
		cb.mu.Unlock()
	}
}

// RecordFailure records a failed request for circuit breaker
func (lb *LoadBalancer) RecordFailure(serviceID string) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	cb, exists := lb.circuitBreakers[serviceID]
	if !exists {
		cb = &CircuitBreaker{
			ServiceID:        serviceID,
			FailureThreshold: 5,
			RecoveryTimeout:  30 * time.Second,
			State:            CircuitClosed,
		}
		lb.circuitBreakers[serviceID] = cb
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.FailureCount++
	cb.LastFailureTime = time.Now()

	if cb.FailureCount >= cb.FailureThreshold && cb.State == CircuitClosed {
		cb.State = CircuitOpen
		lb.logger.Warnf("Circuit breaker opened for service %s", serviceID)
	}
}

// isCircuitOpen checks if circuit breaker is open for a service
func (lb *LoadBalancer) isCircuitOpen(serviceID string) bool {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	cb, exists := lb.circuitBreakers[serviceID]
	if !exists {
		return false
	}

	cb.mu.RLock()
	defer cb.mu.RUnlock()

	switch cb.State {
	case CircuitClosed:
		return false
	case CircuitOpen:
		// Check if recovery timeout has passed
		if time.Since(cb.LastFailureTime) > cb.RecoveryTimeout {
			cb.State = CircuitHalfOpen
			lb.logger.Infof("Circuit breaker half-opened for service %s", serviceID)
			return false
		}
		return true
	case CircuitHalfOpen:
		return false
	default:
		return false
	}
}

// GetCircuitBreakerStatus returns the status of all circuit breakers
func (lb *LoadBalancer) GetCircuitBreakerStatus() map[string]CircuitState {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	status := make(map[string]CircuitState)
	for serviceID, cb := range lb.circuitBreakers {
		cb.mu.RLock()
		status[serviceID] = cb.State
		cb.mu.RUnlock()
	}

	return status
}

// ResetCircuitBreaker resets the circuit breaker for a service
func (lb *LoadBalancer) ResetCircuitBreaker(serviceID string) {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	if cb, exists := lb.circuitBreakers[serviceID]; exists {
		cb.mu.Lock()
		cb.FailureCount = 0
		cb.State = CircuitClosed
		cb.mu.Unlock()
		lb.logger.Infof("Circuit breaker reset for service %s", serviceID)
	}
}

// GetLoadBalancingStats returns statistics about load balancing
func (lb *LoadBalancer) GetLoadBalancingStats() map[string]interface{} {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	stats := map[string]interface{}{
		"strategy":         lb.strategy,
		"circuit_breakers": len(lb.circuitBreakers),
	}

	// Circuit breaker stats
	circuitStats := make(map[string]interface{})
	for serviceID, cb := range lb.circuitBreakers {
		cb.mu.RLock()
		circuitStats[serviceID] = map[string]interface{}{
			"state":         cb.State,
			"failure_count": cb.FailureCount,
			"last_failure":  cb.LastFailureTime,
		}
		cb.mu.RUnlock()
	}
	stats["circuit_breaker_status"] = circuitStats

	return stats
}
