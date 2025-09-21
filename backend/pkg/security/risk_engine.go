package security

import (
	"context"
	"math"
	"net"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// RiskScore represents a calculated risk score
type RiskScore struct {
	Score        float64                `json:"score"` // 0.0 to 1.0
	Level        RiskLevel              `json:"level"`
	Factors      []*RiskFactor          `json:"factors"`
	Confidence   float64                `json:"confidence"` // 0.0 to 1.0
	Metadata     map[string]interface{} `json:"metadata"`
	CalculatedAt time.Time              `json:"calculated_at"`
}

// RiskLevel represents the risk level
type RiskLevel string

const (
	RiskLevelLow      RiskLevel = "low"
	RiskLevelMedium   RiskLevel = "medium"
	RiskLevelHigh     RiskLevel = "high"
	RiskLevelCritical RiskLevel = "critical"
)

// RiskFactor represents an individual risk factor
type RiskFactor struct {
	Type        RiskFactorType         `json:"type"`
	Name        string                 `json:"name"`
	Score       float64                `json:"score"`  // 0.0 to 1.0
	Weight      float64                `json:"weight"` // 0.0 to 1.0
	Description string                 `json:"description"`
	Evidence    map[string]interface{} `json:"evidence"`
}

// RiskFactorType represents the type of risk factor
type RiskFactorType string

const (
	RiskFactorBehavioral     RiskFactorType = "behavioral"
	RiskFactorGeographical   RiskFactorType = "geographical"
	RiskFactorTemporal       RiskFactorType = "temporal"
	RiskFactorDevice         RiskFactorType = "device"
	RiskFactorNetwork        RiskFactorType = "network"
	RiskFactorAuthentication RiskFactorType = "authentication"
	RiskFactorData           RiskFactorType = "data"
)

// UserBehaviorProfile represents a user's behavioral profile
type UserBehaviorProfile struct {
	UserID              string                 `json:"user_id"`
	TypicalLocations    []Location             `json:"typical_locations"`
	TypicalDevices      []string               `json:"typical_devices"`
	TypicalAccessTimes  []TimeWindow           `json:"typical_access_times"`
	TypicalResources    []string               `json:"typical_resources"`
	FailedAttempts      int                    `json:"failed_attempts"`
	LastSuccessfulLogin time.Time              `json:"last_successful_login"`
	AverageSessionTime  time.Duration          `json:"average_session_time"`
	AccessPatterns      map[string]interface{} `json:"access_patterns"`
	CreatedAt           time.Time              `json:"created_at"`
	UpdatedAt           time.Time              `json:"updated_at"`
}

// TimeWindow represents a time window for access patterns
type TimeWindow struct {
	StartHour int `json:"start_hour"`  // 0-23
	EndHour   int `json:"end_hour"`    // 0-23
	DayOfWeek int `json:"day_of_week"` // 0-6 (Sunday-Saturday)
}

// ThreatIntelligence represents threat intelligence data
type ThreatIntelligence struct {
	MaliciousIPs      map[string]ThreatInfo `json:"malicious_ips"`
	SuspiciousDevices map[string]ThreatInfo `json:"suspicious_devices"`
	KnownAttackers    map[string]ThreatInfo `json:"known_attackers"`
	UpdatedAt         time.Time             `json:"updated_at"`
}

// ThreatInfo represents information about a threat
type ThreatInfo struct {
	Severity    string                 `json:"severity"`
	Source      string                 `json:"source"`
	Description string                 `json:"description"`
	FirstSeen   time.Time              `json:"first_seen"`
	LastSeen    time.Time              `json:"last_seen"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// RiskEngine calculates risk scores for access requests
type RiskEngine struct {
	logger             *logrus.Logger
	behaviorProfiles   map[string]*UserBehaviorProfile
	threatIntelligence *ThreatIntelligence
	mu                 sync.RWMutex
	config             *RiskEngineConfig
}

// RiskEngineConfig contains configuration for the risk engine
type RiskEngineConfig struct {
	EnableBehavioralAnalysis    bool          `json:"enable_behavioral_analysis"`
	EnableThreatIntelligence    bool          `json:"enable_threat_intelligence"`
	EnableGeolocation           bool          `json:"enable_geolocation"`
	BehaviorLearningPeriod      time.Duration `json:"behavior_learning_period"`
	ThreatIntelUpdateInterval   time.Duration `json:"threat_intel_update_interval"`
	MaxFailedAttempts           int           `json:"max_failed_attempts"`
	SuspiciousActivityThreshold float64       `json:"suspicious_activity_threshold"`
}

// NewRiskEngine creates a new risk assessment engine
func NewRiskEngine() *RiskEngine {
	config := &RiskEngineConfig{
		EnableBehavioralAnalysis:    true,
		EnableThreatIntelligence:    true,
		EnableGeolocation:           true,
		BehaviorLearningPeriod:      30 * 24 * time.Hour, // 30 days
		ThreatIntelUpdateInterval:   time.Hour,
		MaxFailedAttempts:           5,
		SuspiciousActivityThreshold: 0.7,
	}

	engine := &RiskEngine{
		logger:           logrus.New(),
		behaviorProfiles: make(map[string]*UserBehaviorProfile),
		threatIntelligence: &ThreatIntelligence{
			MaliciousIPs:      make(map[string]ThreatInfo),
			SuspiciousDevices: make(map[string]ThreatInfo),
			KnownAttackers:    make(map[string]ThreatInfo),
			UpdatedAt:         time.Now(),
		},
		config: config,
	}

	// Start background tasks
	go engine.updateThreatIntelligence()
	go engine.cleanupOldProfiles()

	return engine
}

// CalculateRiskScore calculates the risk score for an access request
func (re *RiskEngine) CalculateRiskScore(ctx context.Context, request *AccessRequest) (float64, error) {
	startTime := time.Now()

	var factors []*RiskFactor

	// Behavioral analysis
	if re.config.EnableBehavioralAnalysis {
		behavioralFactors := re.analyzeBehavioralRisk(request)
		factors = append(factors, behavioralFactors...)
	}

	// Geographical analysis
	if re.config.EnableGeolocation && request.Context.Location != nil {
		geoFactors := re.analyzeGeographicalRisk(request)
		factors = append(factors, geoFactors...)
	}

	// Temporal analysis
	temporalFactors := re.analyzeTemporalRisk(request)
	factors = append(factors, temporalFactors...)

	// Device analysis
	deviceFactors := re.analyzeDeviceRisk(request)
	factors = append(factors, deviceFactors...)

	// Network analysis
	networkFactors := re.analyzeNetworkRisk(request)
	factors = append(factors, networkFactors...)

	// Authentication analysis
	authFactors := re.analyzeAuthenticationRisk(request)
	factors = append(factors, authFactors...)

	// Threat intelligence analysis
	if re.config.EnableThreatIntelligence {
		threatFactors := re.analyzeThreatIntelligence(request)
		factors = append(factors, threatFactors...)
	}

	// Calculate weighted risk score
	totalScore := 0.0
	totalWeight := 0.0
	confidence := 1.0

	for _, factor := range factors {
		totalScore += factor.Score * factor.Weight
		totalWeight += factor.Weight
		confidence = math.Min(confidence, 0.9) // Reduce confidence slightly for each factor
	}

	finalScore := 0.0
	if totalWeight > 0 {
		finalScore = totalScore / totalWeight
	}

	// Determine risk level
	level := re.determineRiskLevel(finalScore)

	_ = &RiskScore{
		Score:      finalScore,
		Level:      level,
		Factors:    factors,
		Confidence: confidence,
		Metadata: map[string]interface{}{
			"calculation_time": time.Since(startTime).Milliseconds(),
			"factors_count":    len(factors),
		},
		CalculatedAt: time.Now(),
	}

	re.logger.Debugf("Calculated risk score %.3f (%s) for user %s in %v",
		finalScore, level, request.UserID, time.Since(startTime))

	return finalScore, nil
}

// analyzeBehavioralRisk analyzes behavioral risk factors
func (re *RiskEngine) analyzeBehavioralRisk(request *AccessRequest) []*RiskFactor {
	var factors []*RiskFactor

	re.mu.RLock()
	profile, exists := re.behaviorProfiles[request.UserID]
	re.mu.RUnlock()

	if !exists {
		// New user - medium risk
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorBehavioral,
			Name:        "new_user",
			Score:       0.5,
			Weight:      0.3,
			Description: "New user with no behavioral history",
			Evidence: map[string]interface{}{
				"user_id": request.UserID,
			},
		})
		return factors
	}

	// Check location deviation
	if request.Context.Location != nil {
		locationRisk := re.calculateLocationDeviation(profile, request.Context.Location)
		if locationRisk > 0.1 {
			factors = append(factors, &RiskFactor{
				Type:        RiskFactorBehavioral,
				Name:        "location_deviation",
				Score:       locationRisk,
				Weight:      0.4,
				Description: "Access from unusual location",
				Evidence: map[string]interface{}{
					"current_location":  request.Context.Location,
					"typical_locations": profile.TypicalLocations,
				},
			})
		}
	}

	// Check device deviation
	if request.Context.DeviceID != "" {
		deviceRisk := re.calculateDeviceDeviation(profile, request.Context.DeviceID)
		if deviceRisk > 0.1 {
			factors = append(factors, &RiskFactor{
				Type:        RiskFactorBehavioral,
				Name:        "device_deviation",
				Score:       deviceRisk,
				Weight:      0.3,
				Description: "Access from unusual device",
				Evidence: map[string]interface{}{
					"current_device":  request.Context.DeviceID,
					"typical_devices": profile.TypicalDevices,
				},
			})
		}
	}

	// Check time deviation
	timeRisk := re.calculateTimeDeviation(profile, request.Timestamp)
	if timeRisk > 0.1 {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorBehavioral,
			Name:        "time_deviation",
			Score:       timeRisk,
			Weight:      0.2,
			Description: "Access at unusual time",
			Evidence: map[string]interface{}{
				"current_time":  request.Timestamp,
				"typical_times": profile.TypicalAccessTimes,
			},
		})
	}

	// Check resource deviation
	resourceRisk := re.calculateResourceDeviation(profile, request.Resource)
	if resourceRisk > 0.1 {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorBehavioral,
			Name:        "resource_deviation",
			Score:       resourceRisk,
			Weight:      0.2,
			Description: "Access to unusual resource",
			Evidence: map[string]interface{}{
				"current_resource":  request.Resource,
				"typical_resources": profile.TypicalResources,
			},
		})
	}

	// Check failed attempts
	if profile.FailedAttempts > re.config.MaxFailedAttempts {
		failureRisk := math.Min(1.0, float64(profile.FailedAttempts)/float64(re.config.MaxFailedAttempts*2))
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorBehavioral,
			Name:        "excessive_failures",
			Score:       failureRisk,
			Weight:      0.5,
			Description: "Excessive failed login attempts",
			Evidence: map[string]interface{}{
				"failed_attempts": profile.FailedAttempts,
				"max_allowed":     re.config.MaxFailedAttempts,
			},
		})
	}

	return factors
}

// analyzeGeographicalRisk analyzes geographical risk factors
func (re *RiskEngine) analyzeGeographicalRisk(request *AccessRequest) []*RiskFactor {
	var factors []*RiskFactor

	location := request.Context.Location
	if location == nil {
		return factors
	}

	// Check high-risk countries
	highRiskCountries := []string{"CN", "RU", "KP", "IR"} // Example list
	for _, country := range highRiskCountries {
		if location.Country == country {
			factors = append(factors, &RiskFactor{
				Type:        RiskFactorGeographical,
				Name:        "high_risk_country",
				Score:       0.8,
				Weight:      0.4,
				Description: "Access from high-risk country",
				Evidence: map[string]interface{}{
					"country": location.Country,
				},
			})
			break
		}
	}

	// Check for VPN/Proxy indicators
	if re.isVPNOrProxy(request.Context.IPAddress) {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorGeographical,
			Name:        "vpn_proxy_detected",
			Score:       0.6,
			Weight:      0.3,
			Description: "Access through VPN or proxy",
			Evidence: map[string]interface{}{
				"ip_address": request.Context.IPAddress,
			},
		})
	}

	return factors
}

// analyzeTemporalRisk analyzes temporal risk factors
func (re *RiskEngine) analyzeTemporalRisk(request *AccessRequest) []*RiskFactor {
	var factors []*RiskFactor

	now := request.Timestamp
	hour := now.Hour()
	weekday := int(now.Weekday())

	// Check for off-hours access
	if hour < 6 || hour > 22 { // Outside 6 AM - 10 PM
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorTemporal,
			Name:        "off_hours_access",
			Score:       0.4,
			Weight:      0.2,
			Description: "Access during off-hours",
			Evidence: map[string]interface{}{
				"hour":    hour,
				"weekday": weekday,
			},
		})
	}

	// Check for weekend access
	if weekday == 0 || weekday == 6 { // Sunday or Saturday
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorTemporal,
			Name:        "weekend_access",
			Score:       0.3,
			Weight:      0.1,
			Description: "Access during weekend",
			Evidence: map[string]interface{}{
				"weekday": weekday,
			},
		})
	}

	return factors
}

// analyzeDeviceRisk analyzes device-related risk factors
func (re *RiskEngine) analyzeDeviceRisk(request *AccessRequest) []*RiskFactor {
	var factors []*RiskFactor

	// Check for missing device ID
	if request.Context.DeviceID == "" {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorDevice,
			Name:        "missing_device_id",
			Score:       0.5,
			Weight:      0.3,
			Description: "No device identifier provided",
			Evidence: map[string]interface{}{
				"user_agent": request.Context.UserAgent,
			},
		})
	}

	// Analyze user agent for suspicious patterns
	userAgent := request.Context.UserAgent
	if re.isSuspiciousUserAgent(userAgent) {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorDevice,
			Name:        "suspicious_user_agent",
			Score:       0.7,
			Weight:      0.4,
			Description: "Suspicious user agent detected",
			Evidence: map[string]interface{}{
				"user_agent": userAgent,
			},
		})
	}

	return factors
}

// analyzeNetworkRisk analyzes network-related risk factors
func (re *RiskEngine) analyzeNetworkRisk(request *AccessRequest) []*RiskFactor {
	var factors []*RiskFactor

	ipAddress := request.Context.IPAddress

	// Check for private IP addresses
	if re.isPrivateIP(ipAddress) {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorNetwork,
			Name:        "private_ip_access",
			Score:       0.2,
			Weight:      0.1,
			Description: "Access from private IP address",
			Evidence: map[string]interface{}{
				"ip_address": ipAddress,
			},
		})
	}

	// Check for Tor exit nodes (simplified check)
	if re.isTorExitNode(ipAddress) {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorNetwork,
			Name:        "tor_exit_node",
			Score:       0.9,
			Weight:      0.6,
			Description: "Access from Tor exit node",
			Evidence: map[string]interface{}{
				"ip_address": ipAddress,
			},
		})
	}

	return factors
}

// analyzeAuthenticationRisk analyzes authentication-related risk factors
func (re *RiskEngine) analyzeAuthenticationRisk(request *AccessRequest) []*RiskFactor {
	var factors []*RiskFactor

	// Check for missing session
	if request.Context.SessionID == "" {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorAuthentication,
			Name:        "missing_session",
			Score:       0.6,
			Weight:      0.4,
			Description: "No session identifier provided",
			Evidence: map[string]interface{}{
				"user_id": request.UserID,
			},
		})
	}

	// Check for weak authentication claims
	if len(request.Claims) == 0 {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorAuthentication,
			Name:        "no_auth_claims",
			Score:       0.8,
			Weight:      0.5,
			Description: "No authentication claims provided",
			Evidence: map[string]interface{}{
				"user_id": request.UserID,
			},
		})
	}

	return factors
}

// analyzeThreatIntelligence analyzes threat intelligence data
func (re *RiskEngine) analyzeThreatIntelligence(request *AccessRequest) []*RiskFactor {
	var factors []*RiskFactor

	re.mu.RLock()
	defer re.mu.RUnlock()

	// Check malicious IPs
	if threat, exists := re.threatIntelligence.MaliciousIPs[request.Context.IPAddress]; exists {
		severity := 0.5
		if threat.Severity == "high" {
			severity = 0.8
		} else if threat.Severity == "critical" {
			severity = 1.0
		}

		factors = append(factors, &RiskFactor{
			Type:        RiskFactorNetwork,
			Name:        "malicious_ip",
			Score:       severity,
			Weight:      0.8,
			Description: "IP address found in threat intelligence",
			Evidence: map[string]interface{}{
				"ip_address":  request.Context.IPAddress,
				"threat_info": threat,
			},
		})
	}

	// Check suspicious devices
	if threat, exists := re.threatIntelligence.SuspiciousDevices[request.Context.DeviceID]; exists {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorDevice,
			Name:        "suspicious_device",
			Score:       0.7,
			Weight:      0.6,
			Description: "Device found in threat intelligence",
			Evidence: map[string]interface{}{
				"device_id":   request.Context.DeviceID,
				"threat_info": threat,
			},
		})
	}

	// Check known attackers
	if threat, exists := re.threatIntelligence.KnownAttackers[request.UserID]; exists {
		factors = append(factors, &RiskFactor{
			Type:        RiskFactorBehavioral,
			Name:        "known_attacker",
			Score:       0.95,
			Weight:      0.9,
			Description: "User identified as known attacker",
			Evidence: map[string]interface{}{
				"user_id":     request.UserID,
				"threat_info": threat,
			},
		})
	}

	return factors
}

// Helper methods

func (re *RiskEngine) determineRiskLevel(score float64) RiskLevel {
	if score >= 0.8 {
		return RiskLevelCritical
	} else if score >= 0.6 {
		return RiskLevelHigh
	} else if score >= 0.3 {
		return RiskLevelMedium
	}
	return RiskLevelLow
}

func (re *RiskEngine) calculateLocationDeviation(profile *UserBehaviorProfile, location *Location) float64 {
	if len(profile.TypicalLocations) == 0 {
		return 0.5 // Medium risk for no history
	}

	minDistance := math.Inf(1)
	for _, typical := range profile.TypicalLocations {
		distance := re.calculateDistance(location, &typical)
		if distance < minDistance {
			minDistance = distance
		}
	}

	// Convert distance to risk score (0-1)
	// Distances over 1000km are considered high risk
	if minDistance > 1000 {
		return 0.8
	} else if minDistance > 500 {
		return 0.6
	} else if minDistance > 100 {
		return 0.3
	}
	return 0.1
}

func (re *RiskEngine) calculateDistance(loc1, loc2 *Location) float64 {
	// Simplified distance calculation using Haversine formula
	const earthRadius = 6371 // km

	lat1Rad := loc1.Latitude * math.Pi / 180
	lat2Rad := loc2.Latitude * math.Pi / 180
	deltaLat := (loc2.Latitude - loc1.Latitude) * math.Pi / 180
	deltaLon := (loc2.Longitude - loc1.Longitude) * math.Pi / 180

	a := math.Sin(deltaLat/2)*math.Sin(deltaLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*
			math.Sin(deltaLon/2)*math.Sin(deltaLon/2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return earthRadius * c
}

func (re *RiskEngine) calculateDeviceDeviation(profile *UserBehaviorProfile, deviceID string) float64 {
	if len(profile.TypicalDevices) == 0 {
		return 0.5 // Medium risk for no history
	}

	for _, typical := range profile.TypicalDevices {
		if typical == deviceID {
			return 0.1 // Low risk for known device
		}
	}

	return 0.7 // High risk for unknown device
}

func (re *RiskEngine) calculateTimeDeviation(profile *UserBehaviorProfile, timestamp time.Time) float64 {
	if len(profile.TypicalAccessTimes) == 0 {
		return 0.3 // Medium-low risk for no history
	}

	hour := timestamp.Hour()
	weekday := int(timestamp.Weekday())

	for _, window := range profile.TypicalAccessTimes {
		if window.DayOfWeek == weekday &&
			hour >= window.StartHour && hour <= window.EndHour {
			return 0.1 // Low risk for typical time
		}
	}

	return 0.5 // Medium risk for atypical time
}

func (re *RiskEngine) calculateResourceDeviation(profile *UserBehaviorProfile, resource string) float64 {
	if len(profile.TypicalResources) == 0 {
		return 0.3 // Medium-low risk for no history
	}

	for _, typical := range profile.TypicalResources {
		if typical == resource {
			return 0.1 // Low risk for typical resource
		}
	}

	return 0.4 // Medium risk for atypical resource
}

func (re *RiskEngine) isVPNOrProxy(ipAddress string) bool {
	// This would integrate with VPN/proxy detection services
	// For now, simple heuristic based on common VPN ranges
	vpnRanges := []string{
		"10.0.0.0/8",
		"172.16.0.0/12",
		"192.168.0.0/16",
	}

	ip := net.ParseIP(ipAddress)
	if ip == nil {
		return false
	}

	for _, cidr := range vpnRanges {
		_, network, err := net.ParseCIDR(cidr)
		if err != nil {
			continue
		}
		if network.Contains(ip) {
			return true
		}
	}

	return false
}

func (re *RiskEngine) isPrivateIP(ipAddress string) bool {
	ip := net.ParseIP(ipAddress)
	if ip == nil {
		return false
	}

	return ip.IsPrivate()
}

func (re *RiskEngine) isTorExitNode(ipAddress string) bool {
	// This would check against a list of known Tor exit nodes
	// For now, return false
	return false
}

func (re *RiskEngine) isSuspiciousUserAgent(userAgent string) bool {
	suspiciousPatterns := []string{
		"bot",
		"crawler",
		"spider",
		"scraper",
		"curl",
		"wget",
		"python",
		"java",
	}

	userAgentLower := strings.ToLower(userAgent)
	for _, pattern := range suspiciousPatterns {
		if strings.Contains(userAgentLower, pattern) {
			return true
		}
	}

	return false
}

// Background tasks

func (re *RiskEngine) updateThreatIntelligence() {
	ticker := time.NewTicker(re.config.ThreatIntelUpdateInterval)
	defer ticker.Stop()

	for range ticker.C {
		// This would fetch threat intelligence from external sources
		// For now, just update the timestamp
		re.mu.Lock()
		re.threatIntelligence.UpdatedAt = time.Now()
		re.mu.Unlock()

		re.logger.Debug("Updated threat intelligence data")
	}
}

func (re *RiskEngine) cleanupOldProfiles() {
	ticker := time.NewTicker(24 * time.Hour) // Daily cleanup
	defer ticker.Stop()

	for range ticker.C {
		cutoff := time.Now().Add(-re.config.BehaviorLearningPeriod * 2)

		re.mu.Lock()
		for userID, profile := range re.behaviorProfiles {
			if profile.UpdatedAt.Before(cutoff) {
				delete(re.behaviorProfiles, userID)
				re.logger.Debugf("Cleaned up old behavior profile for user %s", userID)
			}
		}
		re.mu.Unlock()
	}
}

// UpdateUserBehavior updates a user's behavioral profile
func (re *RiskEngine) UpdateUserBehavior(userID string, request *AccessRequest, success bool) {
	re.mu.Lock()
	defer re.mu.Unlock()

	profile, exists := re.behaviorProfiles[userID]
	if !exists {
		profile = &UserBehaviorProfile{
			UserID:             userID,
			TypicalLocations:   make([]Location, 0),
			TypicalDevices:     make([]string, 0),
			TypicalAccessTimes: make([]TimeWindow, 0),
			TypicalResources:   make([]string, 0),
			AccessPatterns:     make(map[string]interface{}),
			CreatedAt:          time.Now(),
		}
		re.behaviorProfiles[userID] = profile
	}

	if success {
		profile.LastSuccessfulLogin = request.Timestamp

		// Update typical locations
		if request.Context.Location != nil {
			re.updateTypicalLocations(profile, request.Context.Location)
		}

		// Update typical devices
		if request.Context.DeviceID != "" {
			re.updateTypicalDevices(profile, request.Context.DeviceID)
		}

		// Update typical access times
		re.updateTypicalAccessTimes(profile, request.Timestamp)

		// Update typical resources
		re.updateTypicalResources(profile, request.Resource)

		// Reset failed attempts on successful login
		profile.FailedAttempts = 0
	} else {
		profile.FailedAttempts++
	}

	profile.UpdatedAt = time.Now()
}

func (re *RiskEngine) updateTypicalLocations(profile *UserBehaviorProfile, location *Location) {
	// Add location if not already present
	for _, typical := range profile.TypicalLocations {
		if re.calculateDistance(location, &typical) < 50 { // Within 50km
			return // Already have similar location
		}
	}

	profile.TypicalLocations = append(profile.TypicalLocations, *location)

	// Keep only the most recent 10 locations
	if len(profile.TypicalLocations) > 10 {
		profile.TypicalLocations = profile.TypicalLocations[1:]
	}
}

func (re *RiskEngine) updateTypicalDevices(profile *UserBehaviorProfile, deviceID string) {
	// Add device if not already present
	for _, typical := range profile.TypicalDevices {
		if typical == deviceID {
			return // Already have this device
		}
	}

	profile.TypicalDevices = append(profile.TypicalDevices, deviceID)

	// Keep only the most recent 5 devices
	if len(profile.TypicalDevices) > 5 {
		profile.TypicalDevices = profile.TypicalDevices[1:]
	}
}

func (re *RiskEngine) updateTypicalAccessTimes(profile *UserBehaviorProfile, timestamp time.Time) {
	hour := timestamp.Hour()
	weekday := int(timestamp.Weekday())

	// Create time window (±1 hour)
	startHour := hour - 1
	endHour := hour + 1
	if startHour < 0 {
		startHour = 0
	}
	if endHour > 23 {
		endHour = 23
	}

	window := TimeWindow{
		StartHour: startHour,
		EndHour:   endHour,
		DayOfWeek: weekday,
	}

	// Check if similar window already exists
	for _, typical := range profile.TypicalAccessTimes {
		if typical.DayOfWeek == weekday &&
			abs(typical.StartHour-startHour) <= 2 &&
			abs(typical.EndHour-endHour) <= 2 {
			return // Already have similar time window
		}
	}

	profile.TypicalAccessTimes = append(profile.TypicalAccessTimes, window)

	// Keep only the most recent 20 time windows
	if len(profile.TypicalAccessTimes) > 20 {
		profile.TypicalAccessTimes = profile.TypicalAccessTimes[1:]
	}
}

func (re *RiskEngine) updateTypicalResources(profile *UserBehaviorProfile, resource string) {
	// Add resource if not already present
	for _, typical := range profile.TypicalResources {
		if typical == resource {
			return // Already have this resource
		}
	}

	profile.TypicalResources = append(profile.TypicalResources, resource)

	// Keep only the most recent 50 resources
	if len(profile.TypicalResources) > 50 {
		profile.TypicalResources = profile.TypicalResources[1:]
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
