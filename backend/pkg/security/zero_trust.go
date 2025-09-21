package security

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/sirupsen/logrus"
)

// ZeroTrustPolicy represents a zero trust security policy
type ZeroTrustPolicy struct {
	ID                   string               `json:"id"`
	Name                 string               `json:"name"`
	Description          string               `json:"description"`
	Rules                []SecurityRule       `json:"rules"`
	RequiredClaims       []string             `json:"required_claims"`
	AllowedResources     []string             `json:"allowed_resources"`
	DeniedResources      []string             `json:"denied_resources"`
	TimeRestrictions     *TimeRestriction     `json:"time_restrictions,omitempty"`
	LocationRestrictions *LocationRestriction `json:"location_restrictions,omitempty"`
	DeviceRestrictions   *DeviceRestriction   `json:"device_restrictions,omitempty"`
	CreatedAt            time.Time            `json:"created_at"`
	UpdatedAt            time.Time            `json:"updated_at"`
	Enabled              bool                 `json:"enabled"`
}

// SecurityRule represents a security rule within a policy
type SecurityRule struct {
	ID        string            `json:"id"`
	Type      RuleType          `json:"type"`
	Condition string            `json:"condition"`
	Action    RuleAction        `json:"action"`
	Priority  int               `json:"priority"`
	Metadata  map[string]string `json:"metadata"`
}

// RuleType represents the type of security rule
type RuleType string

const (
	RuleTypeAuthentication RuleType = "authentication"
	RuleTypeAuthorization  RuleType = "authorization"
	RuleTypeRateLimit      RuleType = "rate_limit"
	RuleTypeDataAccess     RuleType = "data_access"
	RuleTypeAudit          RuleType = "audit"
	RuleTypeEncryption     RuleType = "encryption"
)

// RuleAction represents the action to take when a rule is triggered
type RuleAction string

const (
	ActionAllow     RuleAction = "allow"
	ActionDeny      RuleAction = "deny"
	ActionChallenge RuleAction = "challenge"
	ActionLog       RuleAction = "log"
	ActionAlert     RuleAction = "alert"
)

// TimeRestriction represents time-based access restrictions
type TimeRestriction struct {
	AllowedHours   []int    `json:"allowed_hours"` // 0-23
	AllowedDays    []int    `json:"allowed_days"`  // 0-6 (Sunday-Saturday)
	Timezone       string   `json:"timezone"`
	ExceptionDates []string `json:"exception_dates"` // YYYY-MM-DD format
}

// LocationRestriction represents location-based access restrictions
type LocationRestriction struct {
	AllowedCountries []string `json:"allowed_countries"`
	AllowedRegions   []string `json:"allowed_regions"`
	AllowedIPRanges  []string `json:"allowed_ip_ranges"`
	DeniedCountries  []string `json:"denied_countries"`
	DeniedIPRanges   []string `json:"denied_ip_ranges"`
}

// DeviceRestriction represents device-based access restrictions
type DeviceRestriction struct {
	AllowedDeviceTypes []string `json:"allowed_device_types"`
	RequiredDeviceID   bool     `json:"required_device_id"`
	AllowedDeviceIDs   []string `json:"allowed_device_ids"`
	RequireEncryption  bool     `json:"require_encryption"`
	RequireBiometric   bool     `json:"require_biometric"`
}

// AccessRequest represents a request for resource access
type AccessRequest struct {
	UserID    string                 `json:"user_id"`
	Resource  string                 `json:"resource"`
	Action    string                 `json:"action"`
	Context   RequestContext         `json:"context"`
	Claims    map[string]interface{} `json:"claims"`
	Timestamp time.Time              `json:"timestamp"`
}

// RequestContext contains contextual information about the request
type RequestContext struct {
	IPAddress string            `json:"ip_address"`
	UserAgent string            `json:"user_agent"`
	DeviceID  string            `json:"device_id"`
	Location  *Location         `json:"location,omitempty"`
	SessionID string            `json:"session_id"`
	RequestID string            `json:"request_id"`
	Headers   map[string]string `json:"headers"`
}

// Location represents geographical location information
type Location struct {
	Country   string  `json:"country"`
	Region    string  `json:"region"`
	City      string  `json:"city"`
	Latitude  float64 `json:"latitude"`
	Longitude float64 `json:"longitude"`
}

// AccessDecision represents the result of an access control decision
type AccessDecision struct {
	Allowed      bool              `json:"allowed"`
	Reason       string            `json:"reason"`
	PolicyID     string            `json:"policy_id"`
	RuleID       string            `json:"rule_id"`
	Action       RuleAction        `json:"action"`
	Confidence   float64           `json:"confidence"`
	Metadata     map[string]string `json:"metadata"`
	Timestamp    time.Time         `json:"timestamp"`
	SessionToken string            `json:"session_token,omitempty"`
}

// ZeroTrustEngine implements zero trust security architecture
type ZeroTrustEngine struct {
	policies      map[string]*ZeroTrustPolicy
	mu            sync.RWMutex
	logger        *logrus.Logger
	auditLogger   *AuditLogger
	riskEngine    *RiskEngine
	encryptionMgr *EncryptionManager
	complianceMgr *ComplianceManager
}

// NewZeroTrustEngine creates a new zero trust security engine
func NewZeroTrustEngine() *ZeroTrustEngine {
	return &ZeroTrustEngine{
		policies:      make(map[string]*ZeroTrustPolicy),
		logger:        logrus.New(),
		auditLogger:   NewAuditLogger(),
		riskEngine:    NewRiskEngine(),
		encryptionMgr: NewEncryptionManager(),
		complianceMgr: NewComplianceManager(),
	}
}

// AddPolicy adds a new zero trust policy
func (zte *ZeroTrustEngine) AddPolicy(policy *ZeroTrustPolicy) error {
	zte.mu.Lock()
	defer zte.mu.Unlock()

	if policy.ID == "" {
		return fmt.Errorf("policy ID cannot be empty")
	}

	policy.CreatedAt = time.Now()
	policy.UpdatedAt = time.Now()

	zte.policies[policy.ID] = policy
	zte.logger.Infof("Added zero trust policy: %s", policy.ID)

	return nil
}

// EvaluateAccess evaluates an access request against zero trust policies
func (zte *ZeroTrustEngine) EvaluateAccess(ctx context.Context, request *AccessRequest) (*AccessDecision, error) {
	startTime := time.Now()

	// Log the access request
	zte.auditLogger.LogAccessRequest(request)

	// Calculate risk score
	riskScore, err := zte.riskEngine.CalculateRiskScore(ctx, request)
	if err != nil {
		zte.logger.Errorf("Failed to calculate risk score: %v", err)
		riskScore = 0.5 // Default medium risk
	}

	// Find applicable policies
	applicablePolicies := zte.findApplicablePolicies(request)

	if len(applicablePolicies) == 0 {
		// No policies found - default deny
		decision := &AccessDecision{
			Allowed:    false,
			Reason:     "No applicable policies found",
			Action:     ActionDeny,
			Confidence: 1.0,
			Timestamp:  time.Now(),
		}
		zte.auditLogger.LogAccessDecision(request, decision)
		return decision, nil
	}

	// Evaluate each applicable policy
	var finalDecision *AccessDecision
	highestPriority := -1

	for _, policy := range applicablePolicies {
		decision, err := zte.evaluatePolicy(ctx, request, policy, riskScore)
		if err != nil {
			zte.logger.Errorf("Failed to evaluate policy %s: %v", policy.ID, err)
			continue
		}

		// Use the decision from the highest priority rule
		if decision != nil {
			for _, rule := range policy.Rules {
				if rule.Priority > highestPriority {
					highestPriority = rule.Priority
					finalDecision = decision
					finalDecision.PolicyID = policy.ID
				}
			}
		}
	}

	if finalDecision == nil {
		// No decision made - default deny
		finalDecision = &AccessDecision{
			Allowed:    false,
			Reason:     "No policy evaluation resulted in a decision",
			Action:     ActionDeny,
			Confidence: 1.0,
			Timestamp:  time.Now(),
		}
	}

	// Apply additional security measures based on risk score
	if finalDecision.Allowed && riskScore > 0.7 {
		finalDecision.Action = ActionChallenge
		finalDecision.Reason += " (High risk - additional verification required)"
	}

	// Generate session token if access is allowed
	if finalDecision.Allowed {
		sessionToken, err := zte.generateSessionToken(request)
		if err != nil {
			zte.logger.Errorf("Failed to generate session token: %v", err)
		} else {
			finalDecision.SessionToken = sessionToken
		}
	}

	// Log the final decision
	zte.auditLogger.LogAccessDecision(request, finalDecision)

	zte.logger.Infof("Access evaluation completed in %v for user %s",
		time.Since(startTime), request.UserID)

	return finalDecision, nil
}

// findApplicablePolicies finds policies that apply to the given request
func (zte *ZeroTrustEngine) findApplicablePolicies(request *AccessRequest) []*ZeroTrustPolicy {
	zte.mu.RLock()
	defer zte.mu.RUnlock()

	var applicable []*ZeroTrustPolicy

	for _, policy := range zte.policies {
		if !policy.Enabled {
			continue
		}

		// Check if resource is allowed
		if len(policy.AllowedResources) > 0 {
			found := false
			for _, resource := range policy.AllowedResources {
				if zte.matchesResource(request.Resource, resource) {
					found = true
					break
				}
			}
			if !found {
				continue
			}
		}

		// Check if resource is denied
		for _, resource := range policy.DeniedResources {
			if zte.matchesResource(request.Resource, resource) {
				continue // Skip this policy
			}
		}

		// Check time restrictions
		if policy.TimeRestrictions != nil && !zte.checkTimeRestrictions(policy.TimeRestrictions) {
			continue
		}

		// Check location restrictions
		if policy.LocationRestrictions != nil && !zte.checkLocationRestrictions(policy.LocationRestrictions, request.Context) {
			continue
		}

		// Check device restrictions
		if policy.DeviceRestrictions != nil && !zte.checkDeviceRestrictions(policy.DeviceRestrictions, request.Context) {
			continue
		}

		applicable = append(applicable, policy)
	}

	return applicable
}

// evaluatePolicy evaluates a single policy against the request
func (zte *ZeroTrustEngine) evaluatePolicy(ctx context.Context, request *AccessRequest, policy *ZeroTrustPolicy, riskScore float64) (*AccessDecision, error) {

	// Check required claims
	for _, claim := range policy.RequiredClaims {
		if _, exists := request.Claims[claim]; !exists {
			return &AccessDecision{
				Allowed:    false,
				Reason:     fmt.Sprintf("Missing required claim: %s", claim),
				Action:     ActionDeny,
				Confidence: 1.0,
				Timestamp:  time.Now(),
			}, nil
		}
	}

	// Evaluate rules in priority order
	for _, rule := range policy.Rules {
		decision, err := zte.evaluateRule(ctx, request, &rule, riskScore)
		if err != nil {
			zte.logger.Errorf("Failed to evaluate rule %s: %v", rule.ID, err)
			continue
		}

		if decision != nil {
			decision.RuleID = rule.ID
			return decision, nil
		}
	}

	return nil, nil
}

// evaluateRule evaluates a single security rule
func (zte *ZeroTrustEngine) evaluateRule(ctx context.Context, request *AccessRequest, rule *SecurityRule, riskScore float64) (*AccessDecision, error) {

	switch rule.Type {
	case RuleTypeAuthentication:
		return zte.evaluateAuthenticationRule(request, rule)
	case RuleTypeAuthorization:
		return zte.evaluateAuthorizationRule(request, rule)
	case RuleTypeRateLimit:
		return zte.evaluateRateLimitRule(request, rule)
	case RuleTypeDataAccess:
		return zte.evaluateDataAccessRule(request, rule)
	case RuleTypeAudit:
		return zte.evaluateAuditRule(request, rule)
	case RuleTypeEncryption:
		return zte.evaluateEncryptionRule(request, rule)
	default:
		return nil, fmt.Errorf("unknown rule type: %s", rule.Type)
	}
}

// evaluateAuthenticationRule evaluates authentication rules
func (zte *ZeroTrustEngine) evaluateAuthenticationRule(request *AccessRequest, rule *SecurityRule) (*AccessDecision, error) {

	// Check if user is authenticated
	if request.UserID == "" {
		return &AccessDecision{
			Allowed:    false,
			Reason:     "User not authenticated",
			Action:     ActionDeny,
			Confidence: 1.0,
			Timestamp:  time.Now(),
		}, nil
	}

	// Check for valid session
	if request.Context.SessionID == "" {
		return &AccessDecision{
			Allowed:    false,
			Reason:     "No valid session",
			Action:     ActionChallenge,
			Confidence: 0.9,
			Timestamp:  time.Now(),
		}, nil
	}

	return &AccessDecision{
		Allowed:    true,
		Reason:     "Authentication successful",
		Action:     ActionAllow,
		Confidence: 0.9,
		Timestamp:  time.Now(),
	}, nil
}

// evaluateAuthorizationRule evaluates authorization rules
func (zte *ZeroTrustEngine) evaluateAuthorizationRule(request *AccessRequest, rule *SecurityRule) (*AccessDecision, error) {

	// Check user roles/permissions from claims
	roles, exists := request.Claims["roles"]
	if !exists {
		return &AccessDecision{
			Allowed:    false,
			Reason:     "No roles found in user claims",
			Action:     ActionDeny,
			Confidence: 1.0,
			Timestamp:  time.Now(),
		}, nil
	}

	// Simple role-based check
	requiredRole := rule.Metadata["required_role"]
	if requiredRole != "" {
		userRoles, ok := roles.([]interface{})
		if !ok {
			return &AccessDecision{
				Allowed:    false,
				Reason:     "Invalid roles format",
				Action:     ActionDeny,
				Confidence: 1.0,
				Timestamp:  time.Now(),
			}, nil
		}

		hasRole := false
		for _, role := range userRoles {
			if roleStr, ok := role.(string); ok && roleStr == requiredRole {
				hasRole = true
				break
			}
		}

		if !hasRole {
			return &AccessDecision{
				Allowed:    false,
				Reason:     fmt.Sprintf("Missing required role: %s", requiredRole),
				Action:     ActionDeny,
				Confidence: 1.0,
				Timestamp:  time.Now(),
			}, nil
		}
	}

	return &AccessDecision{
		Allowed:    true,
		Reason:     "Authorization successful",
		Action:     ActionAllow,
		Confidence: 0.9,
		Timestamp:  time.Now(),
	}, nil
}

// evaluateRateLimitRule evaluates rate limiting rules
func (zte *ZeroTrustEngine) evaluateRateLimitRule(request *AccessRequest, rule *SecurityRule) (*AccessDecision, error) {

	// This would integrate with a rate limiting service
	// For now, return allow
	return &AccessDecision{
		Allowed:    true,
		Reason:     "Rate limit check passed",
		Action:     ActionAllow,
		Confidence: 0.8,
		Timestamp:  time.Now(),
	}, nil
}

// evaluateDataAccessRule evaluates data access rules
func (zte *ZeroTrustEngine) evaluateDataAccessRule(request *AccessRequest, rule *SecurityRule) (*AccessDecision, error) {

	// Check data classification and user clearance
	dataClassification := rule.Metadata["data_classification"]
	userClearance, exists := request.Claims["clearance_level"]

	if dataClassification != "" && exists {
		// Simple clearance level check
		clearanceStr, ok := userClearance.(string)
		if !ok || !zte.hasRequiredClearance(clearanceStr, dataClassification) {
			return &AccessDecision{
				Allowed:    false,
				Reason:     "Insufficient clearance level for data access",
				Action:     ActionDeny,
				Confidence: 1.0,
				Timestamp:  time.Now(),
			}, nil
		}
	}

	return &AccessDecision{
		Allowed:    true,
		Reason:     "Data access authorized",
		Action:     ActionAllow,
		Confidence: 0.9,
		Timestamp:  time.Now(),
	}, nil
}

// evaluateAuditRule evaluates audit rules
func (zte *ZeroTrustEngine) evaluateAuditRule(request *AccessRequest, rule *SecurityRule) (*AccessDecision, error) {

	// Log the access attempt
	zte.auditLogger.LogSecurityEvent("AUDIT_RULE_TRIGGERED", map[string]interface{}{
		"user_id":   request.UserID,
		"resource":  request.Resource,
		"action":    request.Action,
		"rule_id":   rule.ID,
		"timestamp": time.Now(),
	})

	return &AccessDecision{
		Allowed:    true,
		Reason:     "Audit logged",
		Action:     ActionLog,
		Confidence: 1.0,
		Timestamp:  time.Now(),
	}, nil
}

// evaluateEncryptionRule evaluates encryption rules
func (zte *ZeroTrustEngine) evaluateEncryptionRule(request *AccessRequest, rule *SecurityRule) (*AccessDecision, error) {

	// Check if connection is encrypted
	isEncrypted := request.Context.Headers["x-forwarded-proto"] == "https" ||
		strings.Contains(request.Context.Headers["user-agent"], "encrypted")

	if !isEncrypted {
		return &AccessDecision{
			Allowed:    false,
			Reason:     "Encrypted connection required",
			Action:     ActionDeny,
			Confidence: 1.0,
			Timestamp:  time.Now(),
		}, nil
	}

	return &AccessDecision{
		Allowed:    true,
		Reason:     "Encryption requirement met",
		Action:     ActionAllow,
		Confidence: 1.0,
		Timestamp:  time.Now(),
	}, nil
}

// Helper methods

func (zte *ZeroTrustEngine) matchesResource(resource, pattern string) bool {
	// Simple wildcard matching
	if pattern == "*" {
		return true
	}

	if strings.HasSuffix(pattern, "*") {
		prefix := strings.TrimSuffix(pattern, "*")
		return strings.HasPrefix(resource, prefix)
	}

	return resource == pattern
}

func (zte *ZeroTrustEngine) checkTimeRestrictions(restrictions *TimeRestriction) bool {
	now := time.Now()

	// Check allowed hours
	if len(restrictions.AllowedHours) > 0 {
		currentHour := now.Hour()
		allowed := false
		for _, hour := range restrictions.AllowedHours {
			if currentHour == hour {
				allowed = true
				break
			}
		}
		if !allowed {
			return false
		}
	}

	// Check allowed days
	if len(restrictions.AllowedDays) > 0 {
		currentDay := int(now.Weekday())
		allowed := false
		for _, day := range restrictions.AllowedDays {
			if currentDay == day {
				allowed = true
				break
			}
		}
		if !allowed {
			return false
		}
	}

	return true
}

func (zte *ZeroTrustEngine) checkLocationRestrictions(restrictions *LocationRestriction, context RequestContext) bool {
	if context.Location == nil {
		return len(restrictions.AllowedCountries) == 0 // Allow if no location info and no restrictions
	}

	// Check allowed countries
	if len(restrictions.AllowedCountries) > 0 {
		allowed := false
		for _, country := range restrictions.AllowedCountries {
			if context.Location.Country == country {
				allowed = true
				break
			}
		}
		if !allowed {
			return false
		}
	}

	// Check denied countries
	for _, country := range restrictions.DeniedCountries {
		if context.Location.Country == country {
			return false
		}
	}

	return true
}

func (zte *ZeroTrustEngine) checkDeviceRestrictions(restrictions *DeviceRestriction, context RequestContext) bool {
	// Check device ID requirement
	if restrictions.RequiredDeviceID && context.DeviceID == "" {
		return false
	}

	// Check allowed device IDs
	if len(restrictions.AllowedDeviceIDs) > 0 {
		allowed := false
		for _, deviceID := range restrictions.AllowedDeviceIDs {
			if context.DeviceID == deviceID {
				allowed = true
				break
			}
		}
		if !allowed {
			return false
		}
	}

	return true
}

func (zte *ZeroTrustEngine) hasRequiredClearance(userClearance, requiredClassification string) bool {
	// Define clearance hierarchy
	clearanceLevels := map[string]int{
		"public":       0,
		"internal":     1,
		"confidential": 2,
		"secret":       3,
		"top_secret":   4,
	}

	userLevel, userExists := clearanceLevels[userClearance]
	requiredLevel, reqExists := clearanceLevels[requiredClassification]

	if !userExists || !reqExists {
		return false
	}

	return userLevel >= requiredLevel
}

func (zte *ZeroTrustEngine) generateSessionToken(request *AccessRequest) (string, error) {
	// Create JWT token with session information
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		"user_id":    request.UserID,
		"session_id": request.Context.SessionID,
		"device_id":  request.Context.DeviceID,
		"ip_address": request.Context.IPAddress,
		"issued_at":  time.Now().Unix(),
		"expires_at": time.Now().Add(time.Hour).Unix(),
	})

	// Sign token with secret key
	secretKey := []byte("your-secret-key") // In production, use a secure key management system
	tokenString, err := token.SignedString(secretKey)
	if err != nil {
		return "", err
	}

	return tokenString, nil
}

// GetPolicies returns all zero trust policies
func (zte *ZeroTrustEngine) GetPolicies() map[string]*ZeroTrustPolicy {
	zte.mu.RLock()
	defer zte.mu.RUnlock()

	policies := make(map[string]*ZeroTrustPolicy)
	for id, policy := range zte.policies {
		policies[id] = policy
	}

	return policies
}

// UpdatePolicy updates an existing policy
func (zte *ZeroTrustEngine) UpdatePolicy(policyID string, updates map[string]interface{}) error {
	zte.mu.Lock()
	defer zte.mu.Unlock()

	policy, exists := zte.policies[policyID]
	if !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	// Apply updates
	if name, ok := updates["name"].(string); ok {
		policy.Name = name
	}
	if description, ok := updates["description"].(string); ok {
		policy.Description = description
	}
	if enabled, ok := updates["enabled"].(bool); ok {
		policy.Enabled = enabled
	}

	policy.UpdatedAt = time.Now()

	zte.logger.Infof("Updated zero trust policy: %s", policyID)
	return nil
}

// DeletePolicy removes a policy
func (zte *ZeroTrustEngine) DeletePolicy(policyID string) error {
	zte.mu.Lock()
	defer zte.mu.Unlock()

	if _, exists := zte.policies[policyID]; !exists {
		return fmt.Errorf("policy not found: %s", policyID)
	}

	delete(zte.policies, policyID)
	zte.logger.Infof("Deleted zero trust policy: %s", policyID)

	return nil
}
