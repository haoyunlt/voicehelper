package security

import (
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// ComplianceManager handles compliance monitoring and reporting
type ComplianceManager struct {
	logger      *logrus.Logger
	standards   map[string]*ComplianceStandard
	policies    map[string]*CompliancePolicy
	violations  []*ComplianceViolation
	mu          sync.RWMutex
	config      *ComplianceConfig
	auditLogger *AuditLogger
}

// ComplianceConfig contains compliance configuration
type ComplianceConfig struct {
	EnabledStandards    []string      `json:"enabled_standards"`
	ViolationThreshold  int           `json:"violation_threshold"`
	ReportingInterval   time.Duration `json:"reporting_interval"`
	AutoRemediation     bool          `json:"auto_remediation"`
	NotificationEnabled bool          `json:"notification_enabled"`
	NotificationURL     string        `json:"notification_url"`
}

// ComplianceStandard represents a compliance standard (e.g., GDPR, SOC2, HIPAA)
type ComplianceStandard struct {
	ID             string                     `json:"id"`
	Name           string                     `json:"name"`
	Version        string                     `json:"version"`
	Description    string                     `json:"description"`
	Requirements   []*ComplianceRequirement   `json:"requirements"`
	Controls       []*ComplianceControl       `json:"controls"`
	Certifications []*ComplianceCertification `json:"certifications"`
	LastAssessment *time.Time                 `json:"last_assessment,omitempty"`
	NextAssessment time.Time                  `json:"next_assessment"`
	Status         ComplianceStatus           `json:"status"`
	Metadata       map[string]interface{}     `json:"metadata"`
}

// ComplianceRequirement represents a specific compliance requirement
type ComplianceRequirement struct {
	ID             string                 `json:"id"`
	StandardID     string                 `json:"standard_id"`
	Title          string                 `json:"title"`
	Description    string                 `json:"description"`
	Category       string                 `json:"category"`
	Priority       RequirementPriority    `json:"priority"`
	Controls       []string               `json:"controls"` // Control IDs
	Evidence       []string               `json:"evidence"` // Evidence types required
	TestProcedures []string               `json:"test_procedures"`
	Status         RequirementStatus      `json:"status"`
	LastTested     *time.Time             `json:"last_tested,omitempty"`
	NextTest       time.Time              `json:"next_test"`
	Findings       []*ComplianceFinding   `json:"findings"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// ComplianceControl represents a security control
type ComplianceControl struct {
	ID             string                 `json:"id"`
	Name           string                 `json:"name"`
	Description    string                 `json:"description"`
	Type           ControlType            `json:"type"`
	Category       ControlCategory        `json:"category"`
	Implementation ControlImplementation  `json:"implementation"`
	Effectiveness  ControlEffectiveness   `json:"effectiveness"`
	TestFrequency  time.Duration          `json:"test_frequency"`
	Owner          string                 `json:"owner"`
	Status         ControlStatus          `json:"status"`
	LastTested     *time.Time             `json:"last_tested,omitempty"`
	NextTest       time.Time              `json:"next_test"`
	TestResults    []*ControlTestResult   `json:"test_results"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// ComplianceCertification represents a compliance certification
type ComplianceCertification struct {
	ID             string                 `json:"id"`
	StandardID     string                 `json:"standard_id"`
	Name           string                 `json:"name"`
	Issuer         string                 `json:"issuer"`
	IssuedDate     time.Time              `json:"issued_date"`
	ExpiryDate     time.Time              `json:"expiry_date"`
	Status         CertificationStatus    `json:"status"`
	CertificateURL string                 `json:"certificate_url"`
	Scope          []string               `json:"scope"`
	Metadata       map[string]interface{} `json:"metadata"`
}

// CompliancePolicy represents a compliance policy
type CompliancePolicy struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	Description   string                 `json:"description"`
	StandardID    string                 `json:"standard_id"`
	Requirements  []string               `json:"requirements"` // Requirement IDs
	Rules         []*PolicyRule          `json:"rules"`
	Enforcement   PolicyEnforcement      `json:"enforcement"`
	Status        PolicyStatus           `json:"status"`
	CreatedAt     time.Time              `json:"created_at"`
	UpdatedAt     time.Time              `json:"updated_at"`
	EffectiveDate time.Time              `json:"effective_date"`
	ReviewDate    time.Time              `json:"review_date"`
	Owner         string                 `json:"owner"`
	Approver      string                 `json:"approver"`
	Metadata      map[string]interface{} `json:"metadata"`
}

// PolicyRule represents a rule within a compliance policy
type PolicyRule struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Condition   string                 `json:"condition"`
	Action      PolicyAction           `json:"action"`
	Severity    ViolationSeverity      `json:"severity"`
	Metadata    map[string]interface{} `json:"metadata"`
}

// ComplianceViolation represents a compliance violation
type ComplianceViolation struct {
	ID              string                 `json:"id"`
	StandardID      string                 `json:"standard_id"`
	RequirementID   string                 `json:"requirement_id"`
	PolicyID        string                 `json:"policy_id"`
	RuleID          string                 `json:"rule_id"`
	Title           string                 `json:"title"`
	Description     string                 `json:"description"`
	Severity        ViolationSeverity      `json:"severity"`
	Category        ViolationCategory      `json:"category"`
	Status          ViolationStatus        `json:"status"`
	DetectedAt      time.Time              `json:"detected_at"`
	ResolvedAt      *time.Time             `json:"resolved_at,omitempty"`
	AssignedTo      string                 `json:"assigned_to"`
	Evidence        map[string]interface{} `json:"evidence"`
	RemediationPlan string                 `json:"remediation_plan"`
	DueDate         time.Time              `json:"due_date"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// ControlTestResult represents the result of a control test
type ControlTestResult struct {
	ID        string                 `json:"id"`
	ControlID string                 `json:"control_id"`
	TestDate  time.Time              `json:"test_date"`
	Tester    string                 `json:"tester"`
	Result    TestResult             `json:"result"`
	Score     float64                `json:"score"` // 0.0 to 1.0
	Findings  []*ComplianceFinding   `json:"findings"`
	Evidence  []string               `json:"evidence"`
	Comments  string                 `json:"comments"`
	Metadata  map[string]interface{} `json:"metadata"`
}

// Enums

type ComplianceStatus string

const (
	ComplianceStatusCompliant    ComplianceStatus = "compliant"
	ComplianceStatusNonCompliant ComplianceStatus = "non_compliant"
	ComplianceStatusInProgress   ComplianceStatus = "in_progress"
	ComplianceStatusNotAssessed  ComplianceStatus = "not_assessed"
)

type RequirementPriority string

const (
	RequirementPriorityHigh   RequirementPriority = "high"
	RequirementPriorityMedium RequirementPriority = "medium"
	RequirementPriorityLow    RequirementPriority = "low"
)

type RequirementStatus string

const (
	RequirementStatusImplemented    RequirementStatus = "implemented"
	RequirementStatusPartial        RequirementStatus = "partial"
	RequirementStatusNotImplemented RequirementStatus = "not_implemented"
	RequirementStatusNotApplicable  RequirementStatus = "not_applicable"
)

type ControlType string

const (
	ControlTypePreventive   ControlType = "preventive"
	ControlTypeDetective    ControlType = "detective"
	ControlTypeCorrective   ControlType = "corrective"
	ControlTypeCompensating ControlType = "compensating"
)

type ControlCategory string

const (
	ControlCategoryTechnical      ControlCategory = "technical"
	ControlCategoryAdministrative ControlCategory = "administrative"
	ControlCategoryPhysical       ControlCategory = "physical"
)

type ControlImplementation string

const (
	ControlImplementationManual    ControlImplementation = "manual"
	ControlImplementationAutomatic ControlImplementation = "automatic"
	ControlImplementationHybrid    ControlImplementation = "hybrid"
)

type ControlEffectiveness string

const (
	ControlEffectivenessHigh   ControlEffectiveness = "high"
	ControlEffectivenessMedium ControlEffectiveness = "medium"
	ControlEffectivenessLow    ControlEffectiveness = "low"
)

type ControlStatus string

const (
	ControlStatusActive     ControlStatus = "active"
	ControlStatusInactive   ControlStatus = "inactive"
	ControlStatusTesting    ControlStatus = "testing"
	ControlStatusDeprecated ControlStatus = "deprecated"
)

type CertificationStatus string

const (
	CertificationStatusValid     CertificationStatus = "valid"
	CertificationStatusExpired   CertificationStatus = "expired"
	CertificationStatusSuspended CertificationStatus = "suspended"
	CertificationStatusRevoked   CertificationStatus = "revoked"
)

type PolicyEnforcement string

const (
	PolicyEnforcementMandatory PolicyEnforcement = "mandatory"
	PolicyEnforcementAdvisory  PolicyEnforcement = "advisory"
	PolicyEnforcementOptional  PolicyEnforcement = "optional"
)

type PolicyStatus string

const (
	PolicyStatusActive     PolicyStatus = "active"
	PolicyStatusDraft      PolicyStatus = "draft"
	PolicyStatusArchived   PolicyStatus = "archived"
	PolicyStatusSuperseded PolicyStatus = "superseded"
)

type PolicyAction string

const (
	PolicyActionAllow     PolicyAction = "allow"
	PolicyActionDeny      PolicyAction = "deny"
	PolicyActionAlert     PolicyAction = "alert"
	PolicyActionLog       PolicyAction = "log"
	PolicyActionRemediate PolicyAction = "remediate"
)

type ViolationSeverity string

const (
	ViolationSeverityCritical ViolationSeverity = "critical"
	ViolationSeverityHigh     ViolationSeverity = "high"
	ViolationSeverityMedium   ViolationSeverity = "medium"
	ViolationSeverityLow      ViolationSeverity = "low"
	ViolationSeverityInfo     ViolationSeverity = "info"
)

type ViolationCategory string

const (
	ViolationCategoryDataProtection ViolationCategory = "data_protection"
	ViolationCategoryAccessControl  ViolationCategory = "access_control"
	ViolationCategoryEncryption     ViolationCategory = "encryption"
	ViolationCategoryAudit          ViolationCategory = "audit"
	ViolationCategoryIncident       ViolationCategory = "incident"
	ViolationCategoryConfiguration  ViolationCategory = "configuration"
)

type ViolationStatus string

const (
	ViolationStatusOpen          ViolationStatus = "open"
	ViolationStatusInProgress    ViolationStatus = "in_progress"
	ViolationStatusResolved      ViolationStatus = "resolved"
	ViolationStatusClosed        ViolationStatus = "closed"
	ViolationStatusFalsePositive ViolationStatus = "false_positive"
)

type TestResult string

const (
	TestResultPass      TestResult = "pass"
	TestResultFail      TestResult = "fail"
	TestResultPartial   TestResult = "partial"
	TestResultNotTested TestResult = "not_tested"
)

// NewComplianceManager creates a new compliance manager
func NewComplianceManager() *ComplianceManager {
	config := &ComplianceConfig{
		EnabledStandards:    []string{"GDPR", "SOC2", "ISO27001"},
		ViolationThreshold:  10,
		ReportingInterval:   24 * time.Hour,
		AutoRemediation:     false,
		NotificationEnabled: true,
	}

	manager := &ComplianceManager{
		logger:      logrus.New(),
		standards:   make(map[string]*ComplianceStandard),
		policies:    make(map[string]*CompliancePolicy),
		violations:  make([]*ComplianceViolation, 0),
		config:      config,
		auditLogger: NewAuditLogger(),
	}

	// Initialize default compliance standards
	manager.initializeStandards()

	// Start background monitoring
	go manager.monitoringRoutine()

	return manager
}

// initializeStandards initializes default compliance standards
func (cm *ComplianceManager) initializeStandards() {
	// GDPR
	gdpr := &ComplianceStandard{
		ID:             "GDPR",
		Name:           "General Data Protection Regulation",
		Version:        "2018",
		Description:    "EU regulation on data protection and privacy",
		Status:         ComplianceStatusInProgress,
		NextAssessment: time.Now().AddDate(0, 6, 0), // 6 months
		Requirements: []*ComplianceRequirement{
			{
				ID:          "GDPR-7",
				StandardID:  "GDPR",
				Title:       "Consent",
				Description: "Consent must be freely given, specific, informed and unambiguous",
				Category:    "Data Processing",
				Priority:    RequirementPriorityHigh,
				Status:      RequirementStatusImplemented,
				NextTest:    time.Now().AddDate(0, 3, 0),
			},
			{
				ID:          "GDPR-25",
				StandardID:  "GDPR",
				Title:       "Data Protection by Design and by Default",
				Description: "Implement appropriate technical and organisational measures",
				Category:    "Data Protection",
				Priority:    RequirementPriorityHigh,
				Status:      RequirementStatusPartial,
				NextTest:    time.Now().AddDate(0, 1, 0),
			},
			{
				ID:          "GDPR-32",
				StandardID:  "GDPR",
				Title:       "Security of Processing",
				Description: "Implement appropriate technical and organisational measures",
				Category:    "Security",
				Priority:    RequirementPriorityHigh,
				Status:      RequirementStatusImplemented,
				NextTest:    time.Now().AddDate(0, 3, 0),
			},
		},
		Controls: []*ComplianceControl{
			{
				ID:             "GDPR-C001",
				Name:           "Data Encryption",
				Description:    "Encrypt personal data at rest and in transit",
				Type:           ControlTypePreventive,
				Category:       ControlCategoryTechnical,
				Implementation: ControlImplementationAutomatic,
				Effectiveness:  ControlEffectivenessHigh,
				TestFrequency:  30 * 24 * time.Hour, // Monthly
				Status:         ControlStatusActive,
				NextTest:       time.Now().AddDate(0, 1, 0),
			},
			{
				ID:             "GDPR-C002",
				Name:           "Access Control",
				Description:    "Implement role-based access control for personal data",
				Type:           ControlTypePreventive,
				Category:       ControlCategoryTechnical,
				Implementation: ControlImplementationAutomatic,
				Effectiveness:  ControlEffectivenessHigh,
				TestFrequency:  30 * 24 * time.Hour,
				Status:         ControlStatusActive,
				NextTest:       time.Now().AddDate(0, 1, 0),
			},
		},
	}

	// SOC 2
	soc2 := &ComplianceStandard{
		ID:             "SOC2",
		Name:           "Service Organization Control 2",
		Version:        "2017",
		Description:    "Auditing standard for service organizations",
		Status:         ComplianceStatusInProgress,
		NextAssessment: time.Now().AddDate(1, 0, 0), // Annual
		Requirements: []*ComplianceRequirement{
			{
				ID:          "SOC2-CC6.1",
				StandardID:  "SOC2",
				Title:       "Logical and Physical Access Controls",
				Description: "Implement controls to restrict logical and physical access",
				Category:    "Access Control",
				Priority:    RequirementPriorityHigh,
				Status:      RequirementStatusImplemented,
				NextTest:    time.Now().AddDate(0, 3, 0),
			},
			{
				ID:          "SOC2-CC6.7",
				StandardID:  "SOC2",
				Title:       "Data Transmission and Disposal",
				Description: "Protect data during transmission and disposal",
				Category:    "Data Protection",
				Priority:    RequirementPriorityHigh,
				Status:      RequirementStatusImplemented,
				NextTest:    time.Now().AddDate(0, 3, 0),
			},
		},
		Controls: []*ComplianceControl{
			{
				ID:             "SOC2-C001",
				Name:           "Multi-Factor Authentication",
				Description:    "Require MFA for system access",
				Type:           ControlTypePreventive,
				Category:       ControlCategoryTechnical,
				Implementation: ControlImplementationAutomatic,
				Effectiveness:  ControlEffectivenessHigh,
				TestFrequency:  90 * 24 * time.Hour, // Quarterly
				Status:         ControlStatusActive,
				NextTest:       time.Now().AddDate(0, 3, 0),
			},
		},
	}

	cm.standards["GDPR"] = gdpr
	cm.standards["SOC2"] = soc2

	cm.logger.Info("Initialized compliance standards")
}

// CheckCompliance evaluates compliance for a specific access request
func (cm *ComplianceManager) CheckCompliance(request *AccessRequest) ([]*ComplianceViolation, error) {
	var violations []*ComplianceViolation

	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Check each enabled standard
	for _, standardID := range cm.config.EnabledStandards {
		standard, exists := cm.standards[standardID]
		if !exists {
			continue
		}

		// Check policies for this standard
		for _, policy := range cm.policies {
			if policy.StandardID == standardID && policy.Status == PolicyStatusActive {
				policyViolations := cm.evaluatePolicy(request, policy)
				violations = append(violations, policyViolations...)
			}
		}

		// Check specific requirements
		standardViolations := cm.evaluateStandardRequirements(request, standard)
		violations = append(violations, standardViolations...)
	}

	// Log violations
	for _, violation := range violations {
		cm.auditLogger.LogComplianceEvent("COMPLIANCE_VIOLATION", "VIOLATION", map[string]interface{}{
			"violation_id":   violation.ID,
			"standard_id":    violation.StandardID,
			"requirement_id": violation.RequirementID,
			"severity":       violation.Severity,
			"description":    violation.Description,
			"user_id":        request.UserID,
			"resource":       request.Resource,
		})
	}

	return violations, nil
}

// evaluatePolicy evaluates a compliance policy against an access request
func (cm *ComplianceManager) evaluatePolicy(request *AccessRequest, policy *CompliancePolicy) []*ComplianceViolation {
	var violations []*ComplianceViolation

	for _, rule := range policy.Rules {
		if cm.evaluateRule(request, rule) {
			violation := &ComplianceViolation{
				ID:          generateViolationID(),
				StandardID:  policy.StandardID,
				PolicyID:    policy.ID,
				RuleID:      rule.ID,
				Title:       fmt.Sprintf("Policy violation: %s", rule.Name),
				Description: rule.Description,
				Severity:    rule.Severity,
				Category:    cm.categorizeViolation(rule),
				Status:      ViolationStatusOpen,
				DetectedAt:  time.Now(),
				DueDate:     time.Now().AddDate(0, 0, cm.getSeverityDueDays(rule.Severity)),
				Evidence: map[string]interface{}{
					"user_id":        request.UserID,
					"resource":       request.Resource,
					"action":         request.Action,
					"ip_address":     request.Context.IPAddress,
					"rule_condition": rule.Condition,
				},
			}

			violations = append(violations, violation)
		}
	}

	return violations
}

// evaluateStandardRequirements evaluates standard requirements
func (cm *ComplianceManager) evaluateStandardRequirements(request *AccessRequest, standard *ComplianceStandard) []*ComplianceViolation {
	var violations []*ComplianceViolation

	// Example: Check GDPR data processing requirements
	if standard.ID == "GDPR" {
		violations = append(violations, cm.checkGDPRCompliance(request)...)
	}

	// Example: Check SOC2 access control requirements
	if standard.ID == "SOC2" {
		violations = append(violations, cm.checkSOC2Compliance(request)...)
	}

	return violations
}

// checkGDPRCompliance checks GDPR-specific compliance
func (cm *ComplianceManager) checkGDPRCompliance(request *AccessRequest) []*ComplianceViolation {
	var violations []*ComplianceViolation

	// Check if accessing personal data without proper consent
	if cm.isPersonalDataResource(request.Resource) {
		if !cm.hasValidConsent(request) {
			violation := &ComplianceViolation{
				ID:            generateViolationID(),
				StandardID:    "GDPR",
				RequirementID: "GDPR-7",
				Title:         "Access to personal data without valid consent",
				Description:   "Personal data accessed without proper user consent",
				Severity:      ViolationSeverityHigh,
				Category:      ViolationCategoryDataProtection,
				Status:        ViolationStatusOpen,
				DetectedAt:    time.Now(),
				DueDate:       time.Now().AddDate(0, 0, 7), // 7 days for high severity
				Evidence: map[string]interface{}{
					"user_id":  request.UserID,
					"resource": request.Resource,
					"reason":   "No valid consent found",
				},
			}
			violations = append(violations, violation)
		}
	}

	// Check data retention compliance
	if cm.isDataRetentionViolation(request) {
		violation := &ComplianceViolation{
			ID:            generateViolationID(),
			StandardID:    "GDPR",
			RequirementID: "GDPR-5",
			Title:         "Data retention period exceeded",
			Description:   "Data accessed beyond permitted retention period",
			Severity:      ViolationSeverityMedium,
			Category:      ViolationCategoryDataProtection,
			Status:        ViolationStatusOpen,
			DetectedAt:    time.Now(),
			DueDate:       time.Now().AddDate(0, 0, 14), // 14 days for medium severity
			Evidence: map[string]interface{}{
				"user_id":  request.UserID,
				"resource": request.Resource,
				"reason":   "Data retention period exceeded",
			},
		}
		violations = append(violations, violation)
	}

	return violations
}

// checkSOC2Compliance checks SOC2-specific compliance
func (cm *ComplianceManager) checkSOC2Compliance(request *AccessRequest) []*ComplianceViolation {
	var violations []*ComplianceViolation

	// Check access control requirements
	if !cm.hasProperAccessControl(request) {
		violation := &ComplianceViolation{
			ID:            generateViolationID(),
			StandardID:    "SOC2",
			RequirementID: "SOC2-CC6.1",
			Title:         "Insufficient access controls",
			Description:   "Access granted without proper authorization controls",
			Severity:      ViolationSeverityHigh,
			Category:      ViolationCategoryAccessControl,
			Status:        ViolationStatusOpen,
			DetectedAt:    time.Now(),
			DueDate:       time.Now().AddDate(0, 0, 7),
			Evidence: map[string]interface{}{
				"user_id":  request.UserID,
				"resource": request.Resource,
				"reason":   "Insufficient access controls",
			},
		}
		violations = append(violations, violation)
	}

	return violations
}

// Helper methods for compliance checks

func (cm *ComplianceManager) evaluateRule(request *AccessRequest, rule *PolicyRule) bool {
	// Simplified rule evaluation - in production, use a proper rule engine
	condition := strings.ToLower(rule.Condition)

	if strings.Contains(condition, "high_risk_country") {
		if request.Context.Location != nil {
			highRiskCountries := []string{"CN", "RU", "KP", "IR"}
			for _, country := range highRiskCountries {
				if request.Context.Location.Country == country {
					return true
				}
			}
		}
	}

	if strings.Contains(condition, "off_hours") {
		hour := time.Now().Hour()
		if hour < 6 || hour > 22 {
			return true
		}
	}

	if strings.Contains(condition, "no_mfa") {
		// Check if MFA was used (simplified check)
		if mfa, exists := request.Claims["mfa"]; !exists || mfa != true {
			return true
		}
	}

	return false
}

func (cm *ComplianceManager) isPersonalDataResource(resource string) bool {
	personalDataResources := []string{
		"/api/v1/users",
		"/api/v1/profiles",
		"/api/v1/conversations",
		"/api/v1/messages",
	}

	for _, pdr := range personalDataResources {
		if strings.HasPrefix(resource, pdr) {
			return true
		}
	}

	return false
}

func (cm *ComplianceManager) hasValidConsent(request *AccessRequest) bool {
	// Check if user has given valid consent
	// This would integrate with a consent management system
	if consent, exists := request.Claims["consent"]; exists {
		if consentMap, ok := consent.(map[string]interface{}); ok {
			if valid, exists := consentMap["valid"]; exists && valid == true {
				return true
			}
		}
	}
	return false
}

func (cm *ComplianceManager) isDataRetentionViolation(request *AccessRequest) bool {
	// Check if data is being accessed beyond retention period
	// This would integrate with a data lifecycle management system
	return false // Simplified implementation
}

func (cm *ComplianceManager) hasProperAccessControl(request *AccessRequest) bool {
	// Check if proper access controls are in place
	if request.Context.SessionID == "" {
		return false
	}

	if len(request.Claims) == 0 {
		return false
	}

	// Check for required roles
	if roles, exists := request.Claims["roles"]; exists {
		if roleList, ok := roles.([]interface{}); ok && len(roleList) > 0 {
			return true
		}
	}

	return false
}

func (cm *ComplianceManager) categorizeViolation(rule *PolicyRule) ViolationCategory {
	ruleName := strings.ToLower(rule.Name)

	if strings.Contains(ruleName, "data") || strings.Contains(ruleName, "privacy") {
		return ViolationCategoryDataProtection
	}
	if strings.Contains(ruleName, "access") || strings.Contains(ruleName, "auth") {
		return ViolationCategoryAccessControl
	}
	if strings.Contains(ruleName, "encrypt") {
		return ViolationCategoryEncryption
	}
	if strings.Contains(ruleName, "audit") || strings.Contains(ruleName, "log") {
		return ViolationCategoryAudit
	}

	return ViolationCategoryConfiguration
}

func (cm *ComplianceManager) getSeverityDueDays(severity ViolationSeverity) int {
	switch severity {
	case ViolationSeverityCritical:
		return 1
	case ViolationSeverityHigh:
		return 7
	case ViolationSeverityMedium:
		return 14
	case ViolationSeverityLow:
		return 30
	default:
		return 30
	}
}

// Background monitoring

func (cm *ComplianceManager) monitoringRoutine() {
	ticker := time.NewTicker(cm.config.ReportingInterval)
	defer ticker.Stop()

	for range ticker.C {
		cm.performPeriodicAssessment()
		cm.generateComplianceReport()
		cm.checkCertificationExpiry()
	}
}

func (cm *ComplianceManager) performPeriodicAssessment() {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	for _, standard := range cm.standards {
		if time.Now().After(standard.NextAssessment) {
			cm.logger.Infof("Performing compliance assessment for %s", standard.ID)

			// Update next assessment date
			standard.NextAssessment = time.Now().Add(365 * 24 * time.Hour) // Annual
			now := time.Now()
			standard.LastAssessment = &now

			// Assess requirements
			for _, requirement := range standard.Requirements {
				if time.Now().After(requirement.NextTest) {
					cm.testRequirement(requirement)
				}
			}

			// Test controls
			for _, control := range standard.Controls {
				if time.Now().After(control.NextTest) {
					cm.testControl(control)
				}
			}
		}
	}
}

func (cm *ComplianceManager) testRequirement(requirement *ComplianceRequirement) {
	// Simulate requirement testing
	cm.logger.Infof("Testing requirement: %s", requirement.ID)

	now := time.Now()
	requirement.LastTested = &now
	requirement.NextTest = time.Now().Add(90 * 24 * time.Hour) // Quarterly

	// In production, this would perform actual compliance tests
}

func (cm *ComplianceManager) testControl(control *ComplianceControl) {
	// Simulate control testing
	cm.logger.Infof("Testing control: %s", control.ID)

	now := time.Now()
	control.LastTested = &now
	control.NextTest = time.Now().Add(control.TestFrequency)

	// Create test result
	testResult := &ControlTestResult{
		ID:        generateTestResultID(),
		ControlID: control.ID,
		TestDate:  time.Now(),
		Tester:    "automated",
		Result:    TestResultPass, // Simplified
		Score:     0.95,
		Comments:  "Automated compliance test passed",
	}

	control.TestResults = append(control.TestResults, testResult)

	// Keep only recent test results
	if len(control.TestResults) > 10 {
		control.TestResults = control.TestResults[1:]
	}
}

func (cm *ComplianceManager) generateComplianceReport() {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	report := &ComplianceReport{
		ID:          generateEventID(),
		StartTime:   time.Now().AddDate(0, 0, -30), // Last 30 days
		EndTime:     time.Now(),
		Standards:   cm.config.EnabledStandards,
		GeneratedAt: time.Now(),
		TotalEvents: len(cm.violations),
		Violations:  len(cm.violations),
	}

	cm.logger.Infof("Generated compliance report: %s", report.ID)

	// In production, this would save the report and send notifications
}

func (cm *ComplianceManager) checkCertificationExpiry() {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	for _, standard := range cm.standards {
		for _, cert := range standard.Certifications {
			if cert.Status == CertificationStatusValid {
				daysUntilExpiry := int(time.Until(cert.ExpiryDate).Hours() / 24)

				if daysUntilExpiry <= 30 {
					cm.logger.Warnf("Certification %s expires in %d days", cert.Name, daysUntilExpiry)

					// Create violation for expiring certification
					violation := &ComplianceViolation{
						ID:          generateViolationID(),
						StandardID:  cert.StandardID,
						Title:       fmt.Sprintf("Certification expiring: %s", cert.Name),
						Description: fmt.Sprintf("Certification %s expires on %s", cert.Name, cert.ExpiryDate.Format("2006-01-02")),
						Severity:    ViolationSeverityMedium,
						Category:    ViolationCategoryConfiguration,
						Status:      ViolationStatusOpen,
						DetectedAt:  time.Now(),
						DueDate:     cert.ExpiryDate,
						Evidence: map[string]interface{}{
							"certification_id": cert.ID,
							"expiry_date":      cert.ExpiryDate,
							"days_remaining":   daysUntilExpiry,
						},
					}

					cm.violations = append(cm.violations, violation)
				}
			}
		}
	}
}

// Public methods

// GetComplianceStatus returns the overall compliance status
func (cm *ComplianceManager) GetComplianceStatus() map[string]ComplianceStatus {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	status := make(map[string]ComplianceStatus)
	for id, standard := range cm.standards {
		status[id] = standard.Status
	}

	return status
}

// GetViolations returns current compliance violations
func (cm *ComplianceManager) GetViolations() []*ComplianceViolation {
	cm.mu.RLock()
	defer cm.mu.RUnlock()

	// Return a copy to prevent external modification
	violations := make([]*ComplianceViolation, len(cm.violations))
	copy(violations, cm.violations)

	return violations
}

// ResolveViolation marks a violation as resolved
func (cm *ComplianceManager) ResolveViolation(violationID, resolution string) error {
	cm.mu.Lock()
	defer cm.mu.Unlock()

	for _, violation := range cm.violations {
		if violation.ID == violationID {
			violation.Status = ViolationStatusResolved
			now := time.Now()
			violation.ResolvedAt = &now
			violation.RemediationPlan = resolution

			cm.auditLogger.LogComplianceEvent("VIOLATION_RESOLVED", "RESOLVED", map[string]interface{}{
				"violation_id": violationID,
				"resolution":   resolution,
			})

			cm.logger.Infof("Resolved compliance violation: %s", violationID)
			return nil
		}
	}

	return fmt.Errorf("violation not found: %s", violationID)
}

// Utility functions

func generateViolationID() string {
	return fmt.Sprintf("violation_%d", time.Now().UnixNano())
}

func generateTestResultID() string {
	return fmt.Sprintf("test_%d", time.Now().UnixNano())
}
