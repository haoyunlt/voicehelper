package handler

import (
	"net/http"
	"strconv"
	"time"

	"chatbot/backend/pkg/security"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// SecurityAdminHandler handles security administration endpoints
type SecurityAdminHandler struct {
	zeroTrustEngine   *security.ZeroTrustEngine
	complianceManager *security.ComplianceManager
	encryptionManager *security.EncryptionManager
	auditLogger       *security.AuditLogger
	logger            *logrus.Logger
}

// NewSecurityAdminHandler creates a new security admin handler
func NewSecurityAdminHandler() *SecurityAdminHandler {
	return &SecurityAdminHandler{
		zeroTrustEngine:   security.NewZeroTrustEngine(),
		complianceManager: security.NewComplianceManager(),
		encryptionManager: security.NewEncryptionManager(),
		auditLogger:       security.NewAuditLogger(),
		logger:            logrus.New(),
	}
}

// Zero Trust Policy Management

// GetPolicies returns all zero trust policies
func (sah *SecurityAdminHandler) GetPolicies(c *gin.Context) {
	policies := sah.zeroTrustEngine.GetPolicies()

	c.JSON(http.StatusOK, gin.H{
		"policies": policies,
		"count":    len(policies),
	})
}

// CreatePolicy creates a new zero trust policy
func (sah *SecurityAdminHandler) CreatePolicy(c *gin.Context) {
	var policy security.ZeroTrustPolicy
	if err := c.ShouldBindJSON(&policy); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid policy data",
			"details": err.Error(),
		})
		return
	}

	if err := sah.zeroTrustEngine.AddPolicy(&policy); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to create policy",
			"details": err.Error(),
		})
		return
	}

	// Log the policy creation
	userID := sah.getUserID(c)
	sah.auditLogger.LogConfigurationChange(userID, "zero_trust_policy", "create", nil, policy)

	c.JSON(http.StatusCreated, gin.H{
		"message": "Policy created successfully",
		"policy":  policy,
	})
}

// UpdatePolicy updates an existing zero trust policy
func (sah *SecurityAdminHandler) UpdatePolicy(c *gin.Context) {
	policyID := c.Param("id")

	var updates map[string]interface{}
	if err := c.ShouldBindJSON(&updates); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid update data",
			"details": err.Error(),
		})
		return
	}

	if err := sah.zeroTrustEngine.UpdatePolicy(policyID, updates); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to update policy",
			"details": err.Error(),
		})
		return
	}

	// Log the policy update
	userID := sah.getUserID(c)
	sah.auditLogger.LogConfigurationChange(userID, "zero_trust_policy", "update", policyID, updates)

	c.JSON(http.StatusOK, gin.H{
		"message": "Policy updated successfully",
	})
}

// DeletePolicy deletes a zero trust policy
func (sah *SecurityAdminHandler) DeletePolicy(c *gin.Context) {
	policyID := c.Param("id")

	if err := sah.zeroTrustEngine.DeletePolicy(policyID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to delete policy",
			"details": err.Error(),
		})
		return
	}

	// Log the policy deletion
	userID := sah.getUserID(c)
	sah.auditLogger.LogConfigurationChange(userID, "zero_trust_policy", "delete", policyID, nil)

	c.JSON(http.StatusOK, gin.H{
		"message": "Policy deleted successfully",
	})
}

// Compliance Management

// GetComplianceStatus returns overall compliance status
func (sah *SecurityAdminHandler) GetComplianceStatus(c *gin.Context) {
	status := sah.complianceManager.GetComplianceStatus()

	c.JSON(http.StatusOK, gin.H{
		"compliance_status": status,
	})
}

// GetViolations returns current compliance violations
func (sah *SecurityAdminHandler) GetViolations(c *gin.Context) {
	violations := sah.complianceManager.GetViolations()

	// Filter by severity if specified
	severity := c.Query("severity")
	if severity != "" {
		filteredViolations := make([]*security.ComplianceViolation, 0)
		for _, violation := range violations {
			if string(violation.Severity) == severity {
				filteredViolations = append(filteredViolations, violation)
			}
		}
		violations = filteredViolations
	}

	// Filter by status if specified
	status := c.Query("status")
	if status != "" {
		filteredViolations := make([]*security.ComplianceViolation, 0)
		for _, violation := range violations {
			if string(violation.Status) == status {
				filteredViolations = append(filteredViolations, violation)
			}
		}
		violations = filteredViolations
	}

	c.JSON(http.StatusOK, gin.H{
		"violations": violations,
		"count":      len(violations),
	})
}

// ResolveViolation marks a compliance violation as resolved
func (sah *SecurityAdminHandler) ResolveViolation(c *gin.Context) {
	violationID := c.Param("id")

	var request struct {
		Resolution string `json:"resolution" binding:"required"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid resolution data",
			"details": err.Error(),
		})
		return
	}

	if err := sah.complianceManager.ResolveViolation(violationID, request.Resolution); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to resolve violation",
			"details": err.Error(),
		})
		return
	}

	// Log the violation resolution
	userID := sah.getUserID(c)
	sah.auditLogger.LogComplianceEvent("VIOLATION_RESOLVED", "RESOLVED", map[string]interface{}{
		"violation_id": violationID,
		"resolved_by":  userID,
		"resolution":   request.Resolution,
	})

	c.JSON(http.StatusOK, gin.H{
		"message": "Violation resolved successfully",
	})
}

// Encryption Key Management

// GetKeys returns all encryption keys (without key data)
func (sah *SecurityAdminHandler) GetKeys(c *gin.Context) {
	keys := sah.encryptionManager.ListKeys()

	// Filter by purpose if specified
	purpose := c.Query("purpose")
	if purpose != "" {
		filteredKeys := make([]*security.EncryptionKey, 0)
		for _, key := range keys {
			if string(key.Purpose) == purpose {
				filteredKeys = append(filteredKeys, key)
			}
		}
		keys = filteredKeys
	}

	c.JSON(http.StatusOK, gin.H{
		"keys":  keys,
		"count": len(keys),
	})
}

// CreateKey generates a new encryption key
func (sah *SecurityAdminHandler) CreateKey(c *gin.Context) {
	var request struct {
		Purpose        string `json:"purpose" binding:"required"`
		ExpirationDays int    `json:"expiration_days"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid key creation data",
			"details": err.Error(),
		})
		return
	}

	if request.ExpirationDays <= 0 {
		request.ExpirationDays = 365 // Default to 1 year
	}

	purpose := security.KeyPurpose(request.Purpose)
	key, err := sah.encryptionManager.GenerateKey(purpose, request.ExpirationDays)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to generate key",
			"details": err.Error(),
		})
		return
	}

	// Log the key creation
	userID := sah.getUserID(c)
	sah.auditLogger.LogConfigurationChange(userID, "encryption_key", "create", nil, map[string]interface{}{
		"key_id":  key.ID,
		"purpose": key.Purpose,
	})

	// Return key without sensitive data
	keyResponse := map[string]interface{}{
		"id":         key.ID,
		"algorithm":  key.Algorithm,
		"purpose":    key.Purpose,
		"status":     key.Status,
		"created_at": key.CreatedAt,
		"expires_at": key.ExpiresAt,
	}

	c.JSON(http.StatusCreated, gin.H{
		"message": "Key generated successfully",
		"key":     keyResponse,
	})
}

// RotateKey rotates an encryption key
func (sah *SecurityAdminHandler) RotateKey(c *gin.Context) {
	keyID := c.Param("id")

	newKey, err := sah.encryptionManager.RotateKey(keyID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to rotate key",
			"details": err.Error(),
		})
		return
	}

	// Log the key rotation
	userID := sah.getUserID(c)
	sah.auditLogger.LogConfigurationChange(userID, "encryption_key", "rotate", keyID, newKey.ID)

	// Return new key without sensitive data
	keyResponse := map[string]interface{}{
		"id":         newKey.ID,
		"algorithm":  newKey.Algorithm,
		"purpose":    newKey.Purpose,
		"status":     newKey.Status,
		"created_at": newKey.CreatedAt,
		"expires_at": newKey.ExpiresAt,
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "Key rotated successfully",
		"new_key": keyResponse,
	})
}

// RevokeKey revokes an encryption key
func (sah *SecurityAdminHandler) RevokeKey(c *gin.Context) {
	keyID := c.Param("id")

	var request struct {
		Reason string `json:"reason" binding:"required"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid revocation data",
			"details": err.Error(),
		})
		return
	}

	if err := sah.encryptionManager.RevokeKey(keyID, request.Reason); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to revoke key",
			"details": err.Error(),
		})
		return
	}

	// Log the key revocation
	userID := sah.getUserID(c)
	sah.auditLogger.LogConfigurationChange(userID, "encryption_key", "revoke", keyID, request.Reason)

	c.JSON(http.StatusOK, gin.H{
		"message": "Key revoked successfully",
	})
}

// GetKeyUsageStats returns key usage statistics
func (sah *SecurityAdminHandler) GetKeyUsageStats(c *gin.Context) {
	stats := sah.encryptionManager.GetKeyUsageStats()

	c.JSON(http.StatusOK, gin.H{
		"usage_stats": stats,
	})
}

// Audit and Monitoring

// GetAuditEvents returns audit events
func (sah *SecurityAdminHandler) GetAuditEvents(c *gin.Context) {
	// Parse query parameters
	filters := make(map[string]interface{})

	if userID := c.Query("user_id"); userID != "" {
		filters["user_id"] = userID
	}

	if eventType := c.Query("type"); eventType != "" {
		filters["type"] = eventType
	}

	if severity := c.Query("severity"); severity != "" {
		filters["severity"] = severity
	}

	if category := c.Query("category"); category != "" {
		filters["category"] = category
	}

	// Parse pagination
	limit := 50
	if limitStr := c.Query("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 1000 {
			limit = l
		}
	}

	offset := 0
	if offsetStr := c.Query("offset"); offsetStr != "" {
		if o, err := strconv.Atoi(offsetStr); err == nil && o >= 0 {
			offset = o
		}
	}

	filters["limit"] = limit
	filters["offset"] = offset

	events, err := sah.auditLogger.GetAuditEvents(filters)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to retrieve audit events",
			"details": err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"events": events,
		"count":  len(events),
		"limit":  limit,
		"offset": offset,
	})
}

// GenerateComplianceReport generates a compliance report
func (sah *SecurityAdminHandler) GenerateComplianceReport(c *gin.Context) {
	var request struct {
		Standards []string `json:"standards"`
		StartDate string   `json:"start_date"`
		EndDate   string   `json:"end_date"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error":   "Invalid report request data",
			"details": err.Error(),
		})
		return
	}

	// Parse dates (simplified)
	startTime := time.Now().AddDate(0, 0, -30) // Default to last 30 days
	endTime := time.Now()

	report, err := sah.auditLogger.GenerateComplianceReport(startTime, endTime, request.Standards)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":   "Failed to generate compliance report",
			"details": err.Error(),
		})
		return
	}

	// Log the report generation
	userID := sah.getUserID(c)
	sah.auditLogger.LogSecurityEvent("COMPLIANCE_REPORT_GENERATED", map[string]interface{}{
		"generated_by": userID,
		"report_id":    report.ID,
		"standards":    request.Standards,
	})

	c.JSON(http.StatusOK, gin.H{
		"report": report,
	})
}

// Security Dashboard

// GetSecurityDashboard returns security dashboard data
func (sah *SecurityAdminHandler) GetSecurityDashboard(c *gin.Context) {
	// Get compliance status
	complianceStatus := sah.complianceManager.GetComplianceStatus()

	// Get violations summary
	violations := sah.complianceManager.GetViolations()
	violationsSummary := sah.summarizeViolations(violations)

	// Get key statistics
	keyStats := sah.encryptionManager.GetKeyUsageStats()
	keysSummary := sah.summarizeKeys(keyStats)

	// Get policy count
	policies := sah.zeroTrustEngine.GetPolicies()

	dashboard := map[string]interface{}{
		"compliance": map[string]interface{}{
			"status":     complianceStatus,
			"violations": violationsSummary,
		},
		"encryption": map[string]interface{}{
			"keys": keysSummary,
		},
		"policies": map[string]interface{}{
			"total":  len(policies),
			"active": sah.countActivePolicies(policies),
		},
		"timestamp": time.Now(),
	}

	c.JSON(http.StatusOK, dashboard)
}

// Helper methods

func (sah *SecurityAdminHandler) getUserID(c *gin.Context) string {
	if userID, exists := c.Get("user_id"); exists {
		if userIDStr, ok := userID.(string); ok {
			return userIDStr
		}
	}
	return "unknown"
}

func (sah *SecurityAdminHandler) summarizeViolations(violations []*security.ComplianceViolation) map[string]interface{} {
	summary := map[string]interface{}{
		"total": len(violations),
		"by_severity": map[string]int{
			"critical": 0,
			"high":     0,
			"medium":   0,
			"low":      0,
			"info":     0,
		},
		"by_status": map[string]int{
			"open":        0,
			"in_progress": 0,
			"resolved":    0,
			"closed":      0,
		},
	}

	for _, violation := range violations {
		// Count by severity
		severityMap := summary["by_severity"].(map[string]int)
		severityMap[string(violation.Severity)]++

		// Count by status
		statusMap := summary["by_status"].(map[string]int)
		statusMap[string(violation.Status)]++
	}

	return summary
}

func (sah *SecurityAdminHandler) summarizeKeys(keyStats map[string]*security.KeyUsage) map[string]interface{} {
	summary := map[string]interface{}{
		"total":                 len(keyStats),
		"total_encryptions":     int64(0),
		"total_decryptions":     int64(0),
		"total_bytes_encrypted": int64(0),
		"total_bytes_decrypted": int64(0),
	}

	for _, usage := range keyStats {
		summary["total_encryptions"] = summary["total_encryptions"].(int64) + int64(usage.EncryptionCount)
		summary["total_decryptions"] = summary["total_decryptions"].(int64) + int64(usage.DecryptionCount)
		summary["total_bytes_encrypted"] = summary["total_bytes_encrypted"].(int64) + usage.BytesEncrypted
		summary["total_bytes_decrypted"] = summary["total_bytes_decrypted"].(int64) + usage.BytesDecrypted
	}

	return summary
}

func (sah *SecurityAdminHandler) countActivePolicies(policies map[string]*security.ZeroTrustPolicy) int {
	count := 0
	for _, policy := range policies {
		if policy.Enabled {
			count++
		}
	}
	return count
}
