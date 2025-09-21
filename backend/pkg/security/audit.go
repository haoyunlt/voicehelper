package security

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// AuditEvent represents a security audit event
type AuditEvent struct {
	ID        string                 `json:"id"`
	Type      string                 `json:"type"`
	UserID    string                 `json:"user_id,omitempty"`
	Resource  string                 `json:"resource,omitempty"`
	Action    string                 `json:"action,omitempty"`
	Result    string                 `json:"result"`
	IPAddress string                 `json:"ip_address,omitempty"`
	UserAgent string                 `json:"user_agent,omitempty"`
	SessionID string                 `json:"session_id,omitempty"`
	DeviceID  string                 `json:"device_id,omitempty"`
	Location  *Location              `json:"location,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
	Timestamp time.Time              `json:"timestamp"`
	Severity  AuditSeverity          `json:"severity"`
	Category  AuditCategory          `json:"category"`
}

// AuditSeverity represents the severity level of an audit event
type AuditSeverity string

const (
	SeverityInfo     AuditSeverity = "info"
	SeverityWarning  AuditSeverity = "warning"
	SeverityError    AuditSeverity = "error"
	SeverityCritical AuditSeverity = "critical"
)

// AuditCategory represents the category of an audit event
type AuditCategory string

const (
	CategoryAuthentication AuditCategory = "authentication"
	CategoryAuthorization  AuditCategory = "authorization"
	CategoryDataAccess     AuditCategory = "data_access"
	CategorySystemAccess   AuditCategory = "system_access"
	CategoryConfiguration  AuditCategory = "configuration"
	CategoryCompliance     AuditCategory = "compliance"
	CategorySecurity       AuditCategory = "security"
)

// AuditLogger handles security audit logging
type AuditLogger struct {
	logger     *logrus.Logger
	file       *os.File
	mu         sync.Mutex
	config     *AuditConfig
	buffer     []*AuditEvent
	bufferSize int
}

// AuditConfig contains configuration for audit logging
type AuditConfig struct {
	LogFile         string        `json:"log_file"`
	MaxFileSize     int64         `json:"max_file_size"` // bytes
	MaxFiles        int           `json:"max_files"`
	BufferSize      int           `json:"buffer_size"`
	FlushInterval   time.Duration `json:"flush_interval"`
	EnableSyslog    bool          `json:"enable_syslog"`
	SyslogAddress   string        `json:"syslog_address"`
	EnableRemoteLog bool          `json:"enable_remote_log"`
	RemoteLogURL    string        `json:"remote_log_url"`
	EncryptLogs     bool          `json:"encrypt_logs"`
	RetentionDays   int           `json:"retention_days"`
}

// NewAuditLogger creates a new audit logger
func NewAuditLogger() *AuditLogger {
	config := &AuditConfig{
		LogFile:       "/var/log/chatbot/audit.log",
		MaxFileSize:   100 * 1024 * 1024, // 100MB
		MaxFiles:      10,
		BufferSize:    100,
		FlushInterval: 5 * time.Second,
		RetentionDays: 90,
	}

	logger := logrus.New()
	logger.SetFormatter(&logrus.JSONFormatter{
		TimestampFormat: time.RFC3339Nano,
	})

	auditLogger := &AuditLogger{
		logger:     logger,
		config:     config,
		buffer:     make([]*AuditEvent, 0, config.BufferSize),
		bufferSize: config.BufferSize,
	}

	// Initialize log file
	if err := auditLogger.initLogFile(); err != nil {
		logger.Errorf("Failed to initialize audit log file: %v", err)
	}

	// Start background flush routine
	go auditLogger.flushRoutine()

	return auditLogger
}

// initLogFile initializes the audit log file
func (al *AuditLogger) initLogFile() error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll("/var/log/chatbot", 0755); err != nil {
		return fmt.Errorf("failed to create log directory: %v", err)
	}

	file, err := os.OpenFile(al.config.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open audit log file: %v", err)
	}

	al.file = file
	al.logger.SetOutput(file)

	return nil
}

// LogAccessRequest logs an access request
func (al *AuditLogger) LogAccessRequest(request *AccessRequest) {
	event := &AuditEvent{
		ID:        generateEventID(),
		Type:      "ACCESS_REQUEST",
		UserID:    request.UserID,
		Resource:  request.Resource,
		Action:    request.Action,
		Result:    "PENDING",
		IPAddress: request.Context.IPAddress,
		UserAgent: request.Context.UserAgent,
		SessionID: request.Context.SessionID,
		DeviceID:  request.Context.DeviceID,
		Location:  request.Context.Location,
		Metadata: map[string]interface{}{
			"request_id": request.Context.RequestID,
			"headers":    request.Context.Headers,
			"claims":     request.Claims,
		},
		Timestamp: request.Timestamp,
		Severity:  SeverityInfo,
		Category:  CategorySystemAccess,
	}

	al.logEvent(event)
}

// LogAccessDecision logs an access control decision
func (al *AuditLogger) LogAccessDecision(request *AccessRequest, decision *AccessDecision) {
	result := "DENIED"
	severity := SeverityWarning
	if decision.Allowed {
		result = "ALLOWED"
		severity = SeverityInfo
	}

	event := &AuditEvent{
		ID:        generateEventID(),
		Type:      "ACCESS_DECISION",
		UserID:    request.UserID,
		Resource:  request.Resource,
		Action:    request.Action,
		Result:    result,
		IPAddress: request.Context.IPAddress,
		UserAgent: request.Context.UserAgent,
		SessionID: request.Context.SessionID,
		DeviceID:  request.Context.DeviceID,
		Location:  request.Context.Location,
		Metadata: map[string]interface{}{
			"request_id":    request.Context.RequestID,
			"policy_id":     decision.PolicyID,
			"rule_id":       decision.RuleID,
			"reason":        decision.Reason,
			"confidence":    decision.Confidence,
			"action":        decision.Action,
			"session_token": decision.SessionToken != "",
		},
		Timestamp: decision.Timestamp,
		Severity:  severity,
		Category:  CategoryAuthorization,
	}

	al.logEvent(event)
}

// LogAuthenticationEvent logs authentication events
func (al *AuditLogger) LogAuthenticationEvent(userID, eventType, result string, metadata map[string]interface{}) {
	severity := SeverityInfo
	if result == "FAILED" {
		severity = SeverityWarning
	}

	event := &AuditEvent{
		ID:        generateEventID(),
		Type:      eventType,
		UserID:    userID,
		Result:    result,
		Metadata:  metadata,
		Timestamp: time.Now(),
		Severity:  severity,
		Category:  CategoryAuthentication,
	}

	if ipAddress, ok := metadata["ip_address"].(string); ok {
		event.IPAddress = ipAddress
	}
	if userAgent, ok := metadata["user_agent"].(string); ok {
		event.UserAgent = userAgent
	}
	if sessionID, ok := metadata["session_id"].(string); ok {
		event.SessionID = sessionID
	}

	al.logEvent(event)
}

// LogDataAccessEvent logs data access events
func (al *AuditLogger) LogDataAccessEvent(userID, resource, action, result string, metadata map[string]interface{}) {
	severity := SeverityInfo
	if result == "DENIED" {
		severity = SeverityWarning
	}

	event := &AuditEvent{
		ID:        generateEventID(),
		Type:      "DATA_ACCESS",
		UserID:    userID,
		Resource:  resource,
		Action:    action,
		Result:    result,
		Metadata:  metadata,
		Timestamp: time.Now(),
		Severity:  severity,
		Category:  CategoryDataAccess,
	}

	al.logEvent(event)
}

// LogSecurityEvent logs general security events
func (al *AuditLogger) LogSecurityEvent(eventType string, metadata map[string]interface{}) {
	event := &AuditEvent{
		ID:        generateEventID(),
		Type:      eventType,
		Result:    "INFO",
		Metadata:  metadata,
		Timestamp: time.Now(),
		Severity:  SeverityInfo,
		Category:  CategorySecurity,
	}

	if userID, ok := metadata["user_id"].(string); ok {
		event.UserID = userID
	}
	if resource, ok := metadata["resource"].(string); ok {
		event.Resource = resource
	}
	if action, ok := metadata["action"].(string); ok {
		event.Action = action
	}

	al.logEvent(event)
}

// LogComplianceEvent logs compliance-related events
func (al *AuditLogger) LogComplianceEvent(eventType, result string, metadata map[string]interface{}) {
	severity := SeverityInfo
	if result == "VIOLATION" {
		severity = SeverityError
	}

	event := &AuditEvent{
		ID:        generateEventID(),
		Type:      eventType,
		Result:    result,
		Metadata:  metadata,
		Timestamp: time.Now(),
		Severity:  severity,
		Category:  CategoryCompliance,
	}

	al.logEvent(event)
}

// LogConfigurationChange logs configuration changes
func (al *AuditLogger) LogConfigurationChange(userID, component, action string, oldValue, newValue interface{}) {
	event := &AuditEvent{
		ID:       generateEventID(),
		Type:     "CONFIGURATION_CHANGE",
		UserID:   userID,
		Resource: component,
		Action:   action,
		Result:   "SUCCESS",
		Metadata: map[string]interface{}{
			"old_value": oldValue,
			"new_value": newValue,
		},
		Timestamp: time.Now(),
		Severity:  SeverityInfo,
		Category:  CategoryConfiguration,
	}

	al.logEvent(event)
}

// logEvent adds an event to the buffer
func (al *AuditLogger) logEvent(event *AuditEvent) {
	al.mu.Lock()
	defer al.mu.Unlock()

	al.buffer = append(al.buffer, event)

	// Flush if buffer is full
	if len(al.buffer) >= al.bufferSize {
		al.flushBuffer()
	}
}

// flushBuffer writes buffered events to the log
func (al *AuditLogger) flushBuffer() {
	if len(al.buffer) == 0 {
		return
	}

	for _, event := range al.buffer {
		eventJSON, err := json.Marshal(event)
		if err != nil {
			al.logger.Errorf("Failed to marshal audit event: %v", err)
			continue
		}

		al.logger.Info(string(eventJSON))

		// Send to remote log if configured
		if al.config.EnableRemoteLog {
			go al.sendToRemoteLog(event)
		}
	}

	// Clear buffer
	al.buffer = al.buffer[:0]
}

// flushRoutine periodically flushes the buffer
func (al *AuditLogger) flushRoutine() {
	ticker := time.NewTicker(al.config.FlushInterval)
	defer ticker.Stop()

	for range ticker.C {
		al.mu.Lock()
		al.flushBuffer()
		al.mu.Unlock()
	}
}

// sendToRemoteLog sends audit events to a remote logging service
func (al *AuditLogger) sendToRemoteLog(event *AuditEvent) {
	// Implementation would depend on the remote logging service
	// This is a placeholder for integration with services like:
	// - Elasticsearch
	// - Splunk
	// - AWS CloudWatch
	// - Azure Monitor
	// - Google Cloud Logging
}

// GetAuditEvents retrieves audit events based on filters
func (al *AuditLogger) GetAuditEvents(filters map[string]interface{}) ([]*AuditEvent, error) {
	// This would typically query a database or search index
	// For now, return empty slice
	return []*AuditEvent{}, nil
}

// GenerateComplianceReport generates a compliance report
func (al *AuditLogger) GenerateComplianceReport(startTime, endTime time.Time, standards []string) (*ComplianceReport, error) {
	report := &ComplianceReport{
		ID:          generateEventID(),
		StartTime:   startTime,
		EndTime:     endTime,
		Standards:   standards,
		GeneratedAt: time.Now(),
		TotalEvents: 0,
		Violations:  0,
		Findings:    make([]*ComplianceFinding, 0),
	}

	// This would analyze audit events for compliance violations
	// Implementation depends on specific compliance requirements

	return report, nil
}

// ComplianceReport represents a compliance audit report
type ComplianceReport struct {
	ID          string               `json:"id"`
	StartTime   time.Time            `json:"start_time"`
	EndTime     time.Time            `json:"end_time"`
	Standards   []string             `json:"standards"`
	GeneratedAt time.Time            `json:"generated_at"`
	TotalEvents int                  `json:"total_events"`
	Violations  int                  `json:"violations"`
	Findings    []*ComplianceFinding `json:"findings"`
}

// ComplianceFinding represents a compliance finding
type ComplianceFinding struct {
	ID          string                 `json:"id"`
	Standard    string                 `json:"standard"`
	Requirement string                 `json:"requirement"`
	Severity    string                 `json:"severity"`
	Description string                 `json:"description"`
	Evidence    map[string]interface{} `json:"evidence"`
	Status      string                 `json:"status"`
	CreatedAt   time.Time              `json:"created_at"`
}

// Close closes the audit logger and flushes remaining events
func (al *AuditLogger) Close() error {
	al.mu.Lock()
	defer al.mu.Unlock()

	// Flush remaining events
	al.flushBuffer()

	// Close log file
	if al.file != nil {
		return al.file.Close()
	}

	return nil
}

// generateEventID generates a unique event ID
func generateEventID() string {
	return fmt.Sprintf("audit_%d_%d", time.Now().UnixNano(), time.Now().Nanosecond())
}
