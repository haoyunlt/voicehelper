package validation

import (
	"fmt"
	"regexp"
	"strings"
	"time"
	"unicode/utf8"
)

// LegacyValidationError 旧版验证错误（保持兼容性）
type LegacyValidationError struct {
	Field   string `json:"field"`
	Message string `json:"message"`
	Code    string `json:"code"`
}

// Error 实现error接口
func (e LegacyValidationError) Error() string {
	return fmt.Sprintf("validation error on field '%s': %s", e.Field, e.Message)
}

// ValidationErrors 验证错误集合
type ValidationErrors []LegacyValidationError

// Error 实现error接口
func (e ValidationErrors) Error() string {
	if len(e) == 0 {
		return "no validation errors"
	}

	var messages []string
	for _, err := range e {
		messages = append(messages, err.Error())
	}
	return strings.Join(messages, "; ")
}

// HasErrors 检查是否有错误
func (e ValidationErrors) HasErrors() bool {
	return len(e) > 0
}

// Validator 验证器
type Validator struct {
	errors ValidationErrors
}

// NewValidator 创建验证器
func NewValidator() *Validator {
	return &Validator{
		errors: make(ValidationErrors, 0),
	}
}

// AddError 添加错误
func (v *Validator) AddError(field, message, code string) {
	v.errors = append(v.errors, LegacyValidationError{
		Field:   field,
		Message: message,
		Code:    code,
	})
}

// HasErrors 检查是否有错误
func (v *Validator) HasErrors() bool {
	return len(v.errors) > 0
}

// Errors 获取错误
func (v *Validator) Errors() ValidationErrors {
	return v.errors
}

// Required 必填验证
func (v *Validator) Required(field, value string) *Validator {
	if strings.TrimSpace(value) == "" {
		v.AddError(field, "This field is required", "required")
	}
	return v
}

// MinLength 最小长度验证
func (v *Validator) MinLength(field, value string, min int) *Validator {
	if utf8.RuneCountInString(value) < min {
		v.AddError(field, fmt.Sprintf("Must be at least %d characters long", min), "min_length")
	}
	return v
}

// MaxLength 最大长度验证
func (v *Validator) MaxLength(field, value string, max int) *Validator {
	if utf8.RuneCountInString(value) > max {
		v.AddError(field, fmt.Sprintf("Must be no more than %d characters long", max), "max_length")
	}
	return v
}

// Length 长度范围验证
func (v *Validator) Length(field, value string, min, max int) *Validator {
	length := utf8.RuneCountInString(value)
	if length < min || length > max {
		v.AddError(field, fmt.Sprintf("Must be between %d and %d characters long", min, max), "length_range")
	}
	return v
}

// Email 邮箱验证
func (v *Validator) Email(field, value string) *Validator {
	if value == "" {
		return v
	}

	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	if !emailRegex.MatchString(value) {
		v.AddError(field, "Must be a valid email address", "invalid_email")
	}
	return v
}

// Phone 手机号验证
func (v *Validator) Phone(field, value string) *Validator {
	if value == "" {
		return v
	}

	// 支持中国手机号格式
	phoneRegex := regexp.MustCompile(`^1[3-9]\d{9}$`)
	if !phoneRegex.MatchString(value) {
		v.AddError(field, "Must be a valid phone number", "invalid_phone")
	}
	return v
}

// URL URL验证
func (v *Validator) URL(field, value string) *Validator {
	if value == "" {
		return v
	}

	urlRegex := regexp.MustCompile(`^https?://[^\s/$.?#].[^\s]*$`)
	if !urlRegex.MatchString(value) {
		v.AddError(field, "Must be a valid URL", "invalid_url")
	}
	return v
}

// Pattern 正则表达式验证
func (v *Validator) Pattern(field, value, pattern, message string) *Validator {
	if value == "" {
		return v
	}

	regex, err := regexp.Compile(pattern)
	if err != nil {
		v.AddError(field, "Invalid pattern", "invalid_pattern")
		return v
	}

	if !regex.MatchString(value) {
		v.AddError(field, message, "pattern_mismatch")
	}
	return v
}

// In 枚举值验证
func (v *Validator) In(field, value string, allowed []string) *Validator {
	if value == "" {
		return v
	}

	for _, allowed := range allowed {
		if value == allowed {
			return v
		}
	}

	v.AddError(field, fmt.Sprintf("Must be one of: %s", strings.Join(allowed, ", ")), "invalid_choice")
	return v
}

// NotIn 排除值验证
func (v *Validator) NotIn(field, value string, forbidden []string) *Validator {
	if value == "" {
		return v
	}

	for _, forbidden := range forbidden {
		if value == forbidden {
			v.AddError(field, fmt.Sprintf("Cannot be: %s", value), "forbidden_value")
			return v
		}
	}

	return v
}

// Min 最小值验证
func (v *Validator) Min(field string, value, min int) *Validator {
	if value < min {
		v.AddError(field, fmt.Sprintf("Must be at least %d", min), "min_value")
	}
	return v
}

// Max 最大值验证
func (v *Validator) Max(field string, value, max int) *Validator {
	if value > max {
		v.AddError(field, fmt.Sprintf("Must be no more than %d", max), "max_value")
	}
	return v
}

// Range 范围验证
func (v *Validator) Range(field string, value, min, max int) *Validator {
	if value < min || value > max {
		v.AddError(field, fmt.Sprintf("Must be between %d and %d", min, max), "value_range")
	}
	return v
}

// Future 未来时间验证
func (v *Validator) Future(field string, value time.Time) *Validator {
	if !value.IsZero() && value.Before(time.Now()) {
		v.AddError(field, "Must be a future date", "future_date")
	}
	return v
}

// Past 过去时间验证
func (v *Validator) Past(field string, value time.Time) *Validator {
	if !value.IsZero() && value.After(time.Now()) {
		v.AddError(field, "Must be a past date", "past_date")
	}
	return v
}

// DateRange 日期范围验证
func (v *Validator) DateRange(field string, value, min, max time.Time) *Validator {
	if !value.IsZero() && (value.Before(min) || value.After(max)) {
		v.AddError(field, fmt.Sprintf("Must be between %s and %s",
			min.Format("2006-01-02"), max.Format("2006-01-02")), "date_range")
	}
	return v
}

// JSON JSON格式验证
func (v *Validator) JSON(field, value string) *Validator {
	if value == "" {
		return v
	}

	// 简单的JSON格式检查
	value = strings.TrimSpace(value)
	if !((strings.HasPrefix(value, "{") && strings.HasSuffix(value, "}")) ||
		(strings.HasPrefix(value, "[") && strings.HasSuffix(value, "]"))) {
		v.AddError(field, "Must be valid JSON", "invalid_json")
	}
	return v
}

// Custom 自定义验证
func (v *Validator) Custom(field string, fn func() (bool, string)) *Validator {
	if valid, message := fn(); !valid {
		v.AddError(field, message, "custom_validation")
	}
	return v
}

// 预定义的验证规则

// ValidateTenantID 验证租户ID
func ValidateTenantID(tenantID string) ValidationErrors {
	v := NewValidator()
	v.Required("tenant_id", tenantID).
		MinLength("tenant_id", tenantID, 3).
		MaxLength("tenant_id", tenantID, 50).
		Pattern("tenant_id", tenantID, `^[a-zA-Z0-9_-]+$`, "Tenant ID can only contain letters, numbers, underscores and hyphens")
	return v.Errors()
}

// ValidateUserID 验证用户ID
func ValidateUserID(userID string) ValidationErrors {
	v := NewValidator()
	v.Required("user_id", userID).
		MinLength("user_id", userID, 3).
		MaxLength("user_id", userID, 50).
		Pattern("user_id", userID, `^[a-zA-Z0-9_-]+$`, "User ID can only contain letters, numbers, underscores and hyphens")
	return v.Errors()
}

// ValidateEmail 验证邮箱
func ValidateEmail(email string) ValidationErrors {
	v := NewValidator()
	if email != "" {
		v.Email("email", email).MaxLength("email", email, 100)
	}
	return v.Errors()
}

// ValidatePhone 验证手机号
func ValidatePhone(phone string) ValidationErrors {
	v := NewValidator()
	if phone != "" {
		v.Phone("phone", phone)
	}
	return v.Errors()
}

// ValidateConversationTitle 验证会话标题
func ValidateConversationTitle(title string) ValidationErrors {
	v := NewValidator()
	if title != "" {
		v.MinLength("title", title, 1).MaxLength("title", title, 200)
	}
	return v.Errors()
}

// ValidateMessageContent 验证消息内容
func ValidateMessageContent(content string) ValidationErrors {
	v := NewValidator()
	v.Required("content", content).
		MinLength("content", content, 1).
		MaxLength("content", content, 10000)
	return v.Errors()
}

// ValidateMessageRole 验证消息角色
func ValidateMessageRole(role string) ValidationErrors {
	v := NewValidator()
	v.Required("role", role).
		In("role", role, []string{"user", "assistant", "system"})
	return v.Errors()
}

// ValidateDocumentTitle 验证文档标题
func ValidateDocumentTitle(title string) ValidationErrors {
	v := NewValidator()
	v.Required("title", title).
		MinLength("title", title, 1).
		MaxLength("title", title, 200)
	return v.Errors()
}

// ValidateDocumentContent 验证文档内容
func ValidateDocumentContent(content string) ValidationErrors {
	v := NewValidator()
	v.Required("content", content).
		MinLength("content", content, 1).
		MaxLength("content", content, 1000000) // 1MB
	return v.Errors()
}

// ValidateStatus 验证状态
func ValidateStatus(status string, allowedStatuses []string) ValidationErrors {
	v := NewValidator()
	v.Required("status", status).
		In("status", status, allowedStatuses)
	return v.Errors()
}

// ValidatePlan 验证计划类型
func ValidatePlan(plan string) ValidationErrors {
	v := NewValidator()
	v.Required("plan", plan).
		In("plan", plan, []string{"free", "basic", "premium", "enterprise"})
	return v.Errors()
}

// ValidateVoiceSessionConfig 验证语音会话配置
func ValidateVoiceSessionConfig(config map[string]interface{}) ValidationErrors {
	v := NewValidator()

	// 验证采样率
	if sampleRate, ok := config["sample_rate"]; ok {
		if rate, ok := sampleRate.(float64); ok {
			v.In("sample_rate", fmt.Sprintf("%.0f", rate), []string{"8000", "16000", "22050", "44100", "48000"})
		}
	}

	// 验证声道数
	if channels, ok := config["channels"]; ok {
		if ch, ok := channels.(float64); ok {
			v.Range("channels", int(ch), 1, 2)
		}
	}

	// 验证语言
	if language, ok := config["language"]; ok {
		if lang, ok := language.(string); ok {
			v.In("language", lang, []string{"zh-CN", "en-US", "ja-JP", "ko-KR"})
		}
	}

	return v.Errors()
}
