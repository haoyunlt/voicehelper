package utils

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Validator 验证器
type Validator struct {
	errors []string
}

// NewValidator 创建验证器
func NewValidator() *Validator {
	return &Validator{
		errors: make([]string, 0),
	}
}

// Required 验证必需字段
func (v *Validator) Required(value interface{}, fieldName string) *Validator {
	if IsEmpty(value) {
		v.errors = append(v.errors, fmt.Sprintf("%s is required", fieldName))
	}
	return v
}

// NotNil 验证非空
func (v *Validator) NotNil(value interface{}, fieldName string) *Validator {
	if IsNil(value) {
		v.errors = append(v.errors, fmt.Sprintf("%s cannot be nil", fieldName))
	}
	return v
}

// MinLength 验证最小长度
func (v *Validator) MinLength(value string, minLen int, fieldName string) *Validator {
	if len(value) < minLen {
		v.errors = append(v.errors, fmt.Sprintf("%s must be at least %d characters", fieldName, minLen))
	}
	return v
}

// MaxLength 验证最大长度
func (v *Validator) MaxLength(value string, maxLen int, fieldName string) *Validator {
	if len(value) > maxLen {
		v.errors = append(v.errors, fmt.Sprintf("%s must be at most %d characters", fieldName, maxLen))
	}
	return v
}

// Range 验证数值范围
func (v *Validator) Range(value, min, max int, fieldName string) *Validator {
	if value < min || value > max {
		v.errors = append(v.errors, fmt.Sprintf("%s must be between %d and %d", fieldName, min, max))
	}
	return v
}

// RangeFloat 验证浮点数范围
func (v *Validator) RangeFloat(value, min, max float64, fieldName string) *Validator {
	if value < min || value > max {
		v.errors = append(v.errors, fmt.Sprintf("%s must be between %.2f and %.2f", fieldName, min, max))
	}
	return v
}

// Email 验证邮箱格式
func (v *Validator) Email(value, fieldName string) *Validator {
	if value == "" {
		return v
	}

	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	if !emailRegex.MatchString(value) {
		v.errors = append(v.errors, fmt.Sprintf("%s must be a valid email address", fieldName))
	}
	return v
}

// URL 验证URL格式
func (v *Validator) URL(value, fieldName string) *Validator {
	if value == "" {
		return v
	}

	urlRegex := regexp.MustCompile(`^https?://[^\s/$.?#].[^\s]*$`)
	if !urlRegex.MatchString(value) {
		v.errors = append(v.errors, fmt.Sprintf("%s must be a valid URL", fieldName))
	}
	return v
}

// Phone 验证手机号格式
func (v *Validator) Phone(value, fieldName string) *Validator {
	if value == "" {
		return v
	}

	phoneRegex := regexp.MustCompile(`^1[3-9]\d{9}$`)
	if !phoneRegex.MatchString(value) {
		v.errors = append(v.errors, fmt.Sprintf("%s must be a valid phone number", fieldName))
	}
	return v
}

// Pattern 验证正则表达式
func (v *Validator) Pattern(value, pattern, fieldName string) *Validator {
	if value == "" {
		return v
	}

	regex, err := regexp.Compile(pattern)
	if err != nil {
		v.errors = append(v.errors, fmt.Sprintf("invalid pattern for %s", fieldName))
		return v
	}

	if !regex.MatchString(value) {
		v.errors = append(v.errors, fmt.Sprintf("%s format is invalid", fieldName))
	}
	return v
}

// In 验证值是否在指定列表中
func (v *Validator) In(value interface{}, validValues []interface{}, fieldName string) *Validator {
	for _, valid := range validValues {
		if reflect.DeepEqual(value, valid) {
			return v
		}
	}

	v.errors = append(v.errors, fmt.Sprintf("%s must be one of the valid values", fieldName))
	return v
}

// NotIn 验证值不在指定列表中
func (v *Validator) NotIn(value interface{}, invalidValues []interface{}, fieldName string) *Validator {
	for _, invalid := range invalidValues {
		if reflect.DeepEqual(value, invalid) {
			v.errors = append(v.errors, fmt.Sprintf("%s cannot be one of the forbidden values", fieldName))
			return v
		}
	}
	return v
}

// Future 验证时间是否在未来
func (v *Validator) Future(value time.Time, fieldName string) *Validator {
	if !value.IsZero() && value.Before(time.Now()) {
		v.errors = append(v.errors, fmt.Sprintf("%s must be in the future", fieldName))
	}
	return v
}

// Past 验证时间是否在过去
func (v *Validator) Past(value time.Time, fieldName string) *Validator {
	if !value.IsZero() && value.After(time.Now()) {
		v.errors = append(v.errors, fmt.Sprintf("%s must be in the past", fieldName))
	}
	return v
}

// Custom 自定义验证
func (v *Validator) Custom(condition bool, message string) *Validator {
	if !condition {
		v.errors = append(v.errors, message)
	}
	return v
}

// IsValid 检查是否有效
func (v *Validator) IsValid() bool {
	return len(v.errors) == 0
}

// Errors 获取所有错误
func (v *Validator) Errors() []string {
	return v.errors
}

// Error 获取错误信息
func (v *Validator) Error() error {
	if len(v.errors) == 0 {
		return nil
	}
	return fmt.Errorf("validation failed: %s", strings.Join(v.errors, "; "))
}

// Clear 清除错误
func (v *Validator) Clear() *Validator {
	v.errors = make([]string, 0)
	return v
}

// ValidateStruct 验证结构体
func ValidateStruct(s interface{}) error {
	v := reflect.ValueOf(s)
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}

	if v.Kind() != reflect.Struct {
		return fmt.Errorf("expected struct, got %s", v.Kind())
	}

	validator := NewValidator()
	t := v.Type()

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		fieldType := t.Field(i)

		// 检查required标签
		if tag := fieldType.Tag.Get("validate"); tag != "" {
			rules := strings.Split(tag, ",")
			for _, rule := range rules {
				rule = strings.TrimSpace(rule)
				if err := applyValidationRule(validator, field.Interface(), rule, fieldType.Name); err != nil {
					return err
				}
			}
		}
	}

	return validator.Error()
}

// applyValidationRule 应用验证规则
func applyValidationRule(validator *Validator, value interface{}, rule, fieldName string) error {
	parts := strings.Split(rule, "=")
	ruleName := parts[0]

	switch ruleName {
	case "required":
		validator.Required(value, fieldName)
	case "min":
		if len(parts) > 1 {
			if min, err := strconv.Atoi(parts[1]); err == nil {
				if str, ok := value.(string); ok {
					validator.MinLength(str, min, fieldName)
				} else if num, ok := value.(int); ok {
					validator.Range(num, min, int(^uint(0)>>1), fieldName)
				}
			}
		}
	case "max":
		if len(parts) > 1 {
			if max, err := strconv.Atoi(parts[1]); err == nil {
				if str, ok := value.(string); ok {
					validator.MaxLength(str, max, fieldName)
				} else if num, ok := value.(int); ok {
					validator.Range(num, 0, max, fieldName)
				}
			}
		}
	case "email":
		if str, ok := value.(string); ok {
			validator.Email(str, fieldName)
		}
	case "url":
		if str, ok := value.(string); ok {
			validator.URL(str, fieldName)
		}
	case "phone":
		if str, ok := value.(string); ok {
			validator.Phone(str, fieldName)
		}
	}

	return nil
}

// ValidateMap 验证map数据
func ValidateMap(data map[string]interface{}, rules map[string]string) error {
	validator := NewValidator()

	for field, rule := range rules {
		value, exists := data[field]
		if !exists {
			value = nil
		}

		ruleList := strings.Split(rule, ",")
		for _, r := range ruleList {
			r = strings.TrimSpace(r)
			if err := applyValidationRule(validator, value, r, field); err != nil {
				return err
			}
		}
	}

	return validator.Error()
}

// SanitizeString 清理字符串
func SanitizeString(s string) string {
	// 移除前后空格
	s = strings.TrimSpace(s)

	// 移除多余的空格
	spaceRegex := regexp.MustCompile(`\s+`)
	s = spaceRegex.ReplaceAllString(s, " ")

	return s
}

// SanitizeHTML 清理HTML标签
func SanitizeHTML(s string) string {
	htmlRegex := regexp.MustCompile(`<[^>]*>`)
	return htmlRegex.ReplaceAllString(s, "")
}

// NormalizeEmail 标准化邮箱地址
func NormalizeEmail(email string) string {
	email = strings.ToLower(strings.TrimSpace(email))
	return email
}

// NormalizePhone 标准化手机号
func NormalizePhone(phone string) string {
	// 移除所有非数字字符
	phoneRegex := regexp.MustCompile(`[^\d]`)
	phone = phoneRegex.ReplaceAllString(phone, "")

	// 如果是11位且以1开头，则是有效的中国手机号
	if len(phone) == 11 && phone[0] == '1' {
		return phone
	}

	return ""
}
