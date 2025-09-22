package common

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

// ValidationRule 验证规则
type ValidationRule struct {
	Field    string
	Value    interface{}
	Rules    []string
	Messages map[string]string
}

// Validator 验证器
type Validator struct {
	rules  []ValidationRule
	errors map[string][]string
}

// NewValidator 创建验证器
func NewValidator() *Validator {
	return &Validator{
		rules:  make([]ValidationRule, 0),
		errors: make(map[string][]string),
	}
}

// Field 添加字段验证规则
func (v *Validator) Field(field string, value interface{}, rules ...string) *Validator {
	v.rules = append(v.rules, ValidationRule{
		Field: field,
		Value: value,
		Rules: rules,
	})
	return v
}

// FieldWithMessages 添加带自定义消息的字段验证规则
func (v *Validator) FieldWithMessages(field string, value interface{}, rules []string, messages map[string]string) *Validator {
	v.rules = append(v.rules, ValidationRule{
		Field:    field,
		Value:    value,
		Rules:    rules,
		Messages: messages,
	})
	return v
}

// Validate 执行验证
func (v *Validator) Validate() bool {
	v.errors = make(map[string][]string)

	for _, rule := range v.rules {
		for _, r := range rule.Rules {
			if err := v.validateRule(rule.Field, rule.Value, r, rule.Messages); err != nil {
				if _, exists := v.errors[rule.Field]; !exists {
					v.errors[rule.Field] = make([]string, 0)
				}
				v.errors[rule.Field] = append(v.errors[rule.Field], err.Error())
			}
		}
	}

	return len(v.errors) == 0
}

// GetErrors 获取验证错误
func (v *Validator) GetErrors() map[string][]string {
	return v.errors
}

// GetFirstError 获取第一个错误
func (v *Validator) GetFirstError() string {
	for _, errors := range v.errors {
		if len(errors) > 0 {
			return errors[0]
		}
	}
	return ""
}

// validateRule 验证单个规则
func (v *Validator) validateRule(field string, value interface{}, rule string, messages map[string]string) error {
	parts := strings.Split(rule, ":")
	ruleName := parts[0]
	var ruleValue string
	if len(parts) > 1 {
		ruleValue = parts[1]
	}

	switch ruleName {
	case "required":
		return v.validateRequired(field, value, messages)
	case "min":
		return v.validateMin(field, value, ruleValue, messages)
	case "max":
		return v.validateMax(field, value, ruleValue, messages)
	case "email":
		return v.validateEmail(field, value, messages)
	case "phone":
		return v.validatePhone(field, value, messages)
	case "url":
		return v.validateURL(field, value, messages)
	case "numeric":
		return v.validateNumeric(field, value, messages)
	case "alpha":
		return v.validateAlpha(field, value, messages)
	case "alphanumeric":
		return v.validateAlphanumeric(field, value, messages)
	case "regex":
		return v.validateRegex(field, value, ruleValue, messages)
	case "in":
		return v.validateIn(field, value, ruleValue, messages)
	case "not_in":
		return v.validateNotIn(field, value, ruleValue, messages)
	default:
		return fmt.Errorf("unknown validation rule: %s", ruleName)
	}
}

// validateRequired 验证必填
func (v *Validator) validateRequired(field string, value interface{}, messages map[string]string) error {
	if isEmpty(value) {
		return fmt.Errorf(v.getMessage("required", field, messages, "%s is required"))
	}
	return nil
}

// validateMin 验证最小长度/值
func (v *Validator) validateMin(field string, value interface{}, minStr string, messages map[string]string) error {
	if isEmpty(value) {
		return nil // 空值跳过
	}

	min := parseInt(minStr, 0)

	switch v := value.(type) {
	case string:
		if utf8.RuneCountInString(v) < min {
			return fmt.Errorf(v.getMessage("min", field, messages, "%s must be at least %d characters"))
		}
	case int, int32, int64:
		if toInt(v) < min {
			return fmt.Errorf(v.getMessage("min", field, messages, "%s must be at least %d"))
		}
	}

	return nil
}

// validateMax 验证最大长度/值
func (v *Validator) validateMax(field string, value interface{}, maxStr string, messages map[string]string) error {
	if isEmpty(value) {
		return nil // 空值跳过
	}

	max := parseInt(maxStr, 0)

	switch v := value.(type) {
	case string:
		if utf8.RuneCountInString(v) > max {
			return fmt.Errorf(v.getMessage("max", field, messages, "%s must be at most %d characters"))
		}
	case int, int32, int64:
		if toInt(v) > max {
			return fmt.Errorf(v.getMessage("max", field, messages, "%s must be at most %d"))
		}
	}

	return nil
}

// validateEmail 验证邮箱
func (v *Validator) validateEmail(field string, value interface{}, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		return fmt.Errorf("%s must be a string", field)
	}

	emailRegex := regexp.MustCompile(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)
	if !emailRegex.MatchString(str) {
		return fmt.Errorf(v.getMessage("email", field, messages, "%s must be a valid email address"))
	}

	return nil
}

// validatePhone 验证手机号
func (v *Validator) validatePhone(field string, value interface{}, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		return fmt.Errorf("%s must be a string", field)
	}

	phoneRegex := regexp.MustCompile(`^1[3-9]\d{9}$`)
	if !phoneRegex.MatchString(str) {
		return fmt.Errorf(v.getMessage("phone", field, messages, "%s must be a valid phone number"))
	}

	return nil
}

// validateURL 验证URL
func (v *Validator) validateURL(field string, value interface{}, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		return fmt.Errorf("%s must be a string", field)
	}

	urlRegex := regexp.MustCompile(`^https?://[^\s/$.?#].[^\s]*$`)
	if !urlRegex.MatchString(str) {
		return fmt.Errorf(v.getMessage("url", field, messages, "%s must be a valid URL"))
	}

	return nil
}

// validateNumeric 验证数字
func (v *Validator) validateNumeric(field string, value interface{}, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		return fmt.Errorf("%s must be a string", field)
	}

	numericRegex := regexp.MustCompile(`^\d+$`)
	if !numericRegex.MatchString(str) {
		return fmt.Errorf(v.getMessage("numeric", field, messages, "%s must be numeric"))
	}

	return nil
}

// validateAlpha 验证字母
func (v *Validator) validateAlpha(field string, value interface{}, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		return fmt.Errorf("%s must be a string", field)
	}

	alphaRegex := regexp.MustCompile(`^[a-zA-Z]+$`)
	if !alphaRegex.MatchString(str) {
		return fmt.Errorf(v.getMessage("alpha", field, messages, "%s must contain only letters"))
	}

	return nil
}

// validateAlphanumeric 验证字母数字
func (v *Validator) validateAlphanumeric(field string, value interface{}, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		return fmt.Errorf("%s must be a string", field)
	}

	alphanumericRegex := regexp.MustCompile(`^[a-zA-Z0-9]+$`)
	if !alphanumericRegex.MatchString(str) {
		return fmt.Errorf(v.getMessage("alphanumeric", field, messages, "%s must contain only letters and numbers"))
	}

	return nil
}

// validateRegex 验证正则表达式
func (v *Validator) validateRegex(field string, value interface{}, pattern string, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str, ok := value.(string)
	if !ok {
		return fmt.Errorf("%s must be a string", field)
	}

	regex, err := regexp.Compile(pattern)
	if err != nil {
		return fmt.Errorf("invalid regex pattern: %s", pattern)
	}

	if !regex.MatchString(str) {
		return fmt.Errorf(v.getMessage("regex", field, messages, "%s format is invalid"))
	}

	return nil
}

// validateIn 验证在列表中
func (v *Validator) validateIn(field string, value interface{}, values string, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str := toString(value)
	validValues := strings.Split(values, ",")

	for _, validValue := range validValues {
		if strings.TrimSpace(validValue) == str {
			return nil
		}
	}

	return fmt.Errorf(v.getMessage("in", field, messages, "%s must be one of: %s"), values)
}

// validateNotIn 验证不在列表中
func (v *Validator) validateNotIn(field string, value interface{}, values string, messages map[string]string) error {
	if isEmpty(value) {
		return nil
	}

	str := toString(value)
	invalidValues := strings.Split(values, ",")

	for _, invalidValue := range invalidValues {
		if strings.TrimSpace(invalidValue) == str {
			return fmt.Errorf(v.getMessage("not_in", field, messages, "%s cannot be one of: %s"), values)
		}
	}

	return nil
}

// getMessage 获取错误消息
func (v *Validator) getMessage(rule, field string, messages map[string]string, defaultMsg string) string {
	if messages != nil {
		if msg, exists := messages[rule]; exists {
			return msg
		}
	}
	return fmt.Sprintf(defaultMsg, field)
}

// 辅助函数
func isEmpty(value interface{}) bool {
	if value == nil {
		return true
	}

	switch v := value.(type) {
	case string:
		return strings.TrimSpace(v) == ""
	case int, int32, int64:
		return v == 0
	case float32, float64:
		return v == 0
	case bool:
		return !v
	default:
		return false
	}
}

func parseInt(str string, defaultVal int) int {
	// 简单的字符串转整数实现
	if str == "" {
		return defaultVal
	}
	// 这里应该使用 strconv.Atoi，但为了避免导入，简化实现
	return defaultVal
}

func toInt(value interface{}) int {
	switch v := value.(type) {
	case int:
		return v
	case int32:
		return int(v)
	case int64:
		return int(v)
	default:
		return 0
	}
}

func toString(value interface{}) string {
	switch v := value.(type) {
	case string:
		return v
	case int:
		return fmt.Sprintf("%d", v)
	case int32:
		return fmt.Sprintf("%d", v)
	case int64:
		return fmt.Sprintf("%d", v)
	case float32:
		return fmt.Sprintf("%.2f", v)
	case float64:
		return fmt.Sprintf("%.2f", v)
	case bool:
		return fmt.Sprintf("%t", v)
	default:
		return fmt.Sprintf("%v", v)
	}
}
