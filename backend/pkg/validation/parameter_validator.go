package validation

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/go-playground/validator/v10"
)

// ParameterValidator 参数验证器
type ParameterValidator struct {
	validator *validator.Validate
}

// ValidationError 验证错误
type ValidationError struct {
	Field   string `json:"field"`
	Tag     string `json:"tag"`
	Value   string `json:"value"`
	Message string `json:"message"`
}

// ValidationResponse 验证响应
type ValidationResponse struct {
	Success bool              `json:"success"`
	Message string            `json:"message"`
	Errors  []ValidationError `json:"errors,omitempty"`
}

// NewParameterValidator 创建新的参数验证器
func NewParameterValidator() *ParameterValidator {
	v := validator.New()

	// 注册自定义验证规则
	v.RegisterValidation("audio_format", validateAudioFormat)
	v.RegisterValidation("language_code", validateLanguageCode)
	v.RegisterValidation("voice_id", validateVoiceID)
	v.RegisterValidation("emotion", validateEmotion)
	v.RegisterValidation("search_type", validateSearchType)

	return &ParameterValidator{
		validator: v,
	}
}

// ValidateStruct 验证结构体
func (pv *ParameterValidator) ValidateStruct(s interface{}) *ValidationResponse {
	err := pv.validator.Struct(s)
	if err == nil {
		return &ValidationResponse{
			Success: true,
			Message: "Validation passed",
		}
	}

	var errors []ValidationError
	for _, err := range err.(validator.ValidationErrors) {
		ve := ValidationError{
			Field:   err.Field(),
			Tag:     err.Tag(),
			Value:   fmt.Sprintf("%v", err.Value()),
			Message: pv.getErrorMessage(err),
		}
		errors = append(errors, ve)
	}

	return &ValidationResponse{
		Success: false,
		Message: "Validation failed",
		Errors:  errors,
	}
}

// ValidateJSON 验证JSON请求体
func (pv *ParameterValidator) ValidateJSON(c *gin.Context, obj interface{}) *ValidationResponse {
	if err := c.ShouldBindJSON(obj); err != nil {
		return &ValidationResponse{
			Success: false,
			Message: "Invalid JSON format",
			Errors: []ValidationError{
				{
					Field:   "request_body",
					Tag:     "json",
					Message: fmt.Sprintf("JSON parsing error: %v", err),
				},
			},
		}
	}

	return pv.ValidateStruct(obj)
}

// ValidateQueryParams 验证查询参数
func (pv *ParameterValidator) ValidateQueryParams(c *gin.Context, obj interface{}) *ValidationResponse {
	if err := c.ShouldBindQuery(obj); err != nil {
		return &ValidationResponse{
			Success: false,
			Message: "Invalid query parameters",
			Errors: []ValidationError{
				{
					Field:   "query_params",
					Tag:     "query",
					Message: fmt.Sprintf("Query parameter error: %v", err),
				},
			},
		}
	}

	return pv.ValidateStruct(obj)
}

// ValidatePathParams 验证路径参数
func (pv *ParameterValidator) ValidatePathParams(c *gin.Context, obj interface{}) *ValidationResponse {
	if err := c.ShouldBindUri(obj); err != nil {
		return &ValidationResponse{
			Success: false,
			Message: "Invalid path parameters",
			Errors: []ValidationError{
				{
					Field:   "path_params",
					Tag:     "uri",
					Message: fmt.Sprintf("Path parameter error: %v", err),
				},
			},
		}
	}

	return pv.ValidateStruct(obj)
}

// ValidateMultipart 验证multipart表单
func (pv *ParameterValidator) ValidateMultipart(c *gin.Context, obj interface{}) *ValidationResponse {
	if err := c.ShouldBind(obj); err != nil {
		return &ValidationResponse{
			Success: false,
			Message: "Invalid multipart form data",
			Errors: []ValidationError{
				{
					Field:   "form_data",
					Tag:     "multipart",
					Message: fmt.Sprintf("Multipart form error: %v", err),
				},
			},
		}
	}

	return pv.ValidateStruct(obj)
}

// CheckRequiredFields 检查必需字段
func (pv *ParameterValidator) CheckRequiredFields(data map[string]interface{}, requiredFields []string) *ValidationResponse {
	var errors []ValidationError

	for _, field := range requiredFields {
		value, exists := data[field]
		if !exists {
			errors = append(errors, ValidationError{
				Field:   field,
				Tag:     "required",
				Message: fmt.Sprintf("Field '%s' is required", field),
			})
			continue
		}

		// 检查空值
		if pv.isEmpty(value) {
			errors = append(errors, ValidationError{
				Field:   field,
				Tag:     "required",
				Message: fmt.Sprintf("Field '%s' cannot be empty", field),
			})
		}
	}

	if len(errors) > 0 {
		return &ValidationResponse{
			Success: false,
			Message: "Required fields missing or empty",
			Errors:  errors,
		}
	}

	return &ValidationResponse{
		Success: true,
		Message: "All required fields present",
	}
}

// isEmpty 检查值是否为空
func (pv *ParameterValidator) isEmpty(value interface{}) bool {
	if value == nil {
		return true
	}

	v := reflect.ValueOf(value)
	switch v.Kind() {
	case reflect.String:
		return strings.TrimSpace(v.String()) == ""
	case reflect.Slice, reflect.Map, reflect.Array:
		return v.Len() == 0
	case reflect.Ptr, reflect.Interface:
		return v.IsNil()
	default:
		return false
	}
}

// getErrorMessage 获取错误消息
func (pv *ParameterValidator) getErrorMessage(fe validator.FieldError) string {
	switch fe.Tag() {
	case "required":
		return fmt.Sprintf("Field '%s' is required", fe.Field())
	case "min":
		return fmt.Sprintf("Field '%s' must be at least %s characters/items", fe.Field(), fe.Param())
	case "max":
		return fmt.Sprintf("Field '%s' must be at most %s characters/items", fe.Field(), fe.Param())
	case "email":
		return fmt.Sprintf("Field '%s' must be a valid email address", fe.Field())
	case "url":
		return fmt.Sprintf("Field '%s' must be a valid URL", fe.Field())
	case "oneof":
		return fmt.Sprintf("Field '%s' must be one of: %s", fe.Field(), fe.Param())
	case "audio_format":
		return fmt.Sprintf("Field '%s' must be a valid audio format (wav, mp3, webm, m4a)", fe.Field())
	case "language_code":
		return fmt.Sprintf("Field '%s' must be a valid language code (zh-CN, en-US, ja-JP, etc.)", fe.Field())
	case "voice_id":
		return fmt.Sprintf("Field '%s' must be a valid voice ID", fe.Field())
	case "emotion":
		return fmt.Sprintf("Field '%s' must be a valid emotion (neutral, happy, sad, angry, etc.)", fe.Field())
	case "search_type":
		return fmt.Sprintf("Field '%s' must be a valid search type (semantic, keyword, hybrid)", fe.Field())
	default:
		return fmt.Sprintf("Field '%s' failed validation for tag '%s'", fe.Field(), fe.Tag())
	}
}

// 自定义验证函数
func validateAudioFormat(fl validator.FieldLevel) bool {
	format := fl.Field().String()
	validFormats := []string{"wav", "mp3", "webm", "m4a", "flac", "ogg"}
	for _, valid := range validFormats {
		if format == valid {
			return true
		}
	}
	return false
}

func validateLanguageCode(fl validator.FieldLevel) bool {
	code := fl.Field().String()
	validCodes := []string{
		"zh-CN", "zh-TW", "en-US", "en-GB", "ja-JP", "ko-KR",
		"fr-FR", "de-DE", "es-ES", "it-IT", "pt-BR", "ru-RU",
	}
	for _, valid := range validCodes {
		if code == valid {
			return true
		}
	}
	return false
}

func validateVoiceID(fl validator.FieldLevel) bool {
	voiceID := fl.Field().String()
	// 简单的voice ID格式验证
	return len(voiceID) > 0 && len(voiceID) <= 100
}

func validateEmotion(fl validator.FieldLevel) bool {
	emotion := fl.Field().String()
	validEmotions := []string{
		"neutral", "happy", "sad", "angry", "surprised", "fear",
		"disgust", "calm", "excited", "frustrated", "confident",
	}
	for _, valid := range validEmotions {
		if emotion == valid {
			return true
		}
	}
	return false
}

func validateSearchType(fl validator.FieldLevel) bool {
	searchType := fl.Field().String()
	validTypes := []string{"semantic", "keyword", "hybrid", "fuzzy"}
	for _, valid := range validTypes {
		if searchType == valid {
			return true
		}
	}
	return false
}

// ValidationMiddleware 验证中间件
func ValidationMiddleware() gin.HandlerFunc {
	return gin.HandlerFunc(func(c *gin.Context) {
		// 在上下文中设置验证器
		c.Set("validator", NewParameterValidator())
		c.Next()
	})
}

// GetValidator 从上下文获取验证器
func GetValidator(c *gin.Context) *ParameterValidator {
	if validator, exists := c.Get("validator"); exists {
		return validator.(*ParameterValidator)
	}
	return NewParameterValidator()
}

// SendValidationError 发送验证错误响应
func SendValidationError(c *gin.Context, response *ValidationResponse) {
	c.JSON(400, gin.H{
		"success": false,
		"error": gin.H{
			"code":    "VALIDATION_ERROR",
			"message": response.Message,
			"details": response.Errors,
		},
		"timestamp": gin.H{
			"unix": c.GetInt64("request_time"),
		},
	})
}
