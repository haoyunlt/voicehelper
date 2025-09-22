package security

import (
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// SQLSecurityConfig SQL安全配置
type SQLSecurityConfig struct {
	// 最大查询执行时间
	MaxQueryTimeout time.Duration
	// 最大返回行数
	MaxRows int
	// 是否启用查询日志
	EnableQueryLog bool
	// 是否启用SQL注入检测
	EnableInjectionDetection bool
	// 允许的表名白名单
	AllowedTables []string
	// 禁止的SQL关键词
	ForbiddenKeywords []string
}

// DefaultSQLSecurityConfig 默认安全配置
func DefaultSQLSecurityConfig() *SQLSecurityConfig {
	return &SQLSecurityConfig{
		MaxQueryTimeout:          30 * time.Second,
		MaxRows:                  10000,
		EnableQueryLog:           true,
		EnableInjectionDetection: true,
		AllowedTables: []string{
			"users", "conversations", "messages", "datasets",
			"documents", "api_keys", "audit_logs",
		},
		ForbiddenKeywords: []string{
			"DROP", "TRUNCATE", "ALTER", "CREATE", "GRANT", "REVOKE",
			"EXEC", "EXECUTE", "xp_", "sp_", "UNION ALL", "INFORMATION_SCHEMA",
		},
	}
}

// SecureDB 安全数据库包装器
type SecureDB struct {
	db     *sql.DB
	config *SQLSecurityConfig
	logger *logrus.Logger
}

// NewSecureDB 创建安全数据库包装器
func NewSecureDB(db *sql.DB, config *SQLSecurityConfig) *SecureDB {
	if config == nil {
		config = DefaultSQLSecurityConfig()
	}

	return &SecureDB{
		db:     db,
		config: config,
		logger: logrus.New(),
	}
}

// SQLInjectionDetector SQL注入检测器
type SQLInjectionDetector struct {
	patterns []*regexp.Regexp
}

// NewSQLInjectionDetector 创建SQL注入检测器
func NewSQLInjectionDetector() *SQLInjectionDetector {
	patterns := []*regexp.Regexp{
		// 经典SQL注入模式
		regexp.MustCompile(`(?i)(union\s+select|union\s+all\s+select)`),
		regexp.MustCompile(`(?i)(or\s+1\s*=\s*1|or\s+'1'\s*=\s*'1')`),
		regexp.MustCompile(`(?i)(and\s+1\s*=\s*1|and\s+'1'\s*=\s*'1')`),
		regexp.MustCompile(`(?i)(drop\s+table|drop\s+database)`),
		regexp.MustCompile(`(?i)(insert\s+into|update\s+.*\s+set)`),
		regexp.MustCompile(`(?i)(delete\s+from)`),
		regexp.MustCompile(`(?i)(exec\s*\(|execute\s*\()`),
		regexp.MustCompile(`(?i)(script\s*>|javascript:|vbscript:)`),
		regexp.MustCompile(`(?i)(information_schema|sys\.|master\.)`),
		regexp.MustCompile(`(?i)(xp_cmdshell|sp_executesql)`),
		// 时间盲注
		regexp.MustCompile(`(?i)(waitfor\s+delay|pg_sleep|sleep\s*\()`),
		// 布尔盲注
		regexp.MustCompile(`(?i)(substring\s*\(|ascii\s*\(|length\s*\()`),
		// 错误注入
		regexp.MustCompile(`(?i)(convert\s*\(|cast\s*\(.*as)`),
		// 堆叠查询
		regexp.MustCompile(`(?i)(;\s*drop|;\s*delete|;\s*update|;\s*insert)`),
	}

	return &SQLInjectionDetector{patterns: patterns}
}

// DetectInjection 检测SQL注入
func (d *SQLInjectionDetector) DetectInjection(input string) (bool, string) {
	for _, pattern := range d.patterns {
		if pattern.MatchString(input) {
			return true, pattern.String()
		}
	}
	return false, ""
}

// ValidateQuery 验证查询安全性
func (sdb *SecureDB) ValidateQuery(query string, args ...interface{}) error {
	if sdb.config.EnableInjectionDetection {
		detector := NewSQLInjectionDetector()

		// 检查查询本身
		if isInjection, pattern := detector.DetectInjection(query); isInjection {
			sdb.logger.Warnf("SQL injection detected in query: %s, pattern: %s", query, pattern)
			return fmt.Errorf("potentially malicious SQL query detected")
		}

		// 检查参数
		for i, arg := range args {
			if str, ok := arg.(string); ok {
				if isInjection, pattern := detector.DetectInjection(str); isInjection {
					sdb.logger.Warnf("SQL injection detected in parameter %d: %s, pattern: %s", i, str, pattern)
					return fmt.Errorf("potentially malicious SQL parameter detected")
				}
			}
		}
	}

	// 检查表名白名单
	if len(sdb.config.AllowedTables) > 0 {
		if err := sdb.validateTableAccess(query); err != nil {
			return err
		}
	}

	// 检查禁止的关键词
	if len(sdb.config.ForbiddenKeywords) > 0 {
		if err := sdb.validateKeywords(query); err != nil {
			return err
		}
	}

	return nil
}

// validateTableAccess 验证表访问权限
func (sdb *SecureDB) validateTableAccess(query string) error {
	upperQuery := strings.ToUpper(query)

	// 提取表名的简单正则（可以改进）
	tablePattern := regexp.MustCompile(`(?i)(FROM|JOIN|INTO|UPDATE)\s+([a-zA-Z_][a-zA-Z0-9_]*)`)
	matches := tablePattern.FindAllStringSubmatch(upperQuery, -1)

	for _, match := range matches {
		if len(match) > 2 {
			tableName := strings.ToLower(match[2])
			allowed := false
			for _, allowedTable := range sdb.config.AllowedTables {
				if tableName == strings.ToLower(allowedTable) {
					allowed = true
					break
				}
			}
			if !allowed {
				return fmt.Errorf("access to table '%s' is not allowed", tableName)
			}
		}
	}

	return nil
}

// validateKeywords 验证禁止的关键词
func (sdb *SecureDB) validateKeywords(query string) error {
	upperQuery := strings.ToUpper(query)

	for _, keyword := range sdb.config.ForbiddenKeywords {
		if strings.Contains(upperQuery, strings.ToUpper(keyword)) {
			return fmt.Errorf("forbidden keyword '%s' detected in query", keyword)
		}
	}

	return nil
}

// QueryContext 安全的查询执行
func (sdb *SecureDB) QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error) {
	// 验证查询
	if err := sdb.ValidateQuery(query, args...); err != nil {
		return nil, err
	}

	// 设置超时
	if sdb.config.MaxQueryTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, sdb.config.MaxQueryTimeout)
		defer cancel()
	}

	// 记录查询日志
	if sdb.config.EnableQueryLog {
		start := time.Now()
		defer func() {
			duration := time.Since(start)
			sdb.logger.Infof("SQL Query executed in %v: %s", duration, query)
		}()
	}

	return sdb.db.QueryContext(ctx, query, args...)
}

// QueryRowContext 安全的单行查询
func (sdb *SecureDB) QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row {
	// 验证查询
	if err := sdb.ValidateQuery(query, args...); err != nil {
		// 返回一个包含错误的Row
		return &sql.Row{}
	}

	// 设置超时
	if sdb.config.MaxQueryTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, sdb.config.MaxQueryTimeout)
		defer cancel()
	}

	// 记录查询日志
	if sdb.config.EnableQueryLog {
		start := time.Now()
		defer func() {
			duration := time.Since(start)
			sdb.logger.Infof("SQL QueryRow executed in %v: %s", duration, query)
		}()
	}

	return sdb.db.QueryRowContext(ctx, query, args...)
}

// ExecContext 安全的执行操作
func (sdb *SecureDB) ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error) {
	// 验证查询
	if err := sdb.ValidateQuery(query, args...); err != nil {
		return nil, err
	}

	// 设置超时
	if sdb.config.MaxQueryTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, sdb.config.MaxQueryTimeout)
		defer cancel()
	}

	// 记录查询日志
	if sdb.config.EnableQueryLog {
		start := time.Now()
		defer func() {
			duration := time.Since(start)
			sdb.logger.Infof("SQL Exec executed in %v: %s", duration, query)
		}()
	}

	return sdb.db.ExecContext(ctx, query, args...)
}

// PrepareContext 安全的预处理语句
func (sdb *SecureDB) PrepareContext(ctx context.Context, query string) (*sql.Stmt, error) {
	// 验证查询
	if err := sdb.ValidateQuery(query); err != nil {
		return nil, err
	}

	return sdb.db.PrepareContext(ctx, query)
}

// SanitizeInput 输入清理函数
func SanitizeInput(input string) string {
	// 移除潜在危险字符
	dangerous := []string{
		"'", "\"", ";", "--", "/*", "*/", "xp_", "sp_",
		"<script", "</script>", "javascript:", "vbscript:",
	}

	result := input
	for _, danger := range dangerous {
		result = strings.ReplaceAll(result, danger, "")
	}

	return strings.TrimSpace(result)
}

// ValidateTableName 验证表名
func ValidateTableName(tableName string) error {
	// 表名只能包含字母、数字和下划线
	matched, _ := regexp.MatchString(`^[a-zA-Z_][a-zA-Z0-9_]*$`, tableName)
	if !matched {
		return fmt.Errorf("invalid table name: %s", tableName)
	}

	return nil
}

// ValidateColumnName 验证列名
func ValidateColumnName(columnName string) error {
	// 列名只能包含字母、数字和下划线
	matched, _ := regexp.MatchString(`^[a-zA-Z_][a-zA-Z0-9_]*$`, columnName)
	if !matched {
		return fmt.Errorf("invalid column name: %s", columnName)
	}

	return nil
}

// BuildSafeOrderBy 构建安全的ORDER BY子句
func BuildSafeOrderBy(column, direction string) (string, error) {
	// 验证列名
	if err := ValidateColumnName(column); err != nil {
		return "", err
	}

	// 验证排序方向
	direction = strings.ToUpper(strings.TrimSpace(direction))
	if direction != "ASC" && direction != "DESC" {
		direction = "ASC"
	}

	return fmt.Sprintf("ORDER BY %s %s", column, direction), nil
}

// BuildSafeLimit 构建安全的LIMIT子句
func BuildSafeLimit(limit, offset int) string {
	// 确保limit和offset为正数
	if limit <= 0 {
		limit = 10
	}
	if limit > 1000 {
		limit = 1000 // 最大限制
	}
	if offset < 0 {
		offset = 0
	}

	return fmt.Sprintf("LIMIT %d OFFSET %d", limit, offset)
}
