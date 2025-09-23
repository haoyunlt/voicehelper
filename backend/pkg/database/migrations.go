package database

import (
	"database/sql"
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// Migration 数据库迁移结构
type Migration struct {
	Version     string
	Description string
	UpSQL       string
	DownSQL     string
	Timestamp   time.Time
}

// MigrationManager 迁移管理器
type MigrationManager struct {
	db         *sql.DB
	migrations []Migration
	tableName  string
	schemaName string
}

// NewMigrationManager 创建迁移管理器
func NewMigrationManager(db *sql.DB) *MigrationManager {
	return &MigrationManager{
		db:         db,
		migrations: []Migration{},
		tableName:  "schema_migrations",
		schemaName: "public",
	}
}

// AddMigration 添加迁移
func (mm *MigrationManager) AddMigration(version, description, upSQL, downSQL string) {
	migration := Migration{
		Version:     version,
		Description: description,
		UpSQL:       upSQL,
		DownSQL:     downSQL,
		Timestamp:   time.Now(),
	}
	mm.migrations = append(mm.migrations, migration)
}

// InitMigrationsTable 初始化迁移表
func (mm *MigrationManager) InitMigrationsTable() error {
	createTableSQL := fmt.Sprintf(`
		CREATE TABLE IF NOT EXISTS %s.%s (
			version VARCHAR(50) PRIMARY KEY,
			description TEXT NOT NULL,
			applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			checksum VARCHAR(64)
		)
	`, mm.schemaName, mm.tableName)

	_, err := mm.db.Exec(createTableSQL)
	if err != nil {
		return fmt.Errorf("failed to create migrations table: %v", err)
	}

	logrus.Info("Migrations table initialized")
	return nil
}

// GetAppliedMigrations 获取已应用的迁移
func (mm *MigrationManager) GetAppliedMigrations() (map[string]bool, error) {
	applied := make(map[string]bool)

	query := fmt.Sprintf("SELECT version FROM %s.%s", mm.schemaName, mm.tableName)
	rows, err := mm.db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to query applied migrations: %v", err)
	}
	defer rows.Close()

	for rows.Next() {
		var version string
		if err := rows.Scan(&version); err != nil {
			return nil, fmt.Errorf("failed to scan migration version: %v", err)
		}
		applied[version] = true
	}

	return applied, nil
}

// Migrate 执行迁移
func (mm *MigrationManager) Migrate() error {
	// 初始化迁移表
	if err := mm.InitMigrationsTable(); err != nil {
		return err
	}

	// 获取已应用的迁移
	applied, err := mm.GetAppliedMigrations()
	if err != nil {
		return err
	}

	// 排序迁移
	sort.Slice(mm.migrations, func(i, j int) bool {
		return mm.migrations[i].Version < mm.migrations[j].Version
	})

	// 执行未应用的迁移
	for _, migration := range mm.migrations {
		if applied[migration.Version] {
			logrus.WithField("version", migration.Version).Debug("Migration already applied")
			continue
		}

		logrus.WithFields(logrus.Fields{
			"version":     migration.Version,
			"description": migration.Description,
		}).Info("Applying migration")

		if err := mm.applyMigration(migration); err != nil {
			return fmt.Errorf("failed to apply migration %s: %v", migration.Version, err)
		}

		logrus.WithField("version", migration.Version).Info("Migration applied successfully")
	}

	return nil
}

// applyMigration 应用单个迁移
func (mm *MigrationManager) applyMigration(migration Migration) error {
	tx, err := mm.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %v", err)
	}
	defer tx.Rollback()

	// 执行迁移SQL
	statements := strings.Split(migration.UpSQL, ";")
	for _, stmt := range statements {
		stmt = strings.TrimSpace(stmt)
		if stmt == "" {
			continue
		}

		if _, err := tx.Exec(stmt); err != nil {
			return fmt.Errorf("failed to execute statement '%s': %v", stmt, err)
		}
	}

	// 记录迁移
	insertSQL := fmt.Sprintf(`
		INSERT INTO %s.%s (version, description, applied_at)
		VALUES ($1, $2, CURRENT_TIMESTAMP)
	`, mm.schemaName, mm.tableName)

	if _, err := tx.Exec(insertSQL, migration.Version, migration.Description); err != nil {
		return fmt.Errorf("failed to record migration: %v", err)
	}

	return tx.Commit()
}

// Rollback 回滚迁移
func (mm *MigrationManager) Rollback(targetVersion string) error {
	applied, err := mm.GetAppliedMigrations()
	if err != nil {
		return err
	}

	// 按版本倒序排序
	sort.Slice(mm.migrations, func(i, j int) bool {
		return mm.migrations[i].Version > mm.migrations[j].Version
	})

	for _, migration := range mm.migrations {
		if !applied[migration.Version] {
			continue
		}

		if migration.Version <= targetVersion {
			break
		}

		logrus.WithFields(logrus.Fields{
			"version":     migration.Version,
			"description": migration.Description,
		}).Info("Rolling back migration")

		if err := mm.rollbackMigration(migration); err != nil {
			return fmt.Errorf("failed to rollback migration %s: %v", migration.Version, err)
		}

		logrus.WithField("version", migration.Version).Info("Migration rolled back successfully")
	}

	return nil
}

// rollbackMigration 回滚单个迁移
func (mm *MigrationManager) rollbackMigration(migration Migration) error {
	tx, err := mm.db.Begin()
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %v", err)
	}
	defer tx.Rollback()

	// 执行回滚SQL
	if migration.DownSQL != "" {
		statements := strings.Split(migration.DownSQL, ";")
		for _, stmt := range statements {
			stmt = strings.TrimSpace(stmt)
			if stmt == "" {
				continue
			}

			if _, err := tx.Exec(stmt); err != nil {
				return fmt.Errorf("failed to execute rollback statement '%s': %v", stmt, err)
			}
		}
	}

	// 删除迁移记录
	deleteSQL := fmt.Sprintf(`
		DELETE FROM %s.%s WHERE version = $1
	`, mm.schemaName, mm.tableName)

	if _, err := tx.Exec(deleteSQL, migration.Version); err != nil {
		return fmt.Errorf("failed to delete migration record: %v", err)
	}

	return tx.Commit()
}

// Status 获取迁移状态
func (mm *MigrationManager) Status() ([]MigrationStatus, error) {
	applied, err := mm.GetAppliedMigrations()
	if err != nil {
		return nil, err
	}

	var status []MigrationStatus
	for _, migration := range mm.migrations {
		s := MigrationStatus{
			Version:     migration.Version,
			Description: migration.Description,
			Applied:     applied[migration.Version],
		}
		status = append(status, s)
	}

	// 按版本排序
	sort.Slice(status, func(i, j int) bool {
		return status[i].Version < status[j].Version
	})

	return status, nil
}

// MigrationStatus 迁移状态
type MigrationStatus struct {
	Version     string `json:"version"`
	Description string `json:"description"`
	Applied     bool   `json:"applied"`
}

// RegisterCoreMigrations 注册核心迁移
func (mm *MigrationManager) RegisterCoreMigrations() {
	// 001: 创建租户表
	mm.AddMigration("001", "Create tenants table", `
		CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
		CREATE EXTENSION IF NOT EXISTS "pgcrypto";
		
		CREATE TABLE IF NOT EXISTS tenants (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(50) UNIQUE NOT NULL,
			name VARCHAR(100) NOT NULL,
			plan VARCHAR(20) DEFAULT 'free',
			status VARCHAR(20) DEFAULT 'active',
			config JSONB DEFAULT '{}',
			quota JSONB DEFAULT '{"daily_tokens": 100000, "daily_audio_minutes": 60, "max_concurrent": 3}',
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
		);
		
		CREATE INDEX IF NOT EXISTS idx_tenants_status ON tenants(status);
		CREATE INDEX IF NOT EXISTS idx_tenants_plan ON tenants(plan);
	`, `
		DROP TABLE IF EXISTS tenants CASCADE;
	`)

	// 002: 创建用户表
	mm.AddMigration("002", "Create users table", `
		CREATE TABLE IF NOT EXISTS users (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			user_id VARCHAR(50) UNIQUE NOT NULL,
			tenant_id VARCHAR(50) NOT NULL,
			openid VARCHAR(100),
			unionid VARCHAR(100),
			username VARCHAR(50),
			nickname VARCHAR(100),
			avatar_url TEXT,
			email VARCHAR(100),
			phone VARCHAR(20),
			role VARCHAR(20) DEFAULT 'user',
			status VARCHAR(20) DEFAULT 'active',
			last_login_at TIMESTAMP WITH TIME ZONE,
			metadata JSONB DEFAULT '{}',
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
		);
		
		CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_users_openid ON users(openid);
		CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);
	`, `
		DROP TABLE IF EXISTS users CASCADE;
	`)

	// 003: 创建会话表
	mm.AddMigration("003", "Create conversations table", `
		CREATE TABLE IF NOT EXISTS conversations (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			conversation_id VARCHAR(50) UNIQUE NOT NULL,
			user_id VARCHAR(50) NOT NULL,
			tenant_id VARCHAR(50) NOT NULL,
			title VARCHAR(200),
			summary TEXT,
			status VARCHAR(20) DEFAULT 'active',
			metadata JSONB DEFAULT '{}',
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
			FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
		);
		
		CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
		CREATE INDEX IF NOT EXISTS idx_conversations_tenant ON conversations(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);
		CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at);
	`, `
		DROP TABLE IF EXISTS conversations CASCADE;
	`)

	// 004: 创建消息表
	mm.AddMigration("004", "Create messages table", `
		CREATE TABLE IF NOT EXISTS messages (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			message_id VARCHAR(50) UNIQUE NOT NULL,
			conversation_id VARCHAR(50) NOT NULL,
			user_id VARCHAR(50) NOT NULL,
			tenant_id VARCHAR(50) NOT NULL,
			role VARCHAR(20) NOT NULL,
			content TEXT NOT NULL,
			content_type VARCHAR(20) DEFAULT 'text',
			metadata JSONB DEFAULT '{}',
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
			FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
			FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
		);
		
		CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
		CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id);
		CREATE INDEX IF NOT EXISTS idx_messages_tenant ON messages(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
		CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
	`, `
		DROP TABLE IF EXISTS messages CASCADE;
	`)

	// 005: 创建文档表
	mm.AddMigration("005", "Create documents table", `
		CREATE TABLE IF NOT EXISTS documents (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			document_id VARCHAR(50) UNIQUE NOT NULL,
			tenant_id VARCHAR(50) NOT NULL,
			title VARCHAR(200) NOT NULL,
			content TEXT NOT NULL,
			content_type VARCHAR(50) DEFAULT 'text/plain',
			source VARCHAR(100),
			status VARCHAR(20) DEFAULT 'active',
			metadata JSONB DEFAULT '{}',
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
		);
		
		CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
		CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
		CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
	`, `
		DROP TABLE IF EXISTS documents CASCADE;
	`)

	// 006: 创建API密钥表
	mm.AddMigration("006", "Create api_keys table", `
		CREATE TABLE IF NOT EXISTS api_keys (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			key_id VARCHAR(50) UNIQUE NOT NULL,
			tenant_id VARCHAR(50) NOT NULL,
			user_id VARCHAR(50),
			name VARCHAR(100) NOT NULL,
			key_hash VARCHAR(128) NOT NULL,
			permissions JSONB DEFAULT '[]',
			status VARCHAR(20) DEFAULT 'active',
			expires_at TIMESTAMP WITH TIME ZONE,
			last_used_at TIMESTAMP WITH TIME ZONE,
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
			FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
		);
		
		CREATE INDEX IF NOT EXISTS idx_api_keys_tenant ON api_keys(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
		CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys(status);
		CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
	`, `
		DROP TABLE IF EXISTS api_keys CASCADE;
	`)

	// 007: 创建语音会话表
	mm.AddMigration("007", "Create voice_sessions table", `
		CREATE TABLE IF NOT EXISTS voice_sessions (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			session_id VARCHAR(50) UNIQUE NOT NULL,
			user_id VARCHAR(50) NOT NULL,
			tenant_id VARCHAR(50) NOT NULL,
			conversation_id VARCHAR(50),
			status VARCHAR(20) DEFAULT 'active',
			config JSONB DEFAULT '{}',
			start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			end_time TIMESTAMP WITH TIME ZONE,
			duration_seconds INTEGER,
			audio_duration_seconds INTEGER,
			metadata JSONB DEFAULT '{}',
			FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
			FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
			FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE SET NULL
		);
		
		CREATE INDEX IF NOT EXISTS idx_voice_sessions_user ON voice_sessions(user_id);
		CREATE INDEX IF NOT EXISTS idx_voice_sessions_tenant ON voice_sessions(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_voice_sessions_conversation ON voice_sessions(conversation_id);
		CREATE INDEX IF NOT EXISTS idx_voice_sessions_status ON voice_sessions(status);
		CREATE INDEX IF NOT EXISTS idx_voice_sessions_start_time ON voice_sessions(start_time);
	`, `
		DROP TABLE IF EXISTS voice_sessions CASCADE;
	`)

	// 008: 创建使用统计表
	mm.AddMigration("008", "Create usage_stats table", `
		CREATE TABLE IF NOT EXISTS usage_stats (
			id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
			tenant_id VARCHAR(50) NOT NULL,
			user_id VARCHAR(50),
			date DATE NOT NULL,
			metric_type VARCHAR(50) NOT NULL,
			metric_value BIGINT DEFAULT 0,
			metadata JSONB DEFAULT '{}',
			created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
			FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
			FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
		);
		
		CREATE UNIQUE INDEX IF NOT EXISTS idx_usage_stats_unique ON usage_stats(tenant_id, user_id, date, metric_type);
		CREATE INDEX IF NOT EXISTS idx_usage_stats_tenant ON usage_stats(tenant_id);
		CREATE INDEX IF NOT EXISTS idx_usage_stats_user ON usage_stats(user_id);
		CREATE INDEX IF NOT EXISTS idx_usage_stats_date ON usage_stats(date);
		CREATE INDEX IF NOT EXISTS idx_usage_stats_type ON usage_stats(metric_type);
	`, `
		DROP TABLE IF EXISTS usage_stats CASCADE;
	`)
}
