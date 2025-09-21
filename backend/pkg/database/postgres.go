package database

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"
)

// Config 数据库配置
type Config struct {
	Host            string        `json:"host" yaml:"host"`
	Port            int           `json:"port" yaml:"port"`
	Database        string        `json:"database" yaml:"database"`
	User            string        `json:"user" yaml:"user"`
	Password        string        `json:"password" yaml:"password"`
	SSLMode         string        `json:"ssl_mode" yaml:"ssl_mode"`
	MaxOpenConns    int           `json:"max_open_conns" yaml:"max_open_conns"`
	MaxIdleConns    int           `json:"max_idle_conns" yaml:"max_idle_conns"`
	ConnMaxLifetime time.Duration `json:"conn_max_lifetime" yaml:"conn_max_lifetime"`
	ConnMaxIdleTime time.Duration `json:"conn_max_idle_time" yaml:"conn_max_idle_time"`
}

// DefaultConfig 返回默认配置
func DefaultConfig() *Config {
	return &Config{
		Host:            "localhost",
		Port:            5432,
		Database:        "chatbot",
		User:            "chatbot",
		Password:        "chatbot123",
		SSLMode:         "disable",
		MaxOpenConns:    25,
		MaxIdleConns:    5,
		ConnMaxLifetime: time.Hour,
		ConnMaxIdleTime: time.Minute * 10,
	}
}

// DB 数据库连接包装
type DB struct {
	*sql.DB
	config *Config
}

// NewPostgresDB 创建PostgreSQL连接
func NewPostgresDB(config *Config) (*DB, error) {
	if config == nil {
		config = DefaultConfig()
	}

	// 构建连接字符串
	dsn := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		config.Host, config.Port, config.User, config.Password,
		config.Database, config.SSLMode,
	)

	// 打开数据库连接
	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// 配置连接池
	db.SetMaxOpenConns(config.MaxOpenConns)
	db.SetMaxIdleConns(config.MaxIdleConns)
	db.SetConnMaxLifetime(config.ConnMaxLifetime)
	db.SetConnMaxIdleTime(config.ConnMaxIdleTime)

	// 测试连接
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	logrus.Infof("Connected to PostgreSQL database: %s@%s:%d/%s",
		config.User, config.Host, config.Port, config.Database)

	return &DB{
		DB:     db,
		config: config,
	}, nil
}

// Close 关闭数据库连接
func (db *DB) Close() error {
	logrus.Info("Closing database connection")
	return db.DB.Close()
}

// Health 健康检查
func (db *DB) Health(ctx context.Context) error {
	return db.PingContext(ctx)
}

// Stats 获取连接池统计信息
func (db *DB) Stats() sql.DBStats {
	return db.DB.Stats()
}

// Transaction 执行事务
func (db *DB) Transaction(ctx context.Context, fn func(*sql.Tx) error) error {
	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}

	defer func() {
		if p := recover(); p != nil {
			tx.Rollback()
			panic(p)
		}
	}()

	if err := fn(tx); err != nil {
		if rbErr := tx.Rollback(); rbErr != nil {
			return fmt.Errorf("tx error: %v, rollback error: %v", err, rbErr)
		}
		return err
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// Migrate 执行数据库迁移
func (db *DB) Migrate(ctx context.Context) error {
	migrations := []string{
		// 用户表
		`CREATE TABLE IF NOT EXISTS users (
			id VARCHAR(36) PRIMARY KEY,
			open_id VARCHAR(255) UNIQUE,
			union_id VARCHAR(255),
			tenant_id VARCHAR(36) NOT NULL DEFAULT 'default',
			username VARCHAR(100),
			nickname VARCHAR(100),
			avatar TEXT,
			email VARCHAR(255),
			phone VARCHAR(20),
			role VARCHAR(50) DEFAULT 'user',
			status VARCHAR(20) DEFAULT 'active',
			last_login TIMESTAMP,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			deleted_at TIMESTAMP,
			INDEX idx_users_openid (open_id),
			INDEX idx_users_tenant (tenant_id),
			INDEX idx_users_status (status)
		)`,

		// 会话表
		`CREATE TABLE IF NOT EXISTS conversations (
			id VARCHAR(36) PRIMARY KEY,
			user_id VARCHAR(36) NOT NULL,
			tenant_id VARCHAR(36) NOT NULL,
			title VARCHAR(255),
			summary TEXT,
			status VARCHAR(20) DEFAULT 'active',
			metadata JSONB DEFAULT '{}',
			msg_count INT DEFAULT 0,
			token_count BIGINT DEFAULT 0,
			last_msg_at TIMESTAMP,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			deleted_at TIMESTAMP,
			INDEX idx_conversations_user (user_id),
			INDEX idx_conversations_status (status),
			INDEX idx_conversations_last_msg (last_msg_at DESC)
		)`,

		// 消息表
		`CREATE TABLE IF NOT EXISTS messages (
			id VARCHAR(36) PRIMARY KEY,
			conversation_id VARCHAR(36) NOT NULL,
			role VARCHAR(20) NOT NULL,
			content TEXT NOT NULL,
			modality VARCHAR(20) DEFAULT 'text',
			token_count INT DEFAULT 0,
			metadata JSONB DEFAULT '{}',
			references JSONB DEFAULT '[]',
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			INDEX idx_messages_conversation (conversation_id),
			INDEX idx_messages_created (created_at)
		)`,

		// 数据集表
		`CREATE TABLE IF NOT EXISTS datasets (
			id VARCHAR(36) PRIMARY KEY,
			tenant_id VARCHAR(36) NOT NULL,
			name VARCHAR(255) NOT NULL,
			description TEXT,
			type VARCHAR(50) DEFAULT 'document',
			status VARCHAR(20) DEFAULT 'active',
			doc_count INT DEFAULT 0,
			chunk_count INT DEFAULT 0,
			token_count BIGINT DEFAULT 0,
			metadata JSONB DEFAULT '{}',
			created_by VARCHAR(36),
			updated_by VARCHAR(36),
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			deleted_at TIMESTAMP,
			INDEX idx_datasets_tenant (tenant_id),
			INDEX idx_datasets_status (status)
		)`,

		// 文档表
		`CREATE TABLE IF NOT EXISTS documents (
			id VARCHAR(36) PRIMARY KEY,
			dataset_id VARCHAR(36) NOT NULL,
			name VARCHAR(255) NOT NULL,
			source TEXT,
			type VARCHAR(50),
			size BIGINT DEFAULT 0,
			status VARCHAR(20) DEFAULT 'pending',
			chunk_count INT DEFAULT 0,
			token_count BIGINT DEFAULT 0,
			metadata JSONB DEFAULT '{}',
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			deleted_at TIMESTAMP,
			INDEX idx_documents_dataset (dataset_id),
			INDEX idx_documents_status (status)
		)`,

		// API密钥表
		`CREATE TABLE IF NOT EXISTS api_keys (
			id VARCHAR(36) PRIMARY KEY,
			tenant_id VARCHAR(36) NOT NULL,
			name VARCHAR(255),
			key VARCHAR(255) UNIQUE NOT NULL,
			secret_hash VARCHAR(255) NOT NULL,
			status VARCHAR(20) DEFAULT 'active',
			permissions JSONB DEFAULT '[]',
			rate_limit INT DEFAULT 1000,
			expires_at TIMESTAMP,
			last_used_at TIMESTAMP,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			deleted_at TIMESTAMP,
			INDEX idx_apikeys_key (key),
			INDEX idx_apikeys_tenant (tenant_id),
			INDEX idx_apikeys_status (status)
		)`,

		// 审计日志表
		`CREATE TABLE IF NOT EXISTS audit_logs (
			id SERIAL PRIMARY KEY,
			tenant_id VARCHAR(36),
			user_id VARCHAR(36),
			action VARCHAR(100) NOT NULL,
			resource VARCHAR(100),
			resource_id VARCHAR(36),
			metadata JSONB DEFAULT '{}',
			ip_address VARCHAR(45),
			user_agent TEXT,
			created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
			INDEX idx_audit_tenant (tenant_id),
			INDEX idx_audit_user (user_id),
			INDEX idx_audit_action (action),
			INDEX idx_audit_created (created_at DESC)
		) PARTITION BY RANGE (created_at)`,

		// 创建分区表（按月）
		`CREATE TABLE IF NOT EXISTS audit_logs_y2024m01 PARTITION OF audit_logs
			FOR VALUES FROM ('2024-01-01') TO ('2024-02-01')`,
		`CREATE TABLE IF NOT EXISTS audit_logs_y2024m02 PARTITION OF audit_logs
			FOR VALUES FROM ('2024-02-01') TO ('2024-03-01')`,
		`CREATE TABLE IF NOT EXISTS audit_logs_y2024m03 PARTITION OF audit_logs
			FOR VALUES FROM ('2024-03-01') TO ('2024-04-01')`,

		// 添加更新时间触发器
		`CREATE OR REPLACE FUNCTION update_updated_at_column()
		RETURNS TRIGGER AS $$
		BEGIN
			NEW.updated_at = CURRENT_TIMESTAMP;
			RETURN NEW;
		END;
		$$ language 'plpgsql'`,

		`CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
			FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()`,
		`CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
			FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()`,
		`CREATE TRIGGER update_datasets_updated_at BEFORE UPDATE ON datasets
			FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()`,
		`CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
			FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()`,
		`CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON api_keys
			FOR EACH ROW EXECUTE FUNCTION update_updated_at_column()`,
	}

	// PostgreSQL特定的迁移（修正MySQL语法）
	pgMigrations := []string{
		// 创建索引
		`CREATE INDEX IF NOT EXISTS idx_users_openid ON users(open_id)`,
		`CREATE INDEX IF NOT EXISTS idx_users_tenant ON users(tenant_id)`,
		`CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)`,
		`CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)`,
		`CREATE INDEX IF NOT EXISTS idx_datasets_tenant ON datasets(tenant_id)`,
		`CREATE INDEX IF NOT EXISTS idx_documents_dataset ON documents(dataset_id)`,
		`CREATE INDEX IF NOT EXISTS idx_apikeys_key ON api_keys(key)`,
		
		// 添加外键约束
		`ALTER TABLE conversations ADD CONSTRAINT fk_conversations_user 
			FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE`,
		`ALTER TABLE messages ADD CONSTRAINT fk_messages_conversation 
			FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE`,
		`ALTER TABLE documents ADD CONSTRAINT fk_documents_dataset 
			FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE`,
	}

	// 执行基础迁移
	for _, migration := range migrations {
		if _, err := db.ExecContext(ctx, migration); err != nil {
			// 忽略已存在的错误
			logrus.Warnf("Migration warning: %v", err)
		}
	}

	// 执行PostgreSQL特定迁移
	for _, migration := range pgMigrations {
		if _, err := db.ExecContext(ctx, migration); err != nil {
			// 忽略已存在的错误
			logrus.Warnf("PG migration warning: %v", err)
		}
	}

	logrus.Info("Database migration completed")
	return nil
}
