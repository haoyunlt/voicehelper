package database

import (
	"database/sql"
	"fmt"
	"time"

	_ "github.com/lib/pq"
	"github.com/sirupsen/logrus"

	"voicehelper/backend/pkg/config"
)

// PostgresConnection PostgreSQL连接
type PostgresConnection struct {
	DB *sql.DB
}

// NewPostgresConnection 创建PostgreSQL连接
func NewPostgresConnection(cfg config.DatabaseConfig) (*PostgresConnection, error) {
	dsn := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=%s",
		cfg.Host, cfg.Port, cfg.Username, cfg.Password, cfg.Database, cfg.SSLMode,
	)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %v", err)
	}

	// 配置连接池
	db.SetMaxOpenConns(25)
	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(5 * time.Minute)

	// 测试连接
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %v", err)
	}

	logrus.WithFields(logrus.Fields{
		"host":     cfg.Host,
		"port":     cfg.Port,
		"database": cfg.Database,
	}).Info("Connected to PostgreSQL database")

	return &PostgresConnection{DB: db}, nil
}

// Close 关闭数据库连接
func (p *PostgresConnection) Close() error {
	if p.DB != nil {
		return p.DB.Close()
	}
	return nil
}

// Health 检查数据库健康状态
func (p *PostgresConnection) Health() error {
	return p.DB.Ping()
}
