package database

import (
	"database/sql"
	"log"
	"time"
)

// OptimizedConnectionPool 优化的数据库连接池配置
type OptimizedConnectionPool struct {
	DB *sql.DB
}

// NewOptimizedPool 创建优化的连接池
func NewOptimizedPool(databaseURL string) (*OptimizedConnectionPool, error) {
	db, err := sql.Open("postgres", databaseURL)
	if err != nil {
		return nil, err
	}

	// 优化连接池参数 - 基于性能测试结果
	// 减少最大连接数以降低内存使用
	db.SetMaxOpenConns(15) // 从25降到15
	db.SetMaxIdleConns(5)  // 从10降到5

	// 减少连接生存时间以释放资源
	db.SetConnMaxLifetime(3 * time.Minute)  // 从5分钟降到3分钟
	db.SetConnMaxIdleTime(30 * time.Second) // 新增空闲超时

	// 验证连接
	if err := db.Ping(); err != nil {
		return nil, err
	}

	log.Println("数据库连接池已优化配置")
	log.Printf("最大连接数: %d", 15)
	log.Printf("最大空闲连接数: %d", 5)
	log.Printf("连接最大生存时间: %v", 3*time.Minute)
	log.Printf("连接最大空闲时间: %v", 30*time.Second)

	return &OptimizedConnectionPool{DB: db}, nil
}

// GetStats 获取连接池统计信息
func (pool *OptimizedConnectionPool) GetStats() sql.DBStats {
	return pool.DB.Stats()
}

// LogStats 记录连接池统计信息
func (pool *OptimizedConnectionPool) LogStats() {
	stats := pool.GetStats()
	log.Printf("连接池统计 - 打开连接数: %d, 使用中: %d, 空闲: %d",
		stats.OpenConnections, stats.InUse, stats.Idle)
}

// Close 关闭连接池
func (pool *OptimizedConnectionPool) Close() error {
	return pool.DB.Close()
}
