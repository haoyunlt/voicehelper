package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/sirupsen/logrus"
)

// VoiceSession 语音会话模型
type VoiceSession struct {
	ID                   string                 `json:"id"`
	SessionID            string                 `json:"session_id"`
	UserID               string                 `json:"user_id"`
	TenantID             string                 `json:"tenant_id"`
	ConversationID       string                 `json:"conversation_id,omitempty"`
	Status               string                 `json:"status"` // active, completed, failed, cancelled
	Config               map[string]interface{} `json:"config"`
	StartTime            time.Time              `json:"start_time"`
	EndTime              *time.Time             `json:"end_time,omitempty"`
	DurationSeconds      int                    `json:"duration_seconds"`
	AudioDurationSeconds int                    `json:"audio_duration_seconds"`
	Metadata             map[string]interface{} `json:"metadata"`
}

// VoiceSessionRepository 语音会话仓库接口
type VoiceSessionRepository interface {
	Create(ctx context.Context, session *VoiceSession) error
	GetByID(ctx context.Context, sessionID string) (*VoiceSession, error)
	GetByUserID(ctx context.Context, userID string, limit, offset int) ([]*VoiceSession, error)
	GetByTenantID(ctx context.Context, tenantID string, limit, offset int) ([]*VoiceSession, error)
	GetActiveByUserID(ctx context.Context, userID string) ([]*VoiceSession, error)
	Update(ctx context.Context, session *VoiceSession) error
	Complete(ctx context.Context, sessionID string, endTime time.Time, metadata map[string]interface{}) error
	GetStats(ctx context.Context, tenantID string, startTime, endTime time.Time) (*VoiceSessionStats, error)
	CleanupExpiredSessions(ctx context.Context, expireAfter time.Duration) (int, error)
}

// VoiceSessionStats 语音会话统计
type VoiceSessionStats struct {
	TenantID             string  `json:"tenant_id"`
	TotalSessions        int     `json:"total_sessions"`
	CompletedSessions    int     `json:"completed_sessions"`
	FailedSessions       int     `json:"failed_sessions"`
	CancelledSessions    int     `json:"cancelled_sessions"`
	TotalDurationSeconds int     `json:"total_duration_seconds"`
	AvgDurationSeconds   float64 `json:"avg_duration_seconds"`
	TotalAudioMinutes    int     `json:"total_audio_minutes"`
}

// PostgresVoiceSessionRepository PostgreSQL语音会话仓库实现
type PostgresVoiceSessionRepository struct {
	db *sql.DB
}

// NewPostgresVoiceSessionRepository 创建PostgreSQL语音会话仓库
func NewPostgresVoiceSessionRepository(db *sql.DB) VoiceSessionRepository {
	return &PostgresVoiceSessionRepository{db: db}
}

// Create 创建语音会话
func (r *PostgresVoiceSessionRepository) Create(ctx context.Context, session *VoiceSession) error {
	if session.ID == "" {
		session.ID = uuid.New().String()
	}

	configJSON, err := json.Marshal(session.Config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	metadataJSON, err := json.Marshal(session.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %v", err)
	}

	query := `
		INSERT INTO voice_sessions (id, session_id, user_id, tenant_id, conversation_id, status, config, start_time, metadata)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
	`

	_, err = r.db.ExecContext(ctx, query,
		session.ID, session.SessionID, session.UserID, session.TenantID,
		nullString(session.ConversationID), session.Status, configJSON,
		session.StartTime, metadataJSON)
	if err != nil {
		return fmt.Errorf("failed to create voice session: %v", err)
	}

	logrus.WithFields(logrus.Fields{
		"session_id": session.SessionID,
		"user_id":    session.UserID,
		"tenant_id":  session.TenantID,
	}).Info("Voice session created")

	return nil
}

// GetByID 根据ID获取语音会话
func (r *PostgresVoiceSessionRepository) GetByID(ctx context.Context, sessionID string) (*VoiceSession, error) {
	query := `
		SELECT id, session_id, user_id, tenant_id, conversation_id, status, config, 
		       start_time, end_time, duration_seconds, audio_duration_seconds, metadata
		FROM voice_sessions WHERE session_id = $1
	`

	return r.scanVoiceSession(ctx, query, sessionID)
}

// GetByUserID 根据用户ID获取语音会话
func (r *PostgresVoiceSessionRepository) GetByUserID(ctx context.Context, userID string, limit, offset int) ([]*VoiceSession, error) {
	query := `
		SELECT id, session_id, user_id, tenant_id, conversation_id, status, config, 
		       start_time, end_time, duration_seconds, audio_duration_seconds, metadata
		FROM voice_sessions 
		WHERE user_id = $1
		ORDER BY start_time DESC
		LIMIT $2 OFFSET $3
	`

	rows, err := r.db.QueryContext(ctx, query, userID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to get voice sessions by user: %v", err)
	}
	defer rows.Close()

	var sessions []*VoiceSession
	for rows.Next() {
		session, err := r.scanVoiceSessionRow(rows)
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// GetByTenantID 根据租户ID获取语音会话
func (r *PostgresVoiceSessionRepository) GetByTenantID(ctx context.Context, tenantID string, limit, offset int) ([]*VoiceSession, error) {
	query := `
		SELECT id, session_id, user_id, tenant_id, conversation_id, status, config, 
		       start_time, end_time, duration_seconds, audio_duration_seconds, metadata
		FROM voice_sessions 
		WHERE tenant_id = $1
		ORDER BY start_time DESC
		LIMIT $2 OFFSET $3
	`

	rows, err := r.db.QueryContext(ctx, query, tenantID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to get voice sessions by tenant: %v", err)
	}
	defer rows.Close()

	var sessions []*VoiceSession
	for rows.Next() {
		session, err := r.scanVoiceSessionRow(rows)
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// GetActiveByUserID 获取用户的活跃语音会话
func (r *PostgresVoiceSessionRepository) GetActiveByUserID(ctx context.Context, userID string) ([]*VoiceSession, error) {
	query := `
		SELECT id, session_id, user_id, tenant_id, conversation_id, status, config, 
		       start_time, end_time, duration_seconds, audio_duration_seconds, metadata
		FROM voice_sessions 
		WHERE user_id = $1 AND status = 'active'
		ORDER BY start_time DESC
	`

	rows, err := r.db.QueryContext(ctx, query, userID)
	if err != nil {
		return nil, fmt.Errorf("failed to get active voice sessions: %v", err)
	}
	defer rows.Close()

	var sessions []*VoiceSession
	for rows.Next() {
		session, err := r.scanVoiceSessionRow(rows)
		if err != nil {
			return nil, err
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// Update 更新语音会话
func (r *PostgresVoiceSessionRepository) Update(ctx context.Context, session *VoiceSession) error {
	configJSON, err := json.Marshal(session.Config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %v", err)
	}

	metadataJSON, err := json.Marshal(session.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %v", err)
	}

	query := `
		UPDATE voice_sessions 
		SET status = $2, config = $3, end_time = $4, duration_seconds = $5, 
		    audio_duration_seconds = $6, metadata = $7
		WHERE session_id = $1
	`

	result, err := r.db.ExecContext(ctx, query,
		session.SessionID, session.Status, configJSON, session.EndTime,
		session.DurationSeconds, session.AudioDurationSeconds, metadataJSON)
	if err != nil {
		return fmt.Errorf("failed to update voice session: %v", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %v", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("voice session not found: %s", session.SessionID)
	}

	logrus.WithField("session_id", session.SessionID).Info("Voice session updated")
	return nil
}

// Complete 完成语音会话
func (r *PostgresVoiceSessionRepository) Complete(ctx context.Context, sessionID string, endTime time.Time, metadata map[string]interface{}) error {
	metadataJSON, err := json.Marshal(metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %v", err)
	}

	query := `
		UPDATE voice_sessions 
		SET status = 'completed', end_time = $2, 
		    duration_seconds = EXTRACT(EPOCH FROM ($2 - start_time))::INTEGER,
		    metadata = $3
		WHERE session_id = $1
	`

	result, err := r.db.ExecContext(ctx, query, sessionID, endTime, metadataJSON)
	if err != nil {
		return fmt.Errorf("failed to complete voice session: %v", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %v", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("voice session not found: %s", sessionID)
	}

	logrus.WithField("session_id", sessionID).Info("Voice session completed")
	return nil
}

// GetStats 获取语音会话统计
func (r *PostgresVoiceSessionRepository) GetStats(ctx context.Context, tenantID string, startTime, endTime time.Time) (*VoiceSessionStats, error) {
	query := `
		SELECT 
			COUNT(*) as total_sessions,
			COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_sessions,
			COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_sessions,
			COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_sessions,
			COALESCE(SUM(duration_seconds), 0) as total_duration_seconds,
			COALESCE(AVG(duration_seconds), 0) as avg_duration_seconds,
			COALESCE(SUM(audio_duration_seconds), 0) / 60 as total_audio_minutes
		FROM voice_sessions 
		WHERE tenant_id = $1 AND start_time >= $2 AND start_time <= $3
	`

	stats := &VoiceSessionStats{TenantID: tenantID}

	err := r.db.QueryRowContext(ctx, query, tenantID, startTime, endTime).Scan(
		&stats.TotalSessions, &stats.CompletedSessions, &stats.FailedSessions,
		&stats.CancelledSessions, &stats.TotalDurationSeconds, &stats.AvgDurationSeconds,
		&stats.TotalAudioMinutes,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to get voice session stats: %v", err)
	}

	return stats, nil
}

// CleanupExpiredSessions 清理过期会话
func (r *PostgresVoiceSessionRepository) CleanupExpiredSessions(ctx context.Context, expireAfter time.Duration) (int, error) {
	expireTime := time.Now().Add(-expireAfter)

	query := `
		UPDATE voice_sessions 
		SET status = 'expired'
		WHERE status = 'active' AND start_time < $1
	`

	result, err := r.db.ExecContext(ctx, query, expireTime)
	if err != nil {
		return 0, fmt.Errorf("failed to cleanup expired sessions: %v", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("failed to get rows affected: %v", err)
	}

	if rowsAffected > 0 {
		logrus.WithFields(logrus.Fields{
			"expired_sessions": rowsAffected,
			"expire_time":      expireTime,
		}).Info("Cleaned up expired voice sessions")
	}

	return int(rowsAffected), nil
}

// scanVoiceSession 扫描单个语音会话
func (r *PostgresVoiceSessionRepository) scanVoiceSession(ctx context.Context, query string, args ...interface{}) (*VoiceSession, error) {
	row := r.db.QueryRowContext(ctx, query, args...)
	return r.scanVoiceSessionRow(row)
}

// scanVoiceSessionRow 扫描语音会话行
func (r *PostgresVoiceSessionRepository) scanVoiceSessionRow(scanner interface {
	Scan(dest ...interface{}) error
}) (*VoiceSession, error) {
	var session VoiceSession
	var conversationID sql.NullString
	var endTime sql.NullTime
	var configJSON, metadataJSON []byte

	err := scanner.Scan(
		&session.ID, &session.SessionID, &session.UserID, &session.TenantID,
		&conversationID, &session.Status, &configJSON, &session.StartTime,
		&endTime, &session.DurationSeconds, &session.AudioDurationSeconds, &metadataJSON,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("voice session not found")
		}
		return nil, fmt.Errorf("failed to scan voice session: %v", err)
	}

	// 处理可空字段
	if conversationID.Valid {
		session.ConversationID = conversationID.String
	}

	if endTime.Valid {
		session.EndTime = &endTime.Time
	}

	// 解析JSON字段
	if err := json.Unmarshal(configJSON, &session.Config); err != nil {
		session.Config = make(map[string]interface{})
	}

	if err := json.Unmarshal(metadataJSON, &session.Metadata); err != nil {
		session.Metadata = make(map[string]interface{})
	}

	return &session, nil
}

// nullString 处理可空字符串
func nullString(s string) sql.NullString {
	if s == "" {
		return sql.NullString{Valid: false}
	}
	return sql.NullString{String: s, Valid: true}
}
