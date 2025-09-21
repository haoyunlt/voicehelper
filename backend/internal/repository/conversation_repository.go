package repository

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/lib/pq"
)

// Conversation 会话模型
type Conversation struct {
	ID         string                 `json:"id"`
	UserID     string                 `json:"user_id"`
	TenantID   string                 `json:"tenant_id"`
	Title      string                 `json:"title"`
	Summary    string                 `json:"summary"`
	Status     string                 `json:"status"` // active, archived
	Metadata   map[string]interface{} `json:"metadata"`
	CreatedAt  time.Time              `json:"created_at"`
	UpdatedAt  time.Time              `json:"updated_at"`
	LastMsgAt  time.Time              `json:"last_msg_at"`
	MsgCount   int                    `json:"msg_count"`
	TokenCount int64                  `json:"token_count"`
}

// Message 消息模型
type Message struct {
	ID             string                 `json:"id"`
	ConversationID string                 `json:"conversation_id"`
	Role           string                 `json:"role"` // user, assistant, system
	Content        string                 `json:"content"`
	Modality       string                 `json:"modality"` // text, voice, image
	TokenCount     int                    `json:"token_count"`
	Metadata       map[string]interface{} `json:"metadata"`
	References     []Reference            `json:"references,omitempty"`
	CreatedAt      time.Time              `json:"created_at"`
}

// Reference 引用信息
type Reference struct {
	ChunkID    string  `json:"chunk_id"`
	Source     string  `json:"source"`
	Content    string  `json:"content"`
	Score      float64 `json:"score"`
	PageNumber int     `json:"page_number,omitempty"`
}

// ConversationRepository 会话仓库接口
type ConversationRepository interface {
	Create(ctx context.Context, conv *Conversation) error
	Get(ctx context.Context, id string) (*Conversation, error)
	GetByUserID(ctx context.Context, userID string, opts ListOptions) ([]*Conversation, int, error)
	Update(ctx context.Context, conv *Conversation) error
	Delete(ctx context.Context, id string) error
	Archive(ctx context.Context, id string) error

	// Message operations
	AddMessage(ctx context.Context, msg *Message) error
	GetMessages(ctx context.Context, convID string, opts ListOptions) ([]*Message, int, error)
	GetLastMessages(ctx context.Context, convID string, limit int) ([]*Message, error)
	UpdateMessageReferences(ctx context.Context, msgID string, refs []Reference) error

	// Statistics
	UpdateStats(ctx context.Context, convID string) error
	GetUserStats(ctx context.Context, userID string) (map[string]interface{}, error)
}

// PostgresConversationRepository PostgreSQL实现
type PostgresConversationRepository struct {
	db *sql.DB
}

// NewPostgresConversationRepository 创建PostgreSQL会话仓库
func NewPostgresConversationRepository(db *sql.DB) ConversationRepository {
	return &PostgresConversationRepository{db: db}
}

// Create 创建会话
func (r *PostgresConversationRepository) Create(ctx context.Context, conv *Conversation) error {
	if conv.ID == "" {
		conv.ID = uuid.New().String()
	}
	if conv.Status == "" {
		conv.Status = "active"
	}
	if conv.Title == "" {
		conv.Title = "新会话"
	}

	// 序列化metadata
	metadataJSON, err := json.Marshal(conv.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		INSERT INTO conversations (
			id, user_id, tenant_id, title, summary, status,
			metadata, created_at, updated_at, last_msg_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, NOW(), NOW(), NOW()
		)
	`

	_, err = r.db.ExecContext(ctx, query,
		conv.ID, conv.UserID, conv.TenantID, conv.Title,
		conv.Summary, conv.Status, metadataJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to create conversation: %w", err)
	}

	return nil
}

// Get 获取会话
func (r *PostgresConversationRepository) Get(ctx context.Context, id string) (*Conversation, error) {
	query := `
		SELECT 
			id, user_id, tenant_id, title, summary, status,
			metadata, created_at, updated_at, last_msg_at,
			msg_count, token_count
		FROM conversations
		WHERE id = $1 AND deleted_at IS NULL
	`

	conv := &Conversation{}
	var metadataJSON []byte

	err := r.db.QueryRowContext(ctx, query, id).Scan(
		&conv.ID, &conv.UserID, &conv.TenantID, &conv.Title,
		&conv.Summary, &conv.Status, &metadataJSON,
		&conv.CreatedAt, &conv.UpdatedAt, &conv.LastMsgAt,
		&conv.MsgCount, &conv.TokenCount,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("conversation not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get conversation: %w", err)
	}

	// 解析metadata
	if len(metadataJSON) > 0 {
		if err := json.Unmarshal(metadataJSON, &conv.Metadata); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metadata: %w", err)
		}
	}

	return conv, nil
}

// GetByUserID 获取用户的会话列表
func (r *PostgresConversationRepository) GetByUserID(ctx context.Context, userID string, opts ListOptions) ([]*Conversation, int, error) {
	// 设置默认值
	if opts.Limit == 0 {
		opts.Limit = 20
	}
	if opts.SortBy == "" {
		opts.SortBy = "last_msg_at"
	}
	if opts.Order == "" {
		opts.Order = "desc"
	}

	// 计算总数
	countQuery := `
		SELECT COUNT(*) 
		FROM conversations 
		WHERE user_id = $1 AND deleted_at IS NULL
	`
	var total int
	err := r.db.QueryRowContext(ctx, countQuery, userID).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count conversations: %w", err)
	}

	// 查询数据
	query := fmt.Sprintf(`
		SELECT 
			id, user_id, tenant_id, title, summary, status,
			metadata, created_at, updated_at, last_msg_at,
			msg_count, token_count
		FROM conversations
		WHERE user_id = $1 AND deleted_at IS NULL
		ORDER BY %s %s
		LIMIT $2 OFFSET $3
	`, opts.SortBy, opts.Order)

	rows, err := r.db.QueryContext(ctx, query, userID, opts.Limit, opts.Offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list conversations: %w", err)
	}
	defer rows.Close()

	var conversations []*Conversation
	for rows.Next() {
		conv := &Conversation{}
		var metadataJSON []byte

		err := rows.Scan(
			&conv.ID, &conv.UserID, &conv.TenantID, &conv.Title,
			&conv.Summary, &conv.Status, &metadataJSON,
			&conv.CreatedAt, &conv.UpdatedAt, &conv.LastMsgAt,
			&conv.MsgCount, &conv.TokenCount,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan conversation: %w", err)
		}

		// 解析metadata
		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &conv.Metadata); err != nil {
				return nil, 0, fmt.Errorf("failed to unmarshal metadata: %w", err)
			}
		}

		conversations = append(conversations, conv)
	}

	return conversations, total, nil
}

// Update 更新会话
func (r *PostgresConversationRepository) Update(ctx context.Context, conv *Conversation) error {
	metadataJSON, err := json.Marshal(conv.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	query := `
		UPDATE conversations SET
			title = $2,
			summary = $3,
			status = $4,
			metadata = $5,
			updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query,
		conv.ID, conv.Title, conv.Summary, conv.Status, metadataJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to update conversation: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("conversation not found")
	}

	return nil
}

// Delete 软删除会话
func (r *PostgresConversationRepository) Delete(ctx context.Context, id string) error {
	query := `
		UPDATE conversations 
		SET deleted_at = NOW(), updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete conversation: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("conversation not found")
	}

	return nil
}

// Archive 归档会话
func (r *PostgresConversationRepository) Archive(ctx context.Context, id string) error {
	query := `
		UPDATE conversations 
		SET status = 'archived', updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	_, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to archive conversation: %w", err)
	}

	return nil
}

// AddMessage 添加消息
func (r *PostgresConversationRepository) AddMessage(ctx context.Context, msg *Message) error {
	if msg.ID == "" {
		msg.ID = uuid.New().String()
	}

	// 序列化metadata和references
	metadataJSON, err := json.Marshal(msg.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}

	referencesJSON, err := json.Marshal(msg.References)
	if err != nil {
		return fmt.Errorf("failed to marshal references: %w", err)
	}

	// 开始事务
	tx, err := r.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to begin transaction: %w", err)
	}
	defer tx.Rollback()

	// 插入消息
	query := `
		INSERT INTO messages (
			id, conversation_id, role, content, modality,
			token_count, metadata, references, created_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, NOW()
		)
	`

	_, err = tx.ExecContext(ctx, query,
		msg.ID, msg.ConversationID, msg.Role, msg.Content,
		msg.Modality, msg.TokenCount, metadataJSON, referencesJSON,
	)

	if err != nil {
		return fmt.Errorf("failed to add message: %w", err)
	}

	// 更新会话统计
	updateQuery := `
		UPDATE conversations 
		SET 
			msg_count = msg_count + 1,
			token_count = token_count + $2,
			last_msg_at = NOW(),
			updated_at = NOW()
		WHERE id = $1
	`

	_, err = tx.ExecContext(ctx, updateQuery, msg.ConversationID, msg.TokenCount)
	if err != nil {
		return fmt.Errorf("failed to update conversation stats: %w", err)
	}

	// 如果是第一条用户消息，自动生成标题
	if msg.Role == "user" {
		var msgCount int
		countQuery := `SELECT msg_count FROM conversations WHERE id = $1`
		err = tx.QueryRowContext(ctx, countQuery, msg.ConversationID).Scan(&msgCount)
		if err == nil && msgCount == 1 {
			// 使用消息内容的前50个字符作为标题
			title := msg.Content
			if len(title) > 50 {
				title = title[:50] + "..."
			}
			titleQuery := `UPDATE conversations SET title = $2 WHERE id = $1`
			tx.ExecContext(ctx, titleQuery, msg.ConversationID, title)
		}
	}

	return tx.Commit()
}

// GetMessages 获取会话消息
func (r *PostgresConversationRepository) GetMessages(ctx context.Context, convID string, opts ListOptions) ([]*Message, int, error) {
	// 设置默认值
	if opts.Limit == 0 {
		opts.Limit = 50
	}
	if opts.SortBy == "" {
		opts.SortBy = "created_at"
	}
	if opts.Order == "" {
		opts.Order = "asc"
	}

	// 计算总数
	countQuery := `
		SELECT COUNT(*) 
		FROM messages 
		WHERE conversation_id = $1
	`
	var total int
	err := r.db.QueryRowContext(ctx, countQuery, convID).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count messages: %w", err)
	}

	// 查询数据
	query := fmt.Sprintf(`
		SELECT 
			id, conversation_id, role, content, modality,
			token_count, metadata, references, created_at
		FROM messages
		WHERE conversation_id = $1
		ORDER BY %s %s
		LIMIT $2 OFFSET $3
	`, opts.SortBy, opts.Order)

	rows, err := r.db.QueryContext(ctx, query, convID, opts.Limit, opts.Offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list messages: %w", err)
	}
	defer rows.Close()

	var messages []*Message
	for rows.Next() {
		msg := &Message{}
		var metadataJSON, referencesJSON []byte

		err := rows.Scan(
			&msg.ID, &msg.ConversationID, &msg.Role, &msg.Content,
			&msg.Modality, &msg.TokenCount, &metadataJSON,
			&referencesJSON, &msg.CreatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan message: %w", err)
		}

		// 解析metadata
		if len(metadataJSON) > 0 {
			if err := json.Unmarshal(metadataJSON, &msg.Metadata); err != nil {
				return nil, 0, fmt.Errorf("failed to unmarshal metadata: %w", err)
			}
		}

		// 解析references
		if len(referencesJSON) > 0 {
			if err := json.Unmarshal(referencesJSON, &msg.References); err != nil {
				return nil, 0, fmt.Errorf("failed to unmarshal references: %w", err)
			}
		}

		messages = append(messages, msg)
	}

	return messages, total, nil
}

// GetLastMessages 获取最近的消息
func (r *PostgresConversationRepository) GetLastMessages(ctx context.Context, convID string, limit int) ([]*Message, error) {
	if limit == 0 {
		limit = 10
	}

	query := `
		SELECT 
			id, conversation_id, role, content, modality,
			token_count, metadata, references, created_at
		FROM messages
		WHERE conversation_id = $1
		ORDER BY created_at DESC
		LIMIT $2
	`

	rows, err := r.db.QueryContext(ctx, query, convID, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get last messages: %w", err)
	}
	defer rows.Close()

	var messages []*Message
	for rows.Next() {
		msg := &Message{}
		var metadataJSON, referencesJSON []byte

		err := rows.Scan(
			&msg.ID, &msg.ConversationID, &msg.Role, &msg.Content,
			&msg.Modality, &msg.TokenCount, &metadataJSON,
			&referencesJSON, &msg.CreatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan message: %w", err)
		}

		// 解析JSON字段
		if len(metadataJSON) > 0 {
			json.Unmarshal(metadataJSON, &msg.Metadata)
		}
		if len(referencesJSON) > 0 {
			json.Unmarshal(referencesJSON, &msg.References)
		}

		messages = append(messages, msg)
	}

	// 反转顺序（因为查询时是DESC）
	for i, j := 0, len(messages)-1; i < j; i, j = i+1, j-1 {
		messages[i], messages[j] = messages[j], messages[i]
	}

	return messages, nil
}

// UpdateMessageReferences 更新消息引用
func (r *PostgresConversationRepository) UpdateMessageReferences(ctx context.Context, msgID string, refs []Reference) error {
	referencesJSON, err := json.Marshal(refs)
	if err != nil {
		return fmt.Errorf("failed to marshal references: %w", err)
	}

	query := `
		UPDATE messages 
		SET references = $2
		WHERE id = $1
	`

	_, err = r.db.ExecContext(ctx, query, msgID, referencesJSON)
	if err != nil {
		return fmt.Errorf("failed to update message references: %w", err)
	}

	return nil
}

// UpdateStats 更新会话统计
func (r *PostgresConversationRepository) UpdateStats(ctx context.Context, convID string) error {
	query := `
		UPDATE conversations c
		SET 
			msg_count = (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id),
			token_count = (SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE conversation_id = c.id),
			updated_at = NOW()
		WHERE id = $1
	`

	_, err := r.db.ExecContext(ctx, query, convID)
	if err != nil {
		return fmt.Errorf("failed to update conversation stats: %w", err)
	}

	return nil
}

// GetUserStats 获取用户统计
func (r *PostgresConversationRepository) GetUserStats(ctx context.Context, userID string) (map[string]interface{}, error) {
	query := `
		SELECT 
			COUNT(DISTINCT c.id) as conversation_count,
			COUNT(m.id) as message_count,
			COALESCE(SUM(m.token_count), 0) as total_tokens,
			MAX(c.last_msg_at) as last_active
		FROM conversations c
		LEFT JOIN messages m ON c.id = m.conversation_id
		WHERE c.user_id = $1 AND c.deleted_at IS NULL
	`

	stats := make(map[string]interface{})
	var lastActive pq.NullTime

	err := r.db.QueryRowContext(ctx, query, userID).Scan(
		&stats["conversation_count"],
		&stats["message_count"],
		&stats["total_tokens"],
		&lastActive,
	)

	if err != nil {
		return nil, fmt.Errorf("failed to get user stats: %w", err)
	}

	if lastActive.Valid {
		stats["last_active"] = lastActive.Time
	}

	return stats, nil
}
