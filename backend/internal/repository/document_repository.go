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

// DocumentModel 文档模型
type DocumentModel struct {
	ID          string                 `json:"id"`
	DocumentID  string                 `json:"document_id"`
	TenantID    string                 `json:"tenant_id"`
	Title       string                 `json:"title"`
	Content     string                 `json:"content"`
	ContentType string                 `json:"content_type"`
	Source      string                 `json:"source"`
	Status      string                 `json:"status"` // active, archived, deleted
	Metadata    map[string]interface{} `json:"metadata"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// DocumentChunk 文档块
type DocumentChunk struct {
	ID         string                 `json:"id"`
	DocumentID string                 `json:"document_id"`
	ChunkID    string                 `json:"chunk_id"`
	Content    string                 `json:"content"`
	Position   int                    `json:"position"`
	Metadata   map[string]interface{} `json:"metadata"`
	CreatedAt  time.Time              `json:"created_at"`
}

// DocumentRepository 文档仓库接口
type DocumentRepository interface {
	Create(ctx context.Context, doc *DocumentModel) error
	GetByID(ctx context.Context, documentID string) (*DocumentModel, error)
	GetByTenant(ctx context.Context, tenantID string, limit, offset int) ([]*DocumentModel, error)
	Update(ctx context.Context, doc *DocumentModel) error
	Delete(ctx context.Context, documentID string) error
	Search(ctx context.Context, tenantID, query string, limit, offset int) ([]*DocumentModel, error)
	GetBySource(ctx context.Context, tenantID, source string, limit, offset int) ([]*DocumentModel, error)

	// 文档块操作
	CreateChunk(ctx context.Context, chunk *DocumentChunk) error
	GetChunks(ctx context.Context, documentID string) ([]*DocumentChunk, error)
	DeleteChunks(ctx context.Context, documentID string) error
}

// PostgresDocumentRepository PostgreSQL文档仓库实现
type PostgresDocumentRepository struct {
	db *sql.DB
}

// NewPostgresDocumentRepository 创建PostgreSQL文档仓库
func NewPostgresDocumentRepository(db *sql.DB) DocumentRepository {
	return &PostgresDocumentRepository{db: db}
}

// Create 创建文档
func (r *PostgresDocumentRepository) Create(ctx context.Context, doc *DocumentModel) error {
	if doc.ID == "" {
		doc.ID = uuid.New().String()
	}

	metadataJSON, err := json.Marshal(doc.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %v", err)
	}

	query := `
		INSERT INTO documents (id, document_id, tenant_id, title, content, content_type, source, status, metadata, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
	`

	_, err = r.db.ExecContext(ctx, query,
		doc.ID, doc.DocumentID, doc.TenantID, doc.Title, doc.Content,
		doc.ContentType, doc.Source, doc.Status, metadataJSON)
	if err != nil {
		return fmt.Errorf("failed to create document: %v", err)
	}

	logrus.WithFields(logrus.Fields{
		"document_id": doc.DocumentID,
		"tenant_id":   doc.TenantID,
	}).Info("Document created")

	return nil
}

// GetByID 根据ID获取文档
func (r *PostgresDocumentRepository) GetByID(ctx context.Context, documentID string) (*DocumentModel, error) {
	query := `
		SELECT id, document_id, tenant_id, title, content, content_type, source, status, metadata, created_at, updated_at
		FROM documents WHERE document_id = $1 AND status != 'deleted'
	`

	return r.scanDocument(ctx, query, documentID)
}

// GetByTenant 根据租户获取文档
func (r *PostgresDocumentRepository) GetByTenant(ctx context.Context, tenantID string, limit, offset int) ([]*DocumentModel, error) {
	query := `
		SELECT id, document_id, tenant_id, title, content, content_type, source, status, metadata, created_at, updated_at
		FROM documents 
		WHERE tenant_id = $1 AND status != 'deleted'
		ORDER BY created_at DESC
		LIMIT $2 OFFSET $3
	`

	rows, err := r.db.QueryContext(ctx, query, tenantID, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to get documents by tenant: %v", err)
	}
	defer rows.Close()

	var documents []*DocumentModel
	for rows.Next() {
		doc, err := r.scanDocumentRow(rows)
		if err != nil {
			return nil, err
		}
		documents = append(documents, doc)
	}

	return documents, nil
}

// Update 更新文档
func (r *PostgresDocumentRepository) Update(ctx context.Context, doc *DocumentModel) error {
	metadataJSON, err := json.Marshal(doc.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %v", err)
	}

	query := `
		UPDATE documents 
		SET title = $2, content = $3, content_type = $4, source = $5, status = $6, metadata = $7, updated_at = CURRENT_TIMESTAMP
		WHERE document_id = $1
	`

	result, err := r.db.ExecContext(ctx, query,
		doc.DocumentID, doc.Title, doc.Content, doc.ContentType,
		doc.Source, doc.Status, metadataJSON)
	if err != nil {
		return fmt.Errorf("failed to update document: %v", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %v", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("document not found: %s", doc.DocumentID)
	}

	logrus.WithField("document_id", doc.DocumentID).Info("Document updated")
	return nil
}

// Delete 删除文档
func (r *PostgresDocumentRepository) Delete(ctx context.Context, documentID string) error {
	query := `UPDATE documents SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE document_id = $1`

	result, err := r.db.ExecContext(ctx, query, documentID)
	if err != nil {
		return fmt.Errorf("failed to delete document: %v", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %v", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("document not found: %s", documentID)
	}

	// 同时删除文档块
	r.DeleteChunks(ctx, documentID)

	logrus.WithField("document_id", documentID).Info("Document deleted")
	return nil
}

// Search 搜索文档
func (r *PostgresDocumentRepository) Search(ctx context.Context, tenantID, query string, limit, offset int) ([]*DocumentModel, error) {
	searchQuery := `
		SELECT id, document_id, tenant_id, title, content, content_type, source, status, metadata, created_at, updated_at
		FROM documents 
		WHERE tenant_id = $1 AND status != 'deleted'
		AND (title ILIKE $2 OR content ILIKE $2)
		ORDER BY created_at DESC
		LIMIT $3 OFFSET $4
	`

	searchPattern := "%" + query + "%"
	rows, err := r.db.QueryContext(ctx, searchQuery, tenantID, searchPattern, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to search documents: %v", err)
	}
	defer rows.Close()

	var documents []*DocumentModel
	for rows.Next() {
		doc, err := r.scanDocumentRow(rows)
		if err != nil {
			return nil, err
		}
		documents = append(documents, doc)
	}

	return documents, nil
}

// GetBySource 根据来源获取文档
func (r *PostgresDocumentRepository) GetBySource(ctx context.Context, tenantID, source string, limit, offset int) ([]*DocumentModel, error) {
	query := `
		SELECT id, document_id, tenant_id, title, content, content_type, source, status, metadata, created_at, updated_at
		FROM documents 
		WHERE tenant_id = $1 AND source = $2 AND status != 'deleted'
		ORDER BY created_at DESC
		LIMIT $3 OFFSET $4
	`

	rows, err := r.db.QueryContext(ctx, query, tenantID, source, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("failed to get documents by source: %v", err)
	}
	defer rows.Close()

	var documents []*DocumentModel
	for rows.Next() {
		doc, err := r.scanDocumentRow(rows)
		if err != nil {
			return nil, err
		}
		documents = append(documents, doc)
	}

	return documents, nil
}

// CreateChunk 创建文档块
func (r *PostgresDocumentRepository) CreateChunk(ctx context.Context, chunk *DocumentChunk) error {
	if chunk.ID == "" {
		chunk.ID = uuid.New().String()
	}

	metadataJSON, err := json.Marshal(chunk.Metadata)
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %v", err)
	}

	query := `
		INSERT INTO document_chunks (id, document_id, chunk_id, content, position, metadata, created_at)
		VALUES ($1, $2, $3, $4, $5, $6, CURRENT_TIMESTAMP)
	`

	_, err = r.db.ExecContext(ctx, query,
		chunk.ID, chunk.DocumentID, chunk.ChunkID, chunk.Content,
		chunk.Position, metadataJSON)
	if err != nil {
		return fmt.Errorf("failed to create document chunk: %v", err)
	}

	return nil
}

// GetChunks 获取文档块
func (r *PostgresDocumentRepository) GetChunks(ctx context.Context, documentID string) ([]*DocumentChunk, error) {
	query := `
		SELECT id, document_id, chunk_id, content, position, metadata, created_at
		FROM document_chunks 
		WHERE document_id = $1
		ORDER BY position ASC
	`

	rows, err := r.db.QueryContext(ctx, query, documentID)
	if err != nil {
		return nil, fmt.Errorf("failed to get document chunks: %v", err)
	}
	defer rows.Close()

	var chunks []*DocumentChunk
	for rows.Next() {
		var chunk DocumentChunk
		var metadataJSON []byte

		err := rows.Scan(
			&chunk.ID, &chunk.DocumentID, &chunk.ChunkID,
			&chunk.Content, &chunk.Position, &metadataJSON, &chunk.CreatedAt,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan document chunk: %v", err)
		}

		if err := json.Unmarshal(metadataJSON, &chunk.Metadata); err != nil {
			chunk.Metadata = make(map[string]interface{})
		}

		chunks = append(chunks, &chunk)
	}

	return chunks, nil
}

// DeleteChunks 删除文档块
func (r *PostgresDocumentRepository) DeleteChunks(ctx context.Context, documentID string) error {
	query := `DELETE FROM document_chunks WHERE document_id = $1`

	_, err := r.db.ExecContext(ctx, query, documentID)
	if err != nil {
		return fmt.Errorf("failed to delete document chunks: %v", err)
	}

	return nil
}

// scanDocument 扫描单个文档
func (r *PostgresDocumentRepository) scanDocument(ctx context.Context, query string, args ...interface{}) (*DocumentModel, error) {
	row := r.db.QueryRowContext(ctx, query, args...)
	return r.scanDocumentRow(row)
}

// scanDocumentRow 扫描文档行
func (r *PostgresDocumentRepository) scanDocumentRow(scanner interface {
	Scan(dest ...interface{}) error
}) (*DocumentModel, error) {
	var doc DocumentModel
	var metadataJSON []byte

	err := scanner.Scan(
		&doc.ID, &doc.DocumentID, &doc.TenantID, &doc.Title, &doc.Content,
		&doc.ContentType, &doc.Source, &doc.Status, &metadataJSON,
		&doc.CreatedAt, &doc.UpdatedAt,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("document not found")
		}
		return nil, fmt.Errorf("failed to scan document: %v", err)
	}

	// 解析JSON字段
	if err := json.Unmarshal(metadataJSON, &doc.Metadata); err != nil {
		doc.Metadata = make(map[string]interface{})
	}

	return &doc, nil
}
