package repository

import (
	"context"
	"database/sql"
	"fmt"
	"time"

	"github.com/google/uuid"
)

// Dataset 数据集模型
type Dataset struct {
	ID          string    `json:"id"`
	TenantID    string    `json:"tenant_id"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	Type        string    `json:"type"`   // document, qa, custom
	Status      string    `json:"status"` // active, inactive, processing
	DocCount    int       `json:"doc_count"`
	ChunkCount  int       `json:"chunk_count"`
	TokenCount  int64     `json:"token_count"`
	Metadata    string    `json:"metadata"` // JSON string
	CreatedBy   string    `json:"created_by"`
	UpdatedBy   string    `json:"updated_by"`
	CreatedAt   time.Time `json:"created_at"`
	UpdatedAt   time.Time `json:"updated_at"`
}

// Document 文档模型
type Document struct {
	ID         string    `json:"id"`
	DatasetID  string    `json:"dataset_id"`
	Name       string    `json:"name"`
	Source     string    `json:"source"` // file path or URL
	Type       string    `json:"type"`   // pdf, txt, docx, html, etc.
	Size       int64     `json:"size"`   // file size in bytes
	Status     string    `json:"status"` // pending, processing, completed, failed
	ChunkCount int       `json:"chunk_count"`
	TokenCount int64     `json:"token_count"`
	Metadata   string    `json:"metadata"` // JSON string
	CreatedAt  time.Time `json:"created_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// ListOptions 列表查询选项
type ListOptions struct {
	Offset int
	Limit  int
	SortBy string
	Order  string // asc or desc
}

// DatasetRepository 数据集仓库接口
type DatasetRepository interface {
	List(ctx context.Context, tenantID string, opts ListOptions) ([]*Dataset, int, error)
	Get(ctx context.Context, id string) (*Dataset, error)
	Create(ctx context.Context, dataset *Dataset) error
	Update(ctx context.Context, dataset *Dataset) error
	Delete(ctx context.Context, id string) error
	UpdateStats(ctx context.Context, id string, docCount, chunkCount int, tokenCount int64) error

	// Document related
	ListDocuments(ctx context.Context, datasetID string, opts ListOptions) ([]*Document, int, error)
	CreateDocument(ctx context.Context, doc *Document) error
	UpdateDocumentStatus(ctx context.Context, docID, status string) error
	DeleteDocument(ctx context.Context, docID string) error
}

// PostgresDatasetRepository PostgreSQL实现
type PostgresDatasetRepository struct {
	db *sql.DB
}

// NewPostgresDatasetRepository 创建PostgreSQL数据集仓库
func NewPostgresDatasetRepository(db *sql.DB) DatasetRepository {
	return &PostgresDatasetRepository{db: db}
}

// List 获取数据集列表
func (r *PostgresDatasetRepository) List(ctx context.Context, tenantID string, opts ListOptions) ([]*Dataset, int, error) {
	// 设置默认值
	if opts.Limit == 0 {
		opts.Limit = 20
	}
	if opts.SortBy == "" {
		opts.SortBy = "created_at"
	}
	if opts.Order == "" {
		opts.Order = "desc"
	}

	// 计算总数
	countQuery := `
		SELECT COUNT(*) 
		FROM datasets 
		WHERE tenant_id = $1 AND deleted_at IS NULL
	`
	var total int
	err := r.db.QueryRowContext(ctx, countQuery, tenantID).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count datasets: %w", err)
	}

	// 查询数据
	query := fmt.Sprintf(`
		SELECT 
			id, tenant_id, name, description, type, status,
			doc_count, chunk_count, token_count, metadata,
			created_by, updated_by, created_at, updated_at
		FROM datasets
		WHERE tenant_id = $1 AND deleted_at IS NULL
		ORDER BY %s %s
		LIMIT $2 OFFSET $3
	`, opts.SortBy, opts.Order)

	rows, err := r.db.QueryContext(ctx, query, tenantID, opts.Limit, opts.Offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list datasets: %w", err)
	}
	defer rows.Close()

	var datasets []*Dataset
	for rows.Next() {
		dataset := &Dataset{}
		err := rows.Scan(
			&dataset.ID, &dataset.TenantID, &dataset.Name, &dataset.Description,
			&dataset.Type, &dataset.Status, &dataset.DocCount, &dataset.ChunkCount,
			&dataset.TokenCount, &dataset.Metadata, &dataset.CreatedBy,
			&dataset.UpdatedBy, &dataset.CreatedAt, &dataset.UpdatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan dataset: %w", err)
		}
		datasets = append(datasets, dataset)
	}

	return datasets, total, nil
}

// Get 获取数据集详情
func (r *PostgresDatasetRepository) Get(ctx context.Context, id string) (*Dataset, error) {
	query := `
		SELECT 
			id, tenant_id, name, description, type, status,
			doc_count, chunk_count, token_count, metadata,
			created_by, updated_by, created_at, updated_at
		FROM datasets
		WHERE id = $1 AND deleted_at IS NULL
	`

	dataset := &Dataset{}
	err := r.db.QueryRowContext(ctx, query, id).Scan(
		&dataset.ID, &dataset.TenantID, &dataset.Name, &dataset.Description,
		&dataset.Type, &dataset.Status, &dataset.DocCount, &dataset.ChunkCount,
		&dataset.TokenCount, &dataset.Metadata, &dataset.CreatedBy,
		&dataset.UpdatedBy, &dataset.CreatedAt, &dataset.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("dataset not found")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get dataset: %w", err)
	}

	return dataset, nil
}

// Create 创建数据集
func (r *PostgresDatasetRepository) Create(ctx context.Context, dataset *Dataset) error {
	if dataset.ID == "" {
		dataset.ID = uuid.New().String()
	}
	if dataset.Type == "" {
		dataset.Type = "document"
	}
	if dataset.Status == "" {
		dataset.Status = "active"
	}
	if dataset.Metadata == "" {
		dataset.Metadata = "{}"
	}

	query := `
		INSERT INTO datasets (
			id, tenant_id, name, description, type, status,
			doc_count, chunk_count, token_count, metadata,
			created_by, updated_by, created_at, updated_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW()
		)
	`

	_, err := r.db.ExecContext(ctx, query,
		dataset.ID, dataset.TenantID, dataset.Name, dataset.Description,
		dataset.Type, dataset.Status, dataset.DocCount, dataset.ChunkCount,
		dataset.TokenCount, dataset.Metadata, dataset.CreatedBy, dataset.UpdatedBy,
	)

	if err != nil {
		return fmt.Errorf("failed to create dataset: %w", err)
	}

	return nil
}

// Update 更新数据集
func (r *PostgresDatasetRepository) Update(ctx context.Context, dataset *Dataset) error {
	query := `
		UPDATE datasets SET
			name = $2,
			description = $3,
			type = $4,
			status = $5,
			metadata = $6,
			updated_by = $7,
			updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query,
		dataset.ID, dataset.Name, dataset.Description,
		dataset.Type, dataset.Status, dataset.Metadata, dataset.UpdatedBy,
	)

	if err != nil {
		return fmt.Errorf("failed to update dataset: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("dataset not found")
	}

	return nil
}

// Delete 软删除数据集
func (r *PostgresDatasetRepository) Delete(ctx context.Context, id string) error {
	query := `
		UPDATE datasets 
		SET deleted_at = NOW(), updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query, id)
	if err != nil {
		return fmt.Errorf("failed to delete dataset: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("dataset not found")
	}

	return nil
}

// UpdateStats 更新数据集统计信息
func (r *PostgresDatasetRepository) UpdateStats(ctx context.Context, id string, docCount, chunkCount int, tokenCount int64) error {
	query := `
		UPDATE datasets SET
			doc_count = $2,
			chunk_count = $3,
			token_count = $4,
			updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	_, err := r.db.ExecContext(ctx, query, id, docCount, chunkCount, tokenCount)
	if err != nil {
		return fmt.Errorf("failed to update dataset stats: %w", err)
	}

	return nil
}

// ListDocuments 获取文档列表
func (r *PostgresDatasetRepository) ListDocuments(ctx context.Context, datasetID string, opts ListOptions) ([]*Document, int, error) {
	// 设置默认值
	if opts.Limit == 0 {
		opts.Limit = 20
	}
	if opts.SortBy == "" {
		opts.SortBy = "created_at"
	}
	if opts.Order == "" {
		opts.Order = "desc"
	}

	// 计算总数
	countQuery := `
		SELECT COUNT(*) 
		FROM documents 
		WHERE dataset_id = $1 AND deleted_at IS NULL
	`
	var total int
	err := r.db.QueryRowContext(ctx, countQuery, datasetID).Scan(&total)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to count documents: %w", err)
	}

	// 查询数据
	query := fmt.Sprintf(`
		SELECT 
			id, dataset_id, name, source, type, size, status,
			chunk_count, token_count, metadata, created_at, updated_at
		FROM documents
		WHERE dataset_id = $1 AND deleted_at IS NULL
		ORDER BY %s %s
		LIMIT $2 OFFSET $3
	`, opts.SortBy, opts.Order)

	rows, err := r.db.QueryContext(ctx, query, datasetID, opts.Limit, opts.Offset)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to list documents: %w", err)
	}
	defer rows.Close()

	var documents []*Document
	for rows.Next() {
		doc := &Document{}
		err := rows.Scan(
			&doc.ID, &doc.DatasetID, &doc.Name, &doc.Source,
			&doc.Type, &doc.Size, &doc.Status, &doc.ChunkCount,
			&doc.TokenCount, &doc.Metadata, &doc.CreatedAt, &doc.UpdatedAt,
		)
		if err != nil {
			return nil, 0, fmt.Errorf("failed to scan document: %w", err)
		}
		documents = append(documents, doc)
	}

	return documents, total, nil
}

// CreateDocument 创建文档
func (r *PostgresDatasetRepository) CreateDocument(ctx context.Context, doc *Document) error {
	if doc.ID == "" {
		doc.ID = uuid.New().String()
	}
	if doc.Status == "" {
		doc.Status = "pending"
	}
	if doc.Metadata == "" {
		doc.Metadata = "{}"
	}

	query := `
		INSERT INTO documents (
			id, dataset_id, name, source, type, size, status,
			chunk_count, token_count, metadata, created_at, updated_at
		) VALUES (
			$1, $2, $3, $4, $5, $6, $7, $8, $9, $10, NOW(), NOW()
		)
	`

	_, err := r.db.ExecContext(ctx, query,
		doc.ID, doc.DatasetID, doc.Name, doc.Source,
		doc.Type, doc.Size, doc.Status, doc.ChunkCount,
		doc.TokenCount, doc.Metadata,
	)

	if err != nil {
		return fmt.Errorf("failed to create document: %w", err)
	}

	return nil
}

// UpdateDocumentStatus 更新文档状态
func (r *PostgresDatasetRepository) UpdateDocumentStatus(ctx context.Context, docID, status string) error {
	query := `
		UPDATE documents SET
			status = $2,
			updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	_, err := r.db.ExecContext(ctx, query, docID, status)
	if err != nil {
		return fmt.Errorf("failed to update document status: %w", err)
	}

	return nil
}

// GetDocument 获取文档详情
func (r *PostgresDatasetRepository) GetDocument(ctx context.Context, docID string) (*Document, error) {
	query := `
		SELECT 
			id, dataset_id, name, source, type, size, status,
			chunk_count, token_count, metadata, created_at, updated_at
		FROM documents
		WHERE id = $1 AND deleted_at IS NULL
	`

	doc := &Document{}
	err := r.db.QueryRowContext(ctx, query, docID).Scan(
		&doc.ID, &doc.DatasetID, &doc.Name, &doc.Source,
		&doc.Type, &doc.Size, &doc.Status, &doc.ChunkCount,
		&doc.TokenCount, &doc.Metadata, &doc.CreatedAt, &doc.UpdatedAt,
	)

	if err != nil {
		if err == sql.ErrNoRows {
			return nil, fmt.Errorf("document not found")
		}
		return nil, fmt.Errorf("failed to get document: %w", err)
	}

	return doc, nil
}

// UpdateDocument 更新文档信息
func (r *PostgresDatasetRepository) UpdateDocument(ctx context.Context, doc *Document) error {
	query := `
		UPDATE documents SET
			name = $2,
			source = $3,
			type = $4,
			size = $5,
			status = $6,
			chunk_count = $7,
			token_count = $8,
			metadata = $9,
			updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query,
		doc.ID, doc.Name, doc.Source, doc.Type, doc.Size,
		doc.Status, doc.ChunkCount, doc.TokenCount, doc.Metadata,
	)

	if err != nil {
		return fmt.Errorf("failed to update document: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("document not found")
	}

	return nil
}

// DeleteDocument 软删除文档
func (r *PostgresDatasetRepository) DeleteDocument(ctx context.Context, docID string) error {
	query := `
		UPDATE documents 
		SET deleted_at = NOW(), updated_at = NOW()
		WHERE id = $1 AND deleted_at IS NULL
	`

	result, err := r.db.ExecContext(ctx, query, docID)
	if err != nil {
		return fmt.Errorf("failed to delete document: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}

	if rowsAffected == 0 {
		return fmt.Errorf("document not found")
	}

	return nil
}
