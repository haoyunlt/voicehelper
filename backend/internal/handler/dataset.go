package handler

import (
	"chatbot/internal/repository"
	"chatbot/internal/service"
	"chatbot/pkg/storage"
	"context"
	"fmt"
	"io"
	"net/http"
	"path/filepath"
	"strconv"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// DatasetHandler 数据集处理器（使用真实数据库）
type DatasetHandler struct {
	datasetRepo repository.DatasetRepository
	storage     storage.ObjectStorage
	algoService service.AlgoService
}

// NewDatasetHandler 创建数据集处理器
func NewDatasetHandler(
	datasetRepo repository.DatasetRepository,
	storage storage.ObjectStorage,
	algoService service.AlgoService,
) *DatasetHandler {
	return &DatasetHandler{
		datasetRepo: datasetRepo,
		storage:     storage,
		algoService: algoService,
	}
}

// ListDatasets 获取数据集列表
func (h *DatasetHandler) ListDatasets(c *gin.Context) {
	tenantID := c.GetString("tenant_id")
	if tenantID == "" {
		tenantID = "default"
	}

	// 解析分页参数
	page, _ := strconv.Atoi(c.DefaultQuery("page", "1"))
	pageSize, _ := strconv.Atoi(c.DefaultQuery("page_size", "20"))
	sortBy := c.DefaultQuery("sort_by", "created_at")
	order := c.DefaultQuery("order", "desc")

	// 计算offset
	offset := (page - 1) * pageSize

	// 查询数据集
	opts := repository.ListOptions{
		Offset: offset,
		Limit:  pageSize,
		SortBy: sortBy,
		Order:  order,
	}

	datasets, total, err := h.datasetRepo.List(c.Request.Context(), tenantID, opts)
	if err != nil {
		logrus.WithError(err).Error("Failed to list datasets")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch datasets"})
		return
	}

	// 返回响应
	c.JSON(http.StatusOK, gin.H{
		"datasets":    datasets,
		"total":       total,
		"page":        page,
		"page_size":   pageSize,
		"total_pages": (total + pageSize - 1) / pageSize,
	})
}

// GetDataset 获取数据集详情
func (h *DatasetHandler) GetDataset(c *gin.Context) {
	datasetID := c.Param("id")
	if datasetID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Dataset ID is required"})
		return
	}

	// 获取数据集
	dataset, err := h.datasetRepo.Get(c.Request.Context(), datasetID)
	if err != nil {
		logrus.WithError(err).Error("Failed to get dataset")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to fetch dataset"})
		return
	}

	// 获取文档列表
	opts := repository.ListOptions{
		Limit: 100,
	}
	documents, _, err := h.datasetRepo.ListDocuments(c.Request.Context(), datasetID, opts)
	if err != nil {
		logrus.WithError(err).Error("Failed to list documents")
		// 不影响主流程，继续返回数据集信息
		documents = []*repository.Document{}
	}

	// 返回响应
	c.JSON(http.StatusOK, gin.H{
		"id":          dataset.ID,
		"name":        dataset.Name,
		"description": dataset.Description,
		"type":        dataset.Type,
		"status":      dataset.Status,
		"doc_count":   dataset.DocCount,
		"chunk_count": dataset.ChunkCount,
		"token_count": dataset.TokenCount,
		"metadata":    dataset.Metadata,
		"documents":   documents,
		"created_at":  dataset.CreatedAt,
		"updated_at":  dataset.UpdatedAt,
	})
}

// CreateDataset 创建数据集
func (h *DatasetHandler) CreateDataset(c *gin.Context) {
	tenantID := c.GetString("tenant_id")
	userID := c.GetString("user_id")

	var req struct {
		Name        string                 `json:"name" binding:"required"`
		Description string                 `json:"description"`
		Type        string                 `json:"type"`
		Metadata    map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// 创建数据集
	dataset := &repository.Dataset{
		TenantID:    tenantID,
		Name:        req.Name,
		Description: req.Description,
		Type:        req.Type,
		Status:      "active",
		Metadata:    req.Metadata,
		CreatedBy:   userID,
		UpdatedBy:   userID,
	}

	if dataset.Type == "" {
		dataset.Type = "document"
	}

	if err := h.datasetRepo.Create(c.Request.Context(), dataset); err != nil {
		logrus.WithError(err).Error("Failed to create dataset")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to create dataset"})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"id":      dataset.ID,
		"message": "Dataset created successfully",
	})
}

// UpdateDataset 更新数据集
func (h *DatasetHandler) UpdateDataset(c *gin.Context) {
	datasetID := c.Param("id")
	userID := c.GetString("user_id")

	var req struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Status      string                 `json:"status"`
		Metadata    map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// 获取数据集
	dataset, err := h.datasetRepo.Get(c.Request.Context(), datasetID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Dataset not found"})
		return
	}

	// 更新字段
	if req.Name != "" {
		dataset.Name = req.Name
	}
	if req.Description != "" {
		dataset.Description = req.Description
	}
	if req.Status != "" {
		dataset.Status = req.Status
	}
	if req.Metadata != nil {
		dataset.Metadata = req.Metadata
	}
	dataset.UpdatedBy = userID

	// 保存更新
	if err := h.datasetRepo.Update(c.Request.Context(), dataset); err != nil {
		logrus.WithError(err).Error("Failed to update dataset")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to update dataset"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"message": "Dataset updated successfully",
	})
}

// DeleteDataset 删除数据集
func (h *DatasetHandler) DeleteDataset(c *gin.Context) {
	datasetID := c.Param("id")

	if err := h.datasetRepo.Delete(c.Request.Context(), datasetID); err != nil {
		logrus.WithError(err).Error("Failed to delete dataset")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to delete dataset"})
		return
	}

	// TODO: 异步删除相关的向量数据和文件

	c.JSON(http.StatusOK, gin.H{
		"message": "Dataset deleted successfully",
	})
}

// UploadFiles 上传文件到数据集
func (h *DatasetHandler) UploadFiles(c *gin.Context) {
	datasetID := c.PostForm("dataset_id")
	if datasetID == "" {
		datasetID = "default"
	}

	// 解析multipart表单
	form, err := c.MultipartForm()
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to parse form"})
		return
	}

	files := form.File["files"]
	if len(files) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No files provided"})
		return
	}

	// 确保数据集存在
	dataset, err := h.datasetRepo.Get(c.Request.Context(), datasetID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "Dataset not found"})
		return
	}

	var uploadedFiles []gin.H
	var filePaths []string

	// 处理每个文件
	for _, file := range files {
		// 打开文件
		src, err := file.Open()
		if err != nil {
			logrus.WithError(err).Errorf("Failed to open file: %s", file.Filename)
			continue
		}
		defer src.Close()

		// 生成对象键
		objectKey := storage.GenerateObjectKey(
			fmt.Sprintf("datasets/%s", datasetID),
			file.Filename,
		)

		// 上传到MinIO
		uploadResult, err := h.storage.Upload(
			c.Request.Context(),
			objectKey,
			src,
			file.Size,
			file.Header.Get("Content-Type"),
		)
		if err != nil {
			logrus.WithError(err).Errorf("Failed to upload file: %s", file.Filename)
			continue
		}

		// 创建文档记录
		doc := &repository.Document{
			DatasetID: datasetID,
			Name:      file.Filename,
			Source:    uploadResult.URL,
			Type:      filepath.Ext(file.Filename),
			Size:      file.Size,
			Status:    "pending",
			Metadata: map[string]interface{}{
				"object_key": objectKey,
				"etag":       uploadResult.ETag,
			},
		}

		if err := h.datasetRepo.CreateDocument(c.Request.Context(), doc); err != nil {
			logrus.WithError(err).Error("Failed to create document record")
			// 删除已上传的文件
			h.storage.Delete(c.Request.Context(), objectKey)
			continue
		}

		uploadedFiles = append(uploadedFiles, gin.H{
			"id":       doc.ID,
			"filename": file.Filename,
			"size":     file.Size,
			"url":      uploadResult.URL,
			"status":   "uploaded",
		})

		filePaths = append(filePaths, uploadResult.URL)
	}

	// 异步调用算法服务处理文件
	if len(filePaths) > 0 {
		go h.processFiles(datasetID, filePaths)
	}

	// 更新数据集统计
	h.datasetRepo.UpdateStats(
		c.Request.Context(),
		datasetID,
		dataset.DocCount+len(uploadedFiles),
		dataset.ChunkCount,
		dataset.TokenCount,
	)

	c.JSON(http.StatusOK, gin.H{
		"message": "Files uploaded successfully",
		"files":   uploadedFiles,
		"total":   len(uploadedFiles),
	})
}

// processFiles 异步处理文件
func (h *DatasetHandler) processFiles(datasetID string, filePaths []string) {
	// 调用算法服务入库
	req := &service.IngestRequest{
		DatasetID:    datasetID,
		Files:        filePaths,
		ChunkSize:    600,
		ChunkOverlap: 100,
	}

	ctx := context.Background()
	response, err := h.algoService.Ingest(ctx, req)
	if err != nil {
		logrus.WithError(err).Error("Failed to ingest files")
		// TODO: 更新文档状态为失败
		return
	}

	logrus.Infof("Files ingested successfully: %v", response)
	// TODO: 更新文档状态为完成
}

// Search 搜索预览
func (h *DatasetHandler) Search(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Query parameter 'q' is required"})
		return
	}

	datasetID := c.Query("dataset_id")
	if datasetID == "" {
		datasetID = "default"
	}

	topK, _ := strconv.Atoi(c.DefaultQuery("top_k", "5"))

	// 调用算法服务搜索
	req := &service.SearchRequest{
		DatasetID: datasetID,
		Query:     query,
		TopK:      topK,
	}

	results, err := h.algoService.Search(c.Request.Context(), req)
	if err != nil {
		logrus.WithError(err).Error("Failed to search")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Search failed"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"query":   query,
		"results": results.Results,
		"total":   len(results.Results),
	})
}

// GetDocument 获取文档详情
func (h *DatasetHandler) GetDocument(c *gin.Context) {
	docID := c.Param("doc_id")

	// TODO: 实现文档详情查询
	// 这里需要添加文档repository的GetDocument方法

	c.JSON(http.StatusOK, gin.H{
		"id":     docID,
		"name":   "document.pdf",
		"status": "completed",
		"chunks": []gin.H{},
	})
}

// DeleteDocument 删除文档
func (h *DatasetHandler) DeleteDocument(c *gin.Context) {
	datasetID := c.Param("id")
	docID := c.Param("doc_id")

	// 删除文档记录
	if err := h.datasetRepo.DeleteDocument(c.Request.Context(), docID); err != nil {
		logrus.WithError(err).Error("Failed to delete document")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to delete document"})
		return
	}

	// TODO: 异步删除相关的向量数据和文件

	// 更新数据集统计
	go func() {
		ctx := context.Background()
		h.datasetRepo.UpdateStats(ctx, datasetID)
	}()

	c.JSON(http.StatusOK, gin.H{
		"message": "Document deleted successfully",
	})
}

// GetPresignedUploadURL 获取预签名上传URL
func (h *DatasetHandler) GetPresignedUploadURL(c *gin.Context) {
	datasetID := c.Query("dataset_id")
	filename := c.Query("filename")

	if datasetID == "" || filename == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "dataset_id and filename are required"})
		return
	}

	// 生成对象键
	objectKey := storage.GenerateObjectKey(
		fmt.Sprintf("datasets/%s", datasetID),
		filename,
	)

	// 获取预签名URL
	url, err := h.storage.GetPresignedPutURL(
		c.Request.Context(),
		objectKey,
		time.Hour,
	)
	if err != nil {
		logrus.WithError(err).Error("Failed to generate presigned URL")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate upload URL"})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"url":        url,
		"object_key": objectKey,
		"expires_in": 3600,
	})
}

// DownloadDocument 下载文档
func (h *DatasetHandler) DownloadDocument(c *gin.Context) {
	docID := c.Param("doc_id")

	// TODO: 从数据库获取文档信息
	// 这里假设已经获取到了object_key
	objectKey := fmt.Sprintf("documents/%s", docID)

	// 从存储下载
	reader, err := h.storage.Download(c.Request.Context(), objectKey)
	if err != nil {
		logrus.WithError(err).Error("Failed to download document")
		c.JSON(http.StatusNotFound, gin.H{"error": "Document not found"})
		return
	}
	defer reader.Close()

	// 设置响应头
	c.Header("Content-Type", "application/octet-stream")
	c.Header("Content-Disposition", fmt.Sprintf("attachment; filename=\"%s\"", "document.pdf"))

	// 流式传输文件
	io.Copy(c.Writer, reader)
}
