package handler

import (
	"chatbot/internal/service"
	"net/http"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// UploadFiles 文件上传接口
func (h *Handlers) UploadFiles(c *gin.Context) {
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

	datasetID := c.PostForm("dataset_id")
	if datasetID == "" {
		datasetID = "default"
	}

	// TODO: 保存文件到对象存储
	var filePaths []string
	for _, file := range files {
		// 这里应该保存到对象存储，暂时使用文件名
		filePaths = append(filePaths, file.Filename)
		logrus.Infof("Received file: %s, size: %d", file.Filename, file.Size)
	}

	// 调用算法服务入库
	req := &service.IngestRequest{
		DatasetID: datasetID,
		Files:     filePaths,
		ChunkSize: 600,
	}

	response, err := h.services.AlgoService.Ingest(c.Request.Context(), req)
	if err != nil {
		logrus.WithError(err).Error("Failed to ingest files")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process files"})
		return
	}

	c.JSON(http.StatusOK, response)
}

// IngestURL URL入库接口
func (h *Handlers) IngestURL(c *gin.Context) {
	var req struct {
		URLs      []string `json:"urls" binding:"required"`
		DatasetID string   `json:"dataset_id"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if req.DatasetID == "" {
		req.DatasetID = "default"
	}

	// 调用算法服务入库
	ingestReq := &service.IngestRequest{
		DatasetID: req.DatasetID,
		URLs:      req.URLs,
		ChunkSize: 600,
	}

	response, err := h.services.AlgoService.Ingest(c.Request.Context(), ingestReq)
	if err != nil {
		logrus.WithError(err).Error("Failed to ingest URLs")
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to process URLs"})
		return
	}

	c.JSON(http.StatusOK, response)
}

// GetTask 获取任务状态
func (h *Handlers) GetTask(c *gin.Context) {
	taskID := c.Param("id")
	if taskID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Task ID is required"})
		return
	}

	// TODO: 实现任务状态查询
	c.JSON(http.StatusOK, gin.H{
		"task_id":  taskID,
		"status":   "processing",
		"progress": 50,
	})
}
