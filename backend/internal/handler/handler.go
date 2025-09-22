package handler

import (
	"voicehelper/backend/internal/service"
	"net/http"

	"github.com/gin-gonic/gin"
)

type Handlers struct {
	services *service.Services
}

func NewHandlers(services *service.Services) *Handlers {
	return &Handlers{
		services: services,
	}
}

// HealthCheck 健康检查
func (h *Handlers) HealthCheck(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"status":  "ok",
		"service": "chatbot-backend",
	})
}
