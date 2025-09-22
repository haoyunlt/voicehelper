package main

import (
	"fmt"
	"log"
	"net/http"

	"github.com/gin-gonic/gin"
)

func main() {
	r := gin.Default()

	// 健康检查
	r.GET("/health", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"status":  "ok",
			"service": "voicehelper-backend",
		})
	})

	// 基础API
	r.GET("/api/v1/status", func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{
			"message": "VoiceHelper Backend is running",
			"version": "1.0.0",
		})
	})

	fmt.Println("Backend server starting on :8080")
	log.Fatal(r.Run(":8080"))
}
