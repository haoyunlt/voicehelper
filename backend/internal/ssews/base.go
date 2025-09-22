package ssews

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"

	"github.com/gin-gonic/gin"
)

type BaseHandler struct {
	AlgoServiceURL string
}

func (h *BaseHandler) extractTraceInfo(c *gin.Context) (traceID, tenantID string) {
	traceID = c.GetHeader("X-Trace-ID")
	if traceID == "" {
		traceID = h.generateTraceID()
	}

	tenantID = c.GetHeader("X-Tenant-ID")
	if tenantID == "" {
		tenantID = "default"
	}

	return
}

func (h *BaseHandler) generateTraceID() string {
	bytes := make([]byte, 8)
	rand.Read(bytes)
	return fmt.Sprintf("trace_%s", hex.EncodeToString(bytes))
}

func (h *BaseHandler) createStreamHandler(writer StreamWriter, traceID, tenantID string) *BaseStreamHandler {
	return &BaseStreamHandler{
		Writer:   writer,
		TraceID:  traceID,
		TenantID: tenantID,
	}
}
