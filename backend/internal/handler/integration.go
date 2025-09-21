package handler

import (
	"chatbot/pkg/integration"
	"net/http"
	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/sirupsen/logrus"
)

// IntegrationHandler handles third-party integration requests
type IntegrationHandler struct {
	manager *integration.IntegrationManager
	logger  *logrus.Logger
}

// NewIntegrationHandler creates a new integration handler
func NewIntegrationHandler(manager *integration.IntegrationManager) *IntegrationHandler {
	return &IntegrationHandler{
		manager: manager,
		logger:  logrus.New(),
	}
}

// RegisterRoutes registers integration routes
func (h *IntegrationHandler) RegisterRoutes(r *gin.RouterGroup) {
	integrations := r.Group("/integrations")
	{
		// Service management
		integrations.GET("/services", h.ListServices)
		integrations.POST("/services", h.RegisterService)
		integrations.GET("/services/:id", h.GetService)
		integrations.PUT("/services/:id", h.UpdateService)
		integrations.DELETE("/services/:id", h.RemoveService)
		integrations.GET("/services/category/:category", h.ListServicesByCategory)
		integrations.GET("/services/search", h.SearchServices)

		// Service operations
		integrations.POST("/services/:id/call", h.CallService)
		integrations.POST("/services/:id/test", h.TestService)
		integrations.GET("/services/:id/endpoints", h.GetServiceEndpoints)
		integrations.POST("/services/batch", h.BatchCall)

		// Health and monitoring
		integrations.GET("/health", h.GetServicesHealth)
		integrations.GET("/stats", h.GetIntegrationStats)
		integrations.GET("/metrics", h.GetConnectorMetrics)

		// Workflows
		integrations.GET("/workflows", h.ListWorkflows)
		integrations.POST("/workflows", h.CreateWorkflow)
		integrations.GET("/workflows/:id", h.GetWorkflow)
		integrations.POST("/workflows/:id/execute", h.ExecuteWorkflow)

		// Configuration
		integrations.GET("/config/export", h.ExportConfiguration)
		integrations.POST("/config/import", h.ImportConfiguration)
	}
}

// ListServices lists all registered services
func (h *IntegrationHandler) ListServices(c *gin.Context) {
	services := h.manager.GetServiceRegistry().ListServices()

	c.JSON(http.StatusOK, gin.H{
		"services": services,
		"total":    len(services),
	})
}

// RegisterService registers a new third-party service
func (h *IntegrationHandler) RegisterService(c *gin.Context) {
	var service integration.ServiceConfig
	if err := c.ShouldBindJSON(&service); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := h.manager.GetServiceRegistry().RegisterService(&service); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"message": "Service registered successfully",
		"service": service,
	})
}

// GetService gets a service by ID
func (h *IntegrationHandler) GetService(c *gin.Context) {
	serviceID := c.Param("id")

	service, err := h.manager.GetServiceRegistry().GetService(serviceID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"service": service})
}

// UpdateService updates an existing service
func (h *IntegrationHandler) UpdateService(c *gin.Context) {
	serviceID := c.Param("id")

	var updates map[string]interface{}
	if err := c.ShouldBindJSON(&updates); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := h.manager.GetServiceRegistry().UpdateService(serviceID, updates); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Service updated successfully"})
}

// RemoveService removes a service
func (h *IntegrationHandler) RemoveService(c *gin.Context) {
	serviceID := c.Param("id")

	if err := h.manager.GetServiceRegistry().RemoveService(serviceID); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Service removed successfully"})
}

// ListServicesByCategory lists services by category
func (h *IntegrationHandler) ListServicesByCategory(c *gin.Context) {
	category := integration.ServiceCategory(c.Param("category"))

	services := h.manager.GetServiceRegistry().ListServicesByCategory(category)

	c.JSON(http.StatusOK, gin.H{
		"services": services,
		"category": category,
		"total":    len(services),
	})
}

// SearchServices searches for services
func (h *IntegrationHandler) SearchServices(c *gin.Context) {
	query := c.Query("q")
	if query == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Query parameter 'q' is required"})
		return
	}

	services := h.manager.GetServiceRegistry().SearchServices(query)

	c.JSON(http.StatusOK, gin.H{
		"services": services,
		"query":    query,
		"total":    len(services),
	})
}

// CallService makes a call to a third-party service
func (h *IntegrationHandler) CallService(c *gin.Context) {
	serviceID := c.Param("id")

	var options integration.RequestOptions
	if err := c.ShouldBindJSON(&options); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	response, err := h.manager.CallService(c.Request.Context(), serviceID, options)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"response":   response,
		"service_id": serviceID,
	})
}

// TestService tests connectivity to a service
func (h *IntegrationHandler) TestService(c *gin.Context) {
	serviceID := c.Param("id")

	err := h.manager.GetServiceConnector().TestConnection(c.Request.Context(), serviceID)

	if err != nil {
		c.JSON(http.StatusOK, gin.H{
			"service_id": serviceID,
			"healthy":    false,
			"error":      err.Error(),
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"service_id": serviceID,
		"healthy":    true,
	})
}

// GetServiceEndpoints gets available endpoints for a service
func (h *IntegrationHandler) GetServiceEndpoints(c *gin.Context) {
	serviceID := c.Param("id")

	endpoints, err := h.manager.GetServiceConnector().GetServiceEndpoints(serviceID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"service_id": serviceID,
		"endpoints":  endpoints,
		"total":      len(endpoints),
	})
}

// BatchCall makes batch calls to multiple services
func (h *IntegrationHandler) BatchCall(c *gin.Context) {
	var requests []integration.BatchRequest
	if err := c.ShouldBindJSON(&requests); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	responses := h.manager.GetServiceConnector().MakeBatchRequests(c.Request.Context(), requests)

	c.JSON(http.StatusOK, gin.H{
		"responses": responses,
		"total":     len(responses),
	})
}

// GetServicesHealth gets health status of all services
func (h *IntegrationHandler) GetServicesHealth(c *gin.Context) {
	health := h.manager.GetServiceConnector().GetServiceHealth(c.Request.Context())

	healthyCount := 0
	for _, healthy := range health {
		if healthy {
			healthyCount++
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"health":          health,
		"total_services":  len(health),
		"healthy_count":   healthyCount,
		"unhealthy_count": len(health) - healthyCount,
	})
}

// GetIntegrationStats gets integration statistics
func (h *IntegrationHandler) GetIntegrationStats(c *gin.Context) {
	stats := h.manager.GetIntegrationStats()
	c.JSON(http.StatusOK, stats)
}

// GetConnectorMetrics gets connector metrics
func (h *IntegrationHandler) GetConnectorMetrics(c *gin.Context) {
	metrics := h.manager.GetServiceConnector().GetMetrics()
	c.JSON(http.StatusOK, metrics)
}

// ListWorkflows lists all workflows
func (h *IntegrationHandler) ListWorkflows(c *gin.Context) {
	workflows := h.manager.ListWorkflows()

	c.JSON(http.StatusOK, gin.H{
		"workflows": workflows,
		"total":     len(workflows),
	})
}

// CreateWorkflow creates a new workflow
func (h *IntegrationHandler) CreateWorkflow(c *gin.Context) {
	var workflow integration.Workflow
	if err := c.ShouldBindJSON(&workflow); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := h.manager.CreateWorkflow(&workflow); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusCreated, gin.H{
		"message":  "Workflow created successfully",
		"workflow": workflow,
	})
}

// GetWorkflow gets a workflow by ID
func (h *IntegrationHandler) GetWorkflow(c *gin.Context) {
	workflowID := c.Param("id")

	workflow, err := h.manager.GetWorkflow(workflowID)
	if err != nil {
		c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"workflow": workflow})
}

// ExecuteWorkflow executes a workflow
func (h *IntegrationHandler) ExecuteWorkflow(c *gin.Context) {
	workflowID := c.Param("id")

	var request struct {
		Variables map[string]interface{} `json:"variables"`
	}

	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	execution, err := h.manager.ExecuteWorkflow(c.Request.Context(), workflowID, request.Variables)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{
			"error":     err.Error(),
			"execution": execution,
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"execution": execution,
		"message":   "Workflow executed successfully",
	})
}

// ExportConfiguration exports integration configuration
func (h *IntegrationHandler) ExportConfiguration(c *gin.Context) {
	config, err := h.manager.ExportConfiguration()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Set appropriate headers for file download
	c.Header("Content-Type", "application/json")
	c.Header("Content-Disposition", "attachment; filename=integration_config.json")

	c.JSON(http.StatusOK, config)
}

// ImportConfiguration imports integration configuration
func (h *IntegrationHandler) ImportConfiguration(c *gin.Context) {
	var config map[string]interface{}
	if err := c.ShouldBindJSON(&config); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	if err := h.manager.ImportConfiguration(config); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, gin.H{"message": "Configuration imported successfully"})
}

// ServiceDiscoveryRequest represents a request to discover services
type ServiceDiscoveryRequest struct {
	URL        string            `json:"url"`
	AuthType   string            `json:"auth_type,omitempty"`
	AuthConfig map[string]string `json:"auth_config,omitempty"`
	Timeout    int               `json:"timeout,omitempty"`
}

// DiscoverService attempts to discover service capabilities
func (h *IntegrationHandler) DiscoverService(c *gin.Context) {
	var request ServiceDiscoveryRequest
	if err := c.ShouldBindJSON(&request); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// This would implement service discovery logic
	// For now, return a placeholder response
	c.JSON(http.StatusOK, gin.H{
		"message": "Service discovery not yet implemented",
		"url":     request.URL,
	})
}

// GetServiceCategories returns available service categories
func (h *IntegrationHandler) GetServiceCategories(c *gin.Context) {
	categories := []string{
		string(integration.CategoryAPI),
		string(integration.CategoryDatabase),
		string(integration.CategoryMessaging),
		string(integration.CategoryPayment),
		string(integration.CategoryAnalytics),
		string(integration.CategoryStorage),
		string(integration.CategoryAuth),
		string(integration.CategoryML),
		string(integration.CategorySocial),
		string(integration.CategoryEcommerce),
	}

	c.JSON(http.StatusOK, gin.H{
		"categories": categories,
		"total":      len(categories),
	})
}

// GetServiceTemplates returns service configuration templates
func (h *IntegrationHandler) GetServiceTemplates(c *gin.Context) {
	templates := map[string]interface{}{
		"rest_api": map[string]interface{}{
			"name":        "REST API Service",
			"description": "Generic REST API service template",
			"category":    "api",
			"auth_type":   "api_key",
			"timeout":     30,
			"endpoints": []map[string]interface{}{
				{
					"name":        "get_data",
					"path":        "/data",
					"method":      "GET",
					"description": "Get data from the service",
				},
			},
		},
		"webhook": map[string]interface{}{
			"name":        "Webhook Service",
			"description": "Webhook-based service template",
			"category":    "messaging",
			"auth_type":   "none",
			"timeout":     10,
		},
		"oauth2_api": map[string]interface{}{
			"name":        "OAuth2 API Service",
			"description": "OAuth2 authenticated API service template",
			"category":    "api",
			"auth_type":   "oauth2",
			"timeout":     30,
		},
	}

	c.JSON(http.StatusOK, gin.H{
		"templates": templates,
		"total":     len(templates),
	})
}

// ValidateServiceConfig validates a service configuration
func (h *IntegrationHandler) ValidateServiceConfig(c *gin.Context) {
	var service integration.ServiceConfig
	if err := c.ShouldBindJSON(&service); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Perform validation
	errors := []string{}

	if service.ID == "" {
		errors = append(errors, "Service ID is required")
	}
	if service.Name == "" {
		errors = append(errors, "Service name is required")
	}
	if service.BaseURL == "" {
		errors = append(errors, "Base URL is required")
	}

	if len(errors) > 0 {
		c.JSON(http.StatusBadRequest, gin.H{
			"valid":  false,
			"errors": errors,
		})
		return
	}

	c.JSON(http.StatusOK, gin.H{
		"valid":   true,
		"message": "Service configuration is valid",
	})
}

// GetPopularServices returns a list of popular third-party services
func (h *IntegrationHandler) GetPopularServices(c *gin.Context) {
	limitStr := c.DefaultQuery("limit", "20")
	limit, err := strconv.Atoi(limitStr)
	if err != nil {
		limit = 20
	}

	// Get all services and return the first 'limit' number
	allServices := h.manager.GetServiceRegistry().ListServices()

	var popularServices []*integration.ServiceConfig
	for i, service := range allServices {
		if i >= limit {
			break
		}
		popularServices = append(popularServices, service)
	}

	c.JSON(http.StatusOK, gin.H{
		"services": popularServices,
		"total":    len(popularServices),
		"limit":    limit,
	})
}
