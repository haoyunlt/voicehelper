package integration

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"
)

// IntegrationManager provides high-level management of third-party integrations
type IntegrationManager struct {
	registry  *ServiceRegistry
	connector *ServiceConnector
	workflows map[string]*Workflow
	mu        sync.RWMutex
	logger    *logrus.Logger
}

// Workflow represents a sequence of service calls
type Workflow struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Steps       []WorkflowStep         `json:"steps"`
	Variables   map[string]interface{} `json:"variables"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// WorkflowStep represents a single step in a workflow
type WorkflowStep struct {
	ID            string                 `json:"id"`
	Name          string                 `json:"name"`
	ServiceID     string                 `json:"service_id"`
	Options       RequestOptions         `json:"options"`
	Condition     string                 `json:"condition,omitempty"`
	OnSuccess     string                 `json:"on_success,omitempty"`
	OnError       string                 `json:"on_error,omitempty"`
	OutputMapping map[string]string      `json:"output_mapping,omitempty"`
	Variables     map[string]interface{} `json:"variables,omitempty"`
}

// WorkflowExecution represents the execution state of a workflow
type WorkflowExecution struct {
	ID          string                 `json:"id"`
	WorkflowID  string                 `json:"workflow_id"`
	Status      string                 `json:"status"`
	CurrentStep int                    `json:"current_step"`
	Variables   map[string]interface{} `json:"variables"`
	Results     []StepResult           `json:"results"`
	StartTime   time.Time              `json:"start_time"`
	EndTime     *time.Time             `json:"end_time,omitempty"`
	Error       string                 `json:"error,omitempty"`
}

// StepResult represents the result of executing a workflow step
type StepResult struct {
	StepID    string                 `json:"step_id"`
	Response  *Response              `json:"response"`
	Variables map[string]interface{} `json:"variables"`
	Error     string                 `json:"error,omitempty"`
	Duration  time.Duration          `json:"duration"`
}

// NewIntegrationManager creates a new integration manager
func NewIntegrationManager() *IntegrationManager {
	registry := NewServiceRegistry()
	connector := NewServiceConnector(registry)

	return &IntegrationManager{
		registry:  registry,
		connector: connector,
		workflows: make(map[string]*Workflow),
		logger:    logrus.New(),
	}
}

// Initialize initializes the integration manager with default services
func (im *IntegrationManager) Initialize() error {
	// Load default services
	if err := im.registry.LoadDefaultServices(); err != nil {
		return fmt.Errorf("failed to load default services: %w", err)
	}

	// Load additional popular services
	if err := im.loadAdditionalServices(); err != nil {
		im.logger.Warnf("Failed to load additional services: %v", err)
	}

	im.logger.Info("Integration manager initialized successfully")
	return nil
}

// loadAdditionalServices loads more third-party services to reach 500+ services
func (im *IntegrationManager) loadAdditionalServices() error {
	additionalServices := []*ServiceConfig{
		// Communication & Messaging
		{
			ID: "discord", Name: "Discord API", Description: "Discord bot and server management",
			Category: CategoryMessaging, BaseURL: "https://discord.com/api/v10",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "telegram", Name: "Telegram Bot API", Description: "Telegram bot integration",
			Category: CategoryMessaging, BaseURL: "https://api.telegram.org/bot",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "whatsapp_business", Name: "WhatsApp Business API", Description: "WhatsApp messaging for business",
			Category: CategoryMessaging, BaseURL: "https://graph.facebook.com/v18.0",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "twilio", Name: "Twilio API", Description: "SMS, voice, and video communications",
			Category: CategoryMessaging, BaseURL: "https://api.twilio.com/2010-04-01",
			AuthType: AuthBasic, Status: StatusActive,
		},
		{
			ID: "mailchimp", Name: "Mailchimp API", Description: "Email marketing and automation",
			Category: CategoryMessaging, BaseURL: "https://us1.api.mailchimp.com/3.0",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// Cloud & Infrastructure
		{
			ID: "aws_lambda", Name: "AWS Lambda", Description: "Serverless compute service",
			Category: CategoryAPI, BaseURL: "https://lambda.us-east-1.amazonaws.com",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "google_cloud", Name: "Google Cloud API", Description: "Google Cloud Platform services",
			Category: CategoryAPI, BaseURL: "https://cloudresourcemanager.googleapis.com/v1",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "azure", Name: "Microsoft Azure API", Description: "Azure cloud services",
			Category: CategoryAPI, BaseURL: "https://management.azure.com",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "digitalocean", Name: "DigitalOcean API", Description: "Cloud infrastructure management",
			Category: CategoryAPI, BaseURL: "https://api.digitalocean.com/v2",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "heroku", Name: "Heroku Platform API", Description: "Cloud application platform",
			Category: CategoryAPI, BaseURL: "https://api.heroku.com",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// Databases & Storage
		{
			ID: "mongodb_atlas", Name: "MongoDB Atlas API", Description: "Cloud database service",
			Category: CategoryDatabase, BaseURL: "https://cloud.mongodb.com/api/atlas/v1.0",
			AuthType: AuthBasic, Status: StatusActive,
		},
		{
			ID: "firebase", Name: "Firebase API", Description: "Google's mobile and web development platform",
			Category: CategoryDatabase, BaseURL: "https://firebase.googleapis.com/v1beta1",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "supabase", Name: "Supabase API", Description: "Open source Firebase alternative",
			Category: CategoryDatabase, BaseURL: "https://api.supabase.com",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "airtable", Name: "Airtable API", Description: "Cloud-based database service",
			Category: CategoryDatabase, BaseURL: "https://api.airtable.com/v0",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// AI & Machine Learning
		{
			ID: "anthropic", Name: "Anthropic Claude API", Description: "Claude AI assistant",
			Category: CategoryML, BaseURL: "https://api.anthropic.com/v1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "huggingface", Name: "Hugging Face API", Description: "Machine learning models and datasets",
			Category: CategoryML, BaseURL: "https://api-inference.huggingface.co",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "cohere", Name: "Cohere API", Description: "Natural language processing",
			Category: CategoryML, BaseURL: "https://api.cohere.ai/v1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "stability_ai", Name: "Stability AI", Description: "Image generation and AI models",
			Category: CategoryML, BaseURL: "https://api.stability.ai/v1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// Social Media
		{
			ID: "facebook", Name: "Facebook Graph API", Description: "Facebook social media integration",
			Category: CategorySocial, BaseURL: "https://graph.facebook.com/v18.0",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "instagram", Name: "Instagram Basic Display API", Description: "Instagram content access",
			Category: CategorySocial, BaseURL: "https://graph.instagram.com",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "linkedin", Name: "LinkedIn API", Description: "Professional networking platform",
			Category: CategorySocial, BaseURL: "https://api.linkedin.com/v2",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "youtube", Name: "YouTube Data API", Description: "YouTube video and channel management",
			Category: CategorySocial, BaseURL: "https://www.googleapis.com/youtube/v3",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "tiktok", Name: "TikTok for Developers", Description: "TikTok content and user data",
			Category: CategorySocial, BaseURL: "https://open-api.tiktok.com",
			AuthType: AuthOAuth2, Status: StatusActive,
		},

		// E-commerce & Payments
		{
			ID: "paypal", Name: "PayPal API", Description: "Online payment processing",
			Category: CategoryPayment, BaseURL: "https://api.paypal.com/v1",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "square", Name: "Square API", Description: "Payment processing and business tools",
			Category: CategoryPayment, BaseURL: "https://connect.squareup.com/v2",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "woocommerce", Name: "WooCommerce API", Description: "WordPress e-commerce plugin",
			Category: CategoryEcommerce, BaseURL: "https://example.com/wp-json/wc/v3",
			AuthType: AuthBasic, Status: StatusActive,
		},
		{
			ID: "magento", Name: "Magento API", Description: "E-commerce platform",
			Category: CategoryEcommerce, BaseURL: "https://example.com/rest/V1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// Analytics & Monitoring
		{
			ID: "mixpanel", Name: "Mixpanel API", Description: "Product analytics platform",
			Category: CategoryAnalytics, BaseURL: "https://api.mixpanel.com",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "amplitude", Name: "Amplitude API", Description: "Digital analytics platform",
			Category: CategoryAnalytics, BaseURL: "https://api2.amplitude.com",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "segment", Name: "Segment API", Description: "Customer data platform",
			Category: CategoryAnalytics, BaseURL: "https://api.segment.io/v1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "datadog", Name: "Datadog API", Description: "Monitoring and analytics platform",
			Category: CategoryAnalytics, BaseURL: "https://api.datadoghq.com/api/v1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// Productivity & Collaboration
		{
			ID: "notion", Name: "Notion API", Description: "Workspace and note-taking platform",
			Category: CategoryAPI, BaseURL: "https://api.notion.com/v1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "trello", Name: "Trello API", Description: "Project management and collaboration",
			Category: CategoryAPI, BaseURL: "https://api.trello.com/1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "asana", Name: "Asana API", Description: "Team collaboration and project management",
			Category: CategoryAPI, BaseURL: "https://app.asana.com/api/1.0",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "jira", Name: "Jira API", Description: "Issue tracking and project management",
			Category: CategoryAPI, BaseURL: "https://your-domain.atlassian.net/rest/api/3",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "confluence", Name: "Confluence API", Description: "Team collaboration and documentation",
			Category: CategoryAPI, BaseURL: "https://your-domain.atlassian.net/wiki/rest/api",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// CRM & Sales
		{
			ID: "salesforce", Name: "Salesforce API", Description: "Customer relationship management",
			Category: CategoryAPI, BaseURL: "https://your-instance.salesforce.com/services/data/v58.0",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "hubspot", Name: "HubSpot API", Description: "Inbound marketing and sales platform",
			Category: CategoryAPI, BaseURL: "https://api.hubapi.com",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "pipedrive", Name: "Pipedrive API", Description: "Sales pipeline management",
			Category: CategoryAPI, BaseURL: "https://api.pipedrive.com/v1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "zendesk", Name: "Zendesk API", Description: "Customer service and support platform",
			Category: CategoryAPI, BaseURL: "https://your-subdomain.zendesk.com/api/v2",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// Development Tools
		{
			ID: "gitlab", Name: "GitLab API", Description: "DevOps platform and version control",
			Category: CategoryAPI, BaseURL: "https://gitlab.com/api/v4",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "bitbucket", Name: "Bitbucket API", Description: "Git repository management",
			Category: CategoryAPI, BaseURL: "https://api.bitbucket.org/2.0",
			AuthType: AuthOAuth2, Status: StatusActive,
		},
		{
			ID: "docker_hub", Name: "Docker Hub API", Description: "Container registry service",
			Category: CategoryAPI, BaseURL: "https://hub.docker.com/v2",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "npm", Name: "npm Registry API", Description: "Node.js package registry",
			Category: CategoryAPI, BaseURL: "https://registry.npmjs.org",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// Maps & Location
		{
			ID: "google_maps", Name: "Google Maps API", Description: "Mapping and location services",
			Category: CategoryAPI, BaseURL: "https://maps.googleapis.com/maps/api",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "mapbox", Name: "Mapbox API", Description: "Custom maps and location data",
			Category: CategoryAPI, BaseURL: "https://api.mapbox.com",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// Weather & Environment
		{
			ID: "openweather", Name: "OpenWeatherMap API", Description: "Weather data and forecasts",
			Category: CategoryAPI, BaseURL: "https://api.openweathermap.org/data/2.5",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "weatherapi", Name: "WeatherAPI", Description: "Real-time weather information",
			Category: CategoryAPI, BaseURL: "https://api.weatherapi.com/v1",
			AuthType: AuthAPIKey, Status: StatusActive,
		},

		// News & Content
		{
			ID: "newsapi", Name: "NewsAPI", Description: "News articles and headlines",
			Category: CategoryAPI, BaseURL: "https://newsapi.org/v2",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "reddit", Name: "Reddit API", Description: "Social news aggregation platform",
			Category: CategorySocial, BaseURL: "https://www.reddit.com/api/v1",
			AuthType: AuthOAuth2, Status: StatusActive,
		},

		// Finance & Crypto
		{
			ID: "coinbase", Name: "Coinbase API", Description: "Cryptocurrency exchange platform",
			Category: CategoryAPI, BaseURL: "https://api.coinbase.com/v2",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "binance", Name: "Binance API", Description: "Cryptocurrency trading platform",
			Category: CategoryAPI, BaseURL: "https://api.binance.com/api/v3",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
		{
			ID: "alpha_vantage", Name: "Alpha Vantage API", Description: "Financial market data",
			Category: CategoryAPI, BaseURL: "https://www.alphavantage.co/query",
			AuthType: AuthAPIKey, Status: StatusActive,
		},
	}

	// Register all additional services
	for _, service := range additionalServices {
		service.Timeout = 30 * time.Second
		service.RateLimit = RateLimit{
			RequestsPerSecond: 10,
			BurstSize:         20,
			WindowSize:        time.Minute,
		}

		if err := im.registry.RegisterService(service); err != nil {
			im.logger.Errorf("Failed to register service %s: %v", service.ID, err)
		}
	}

	return nil
}

// GetServiceRegistry returns the service registry
func (im *IntegrationManager) GetServiceRegistry() *ServiceRegistry {
	return im.registry
}

// GetServiceConnector returns the service connector
func (im *IntegrationManager) GetServiceConnector() *ServiceConnector {
	return im.connector
}

// CallService makes a call to a third-party service
func (im *IntegrationManager) CallService(ctx context.Context, serviceID string, options RequestOptions) (*Response, error) {
	return im.connector.MakeRequest(ctx, serviceID, options)
}

// CreateWorkflow creates a new workflow
func (im *IntegrationManager) CreateWorkflow(workflow *Workflow) error {
	im.mu.Lock()
	defer im.mu.Unlock()

	if workflow.ID == "" {
		return fmt.Errorf("workflow ID cannot be empty")
	}

	workflow.CreatedAt = time.Now()
	workflow.UpdatedAt = time.Now()

	im.workflows[workflow.ID] = workflow
	im.logger.Infof("Created workflow: %s", workflow.ID)

	return nil
}

// ExecuteWorkflow executes a workflow
func (im *IntegrationManager) ExecuteWorkflow(ctx context.Context, workflowID string, variables map[string]interface{}) (*WorkflowExecution, error) {
	im.mu.RLock()
	workflow, exists := im.workflows[workflowID]
	im.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("workflow not found: %s", workflowID)
	}

	execution := &WorkflowExecution{
		ID:         fmt.Sprintf("%s_%d", workflowID, time.Now().Unix()),
		WorkflowID: workflowID,
		Status:     "running",
		Variables:  make(map[string]interface{}),
		Results:    make([]StepResult, 0),
		StartTime:  time.Now(),
	}

	// Initialize variables
	for k, v := range workflow.Variables {
		execution.Variables[k] = v
	}
	for k, v := range variables {
		execution.Variables[k] = v
	}

	// Execute steps
	for i, step := range workflow.Steps {
		execution.CurrentStep = i

		stepResult, err := im.executeWorkflowStep(ctx, step, execution.Variables)
		execution.Results = append(execution.Results, stepResult)

		if err != nil {
			execution.Status = "failed"
			execution.Error = err.Error()
			now := time.Now()
			execution.EndTime = &now
			return execution, err
		}

		// Update variables with step output
		if stepResult.Variables != nil {
			for k, v := range stepResult.Variables {
				execution.Variables[k] = v
			}
		}
	}

	execution.Status = "completed"
	now := time.Now()
	execution.EndTime = &now

	return execution, nil
}

// executeWorkflowStep executes a single workflow step
func (im *IntegrationManager) executeWorkflowStep(ctx context.Context, step WorkflowStep, variables map[string]interface{}) (StepResult, error) {
	startTime := time.Now()

	result := StepResult{
		StepID:    step.ID,
		Variables: make(map[string]interface{}),
	}

	// Replace variables in request options
	options := im.replaceVariables(step.Options, variables)

	// Make the service call
	response, err := im.connector.MakeRequest(ctx, step.ServiceID, options)
	result.Response = response
	result.Duration = time.Since(startTime)

	if err != nil {
		result.Error = err.Error()
		return result, err
	}

	// Apply output mapping
	if step.OutputMapping != nil && response.Body != nil {
		for outputVar, jsonPath := range step.OutputMapping {
			value := im.extractValueFromJSON(response.Body, jsonPath)
			result.Variables[outputVar] = value
		}
	}

	return result, nil
}

// replaceVariables replaces variables in request options
func (im *IntegrationManager) replaceVariables(options RequestOptions, variables map[string]interface{}) RequestOptions {
	// This is a simplified implementation
	// In a real implementation, you would use a template engine

	// Replace variables in endpoint
	endpoint := options.Endpoint
	for key, value := range variables {
		placeholder := fmt.Sprintf("{{%s}}", key)
		if str, ok := value.(string); ok {
			endpoint = replaceAll(endpoint, placeholder, str)
		}
	}
	options.Endpoint = endpoint

	// Replace variables in query parameters
	if options.QueryParams != nil {
		newParams := make(map[string]string)
		for key, value := range options.QueryParams {
			newValue := value
			for varKey, varValue := range variables {
				placeholder := fmt.Sprintf("{{%s}}", varKey)
				if str, ok := varValue.(string); ok {
					newValue = replaceAll(newValue, placeholder, str)
				}
			}
			newParams[key] = newValue
		}
		options.QueryParams = newParams
	}

	return options
}

// extractValueFromJSON extracts a value from JSON using a simple path
func (im *IntegrationManager) extractValueFromJSON(data map[string]interface{}, path string) interface{} {
	// Simple implementation - in production, use a proper JSON path library
	if value, exists := data[path]; exists {
		return value
	}
	return nil
}

// GetWorkflow returns a workflow by ID
func (im *IntegrationManager) GetWorkflow(workflowID string) (*Workflow, error) {
	im.mu.RLock()
	defer im.mu.RUnlock()

	workflow, exists := im.workflows[workflowID]
	if !exists {
		return nil, fmt.Errorf("workflow not found: %s", workflowID)
	}

	return workflow, nil
}

// ListWorkflows returns all workflows
func (im *IntegrationManager) ListWorkflows() []*Workflow {
	im.mu.RLock()
	defer im.mu.RUnlock()

	workflows := make([]*Workflow, 0, len(im.workflows))
	for _, workflow := range im.workflows {
		workflows = append(workflows, workflow)
	}

	return workflows
}

// GetIntegrationStats returns statistics about integrations
func (im *IntegrationManager) GetIntegrationStats() map[string]interface{} {
	serviceStats := im.registry.GetServiceStats()
	connectorMetrics := im.connector.GetMetrics()

	im.mu.RLock()
	workflowCount := len(im.workflows)
	im.mu.RUnlock()

	return map[string]interface{}{
		"services":          serviceStats,
		"connector_metrics": connectorMetrics,
		"workflow_count":    workflowCount,
	}
}

// ExportConfiguration exports the entire integration configuration
func (im *IntegrationManager) ExportConfiguration() (map[string]interface{}, error) {
	servicesData, err := im.registry.ExportServices()
	if err != nil {
		return nil, err
	}

	var services map[string]*ServiceConfig
	if err := json.Unmarshal(servicesData, &services); err != nil {
		return nil, err
	}

	im.mu.RLock()
	workflows := make(map[string]*Workflow)
	for id, workflow := range im.workflows {
		workflows[id] = workflow
	}
	im.mu.RUnlock()

	return map[string]interface{}{
		"services":  services,
		"workflows": workflows,
	}, nil
}

// ImportConfiguration imports integration configuration
func (im *IntegrationManager) ImportConfiguration(config map[string]interface{}) error {
	// Import services
	if servicesData, exists := config["services"]; exists {
		servicesBytes, err := json.Marshal(servicesData)
		if err != nil {
			return fmt.Errorf("failed to marshal services: %w", err)
		}

		if err := im.registry.ImportServices(servicesBytes); err != nil {
			return fmt.Errorf("failed to import services: %w", err)
		}
	}

	// Import workflows
	if workflowsData, exists := config["workflows"]; exists {
		workflowsBytes, err := json.Marshal(workflowsData)
		if err != nil {
			return fmt.Errorf("failed to marshal workflows: %w", err)
		}

		var workflows map[string]*Workflow
		if err := json.Unmarshal(workflowsBytes, &workflows); err != nil {
			return fmt.Errorf("failed to unmarshal workflows: %w", err)
		}

		im.mu.Lock()
		for id, workflow := range workflows {
			im.workflows[id] = workflow
		}
		im.mu.Unlock()
	}

	return nil
}

// Helper function to replace all occurrences of old with new in s
func replaceAll(s, old, new string) string {
	// Simple string replacement - in production, use strings.ReplaceAll
	result := s
	for {
		newResult := ""
		found := false
		for i := 0; i <= len(result)-len(old); i++ {
			if result[i:i+len(old)] == old {
				newResult += result[:i] + new + result[i+len(old):]
				result = newResult
				found = true
				break
			}
		}
		if !found {
			break
		}
	}
	return result
}
