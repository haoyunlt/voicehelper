package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"

	"voicehelper/backend/internal/repository"
	"voicehelper/backend/pkg/config"
	"voicehelper/backend/pkg/database"
	"voicehelper/backend/pkg/persistence"
)

func main() {
	log.Println("Starting persistence system demo")

	// 加载配置
	cfg, err := config.Load()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// 连接数据库
	dbConn, err := database.NewPostgresConnection(cfg.Database)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer dbConn.Close()

	// 连接Redis
	redisClient := redis.NewClient(&redis.Options{
		Addr:     fmt.Sprintf("%s:%d", cfg.Redis.Host, cfg.Redis.Port),
		Password: cfg.Redis.Password,
		DB:       cfg.Redis.Database,
	})
	defer redisClient.Close()

	// 创建持久化管理器
	persistenceConfig := &persistence.PersistenceConfig{
		CacheEnabled:      true,
		CacheDefaultTTL:   30 * time.Minute,
		MigrationEnabled:  true,
		ValidationEnabled: true,
	}

	pm := persistence.NewPersistenceManager(dbConn.DB, redisClient, persistenceConfig)

	// 初始化持久化系统
	ctx := context.Background()
	if err := pm.Initialize(ctx); err != nil {
		log.Fatalf("Failed to initialize persistence manager: %v", err)
	}

	log.Println("Persistence system initialized successfully")

	// 演示功能
	if err := demonstratePersistence(ctx, pm); err != nil {
		log.Fatalf("Demonstration failed: %v", err)
	}

	// 健康检查
	if err := pm.Health(ctx); err != nil {
		log.Printf("Health check failed: %v", err)
	} else {
		log.Println("Health check passed")
	}

	// 获取统计信息
	stats := pm.GetStats(ctx)
	log.Printf("Persistence stats: %+v", stats)

	// 关闭持久化管理器
	if err := pm.Close(); err != nil {
		log.Printf("Failed to close persistence manager: %v", err)
	}

	log.Println("Persistence system demo completed")
}

func demonstratePersistence(ctx context.Context, pm *persistence.PersistenceManager) error {
	log.Println("Demonstrating persistence functionality...")

	// 1. 创建租户
	tenant := &repository.Tenant{
		TenantID: "demo-tenant",
		Name:     "Demo Tenant",
		Plan:     "premium",
		Status:   "active",
		Config: map[string]interface{}{
			"max_users":         100,
			"max_conversations": 1000,
		},
		Quota: map[string]interface{}{
			"daily_tokens":        100000,
			"daily_audio_minutes": 120,
			"max_concurrent":      5,
		},
	}

	if err := pm.CreateTenantWithCache(ctx, tenant); err != nil {
		return fmt.Errorf("failed to create tenant: %v", err)
	}
	log.Printf("Created tenant: %s", tenant.TenantID)

	// 2. 获取租户（应该从缓存获取）
	retrievedTenant, err := pm.GetTenantWithCache(ctx, tenant.TenantID)
	if err != nil {
		return fmt.Errorf("failed to get tenant: %v", err)
	}
	log.Printf("Retrieved tenant: %s (plan: %s)", retrievedTenant.Name, retrievedTenant.Plan)

	// 3. 创建用户
	user := &repository.User{
		ID:       "demo-user",
		TenantID: tenant.TenantID,
		Username: "demo_user",
		Nickname: "Demo User",
		Email:    "demo@example.com",
		Role:     "user",
		Status:   "active",
	}

	userRepo := pm.GetUserRepository()
	if err := userRepo.Create(ctx, user); err != nil {
		return fmt.Errorf("failed to create user: %v", err)
	}
	log.Printf("Created user: %s", user.ID)

	// 4. 获取用户（带缓存）
	retrievedUser, err := pm.GetUserWithCache(ctx, user.ID)
	if err != nil {
		return fmt.Errorf("failed to get user: %v", err)
	}
	log.Printf("Retrieved user: %s (%s)", retrievedUser.Nickname, retrievedUser.Email)

	// 5. 创建会话
	conversation := &repository.Conversation{
		ID:       "demo-conversation",
		UserID:   user.ID,
		TenantID: tenant.TenantID,
		Title:    "Demo Conversation",
		Summary:  "This is a demo conversation",
		Status:   "active",
		Metadata: map[string]interface{}{
			"tags": []string{"demo", "test"},
		},
	}

	conversationRepo := pm.GetConversationRepository()
	if err := conversationRepo.Create(ctx, conversation); err != nil {
		return fmt.Errorf("failed to create conversation: %v", err)
	}
	log.Printf("Created conversation: %s", conversation.ID)

	// 6. 添加消息
	message := &repository.Message{
		ID:             "demo-message",
		ConversationID: conversation.ID,
		Role:           "user",
		Content:        "Hello, this is a demo message!",
		Modality:       "text",
		TokenCount:     10,
		Metadata: map[string]interface{}{
			"source": "demo",
		},
	}

	if err := pm.CreateMessageWithCache(ctx, message); err != nil {
		return fmt.Errorf("failed to create message: %v", err)
	}
	log.Printf("Created message: %s", message.Content)

	// 7. 创建语音会话
	voiceSession := &repository.VoiceSession{
		SessionID:      "demo-voice-session",
		UserID:         user.ID,
		TenantID:       tenant.TenantID,
		ConversationID: conversation.ID,
		Status:         "active",
		Config: map[string]interface{}{
			"sample_rate": 16000,
			"channels":    1,
			"language":    "zh-CN",
		},
		StartTime: time.Now(),
		Metadata: map[string]interface{}{
			"device": "demo-device",
		},
	}

	voiceSessionRepo := pm.GetVoiceSessionRepository()
	if err := voiceSessionRepo.Create(ctx, voiceSession); err != nil {
		return fmt.Errorf("failed to create voice session: %v", err)
	}
	log.Printf("Created voice session: %s", voiceSession.SessionID)

	// 8. 获取语音会话（带缓存）
	retrievedVoiceSession, err := pm.GetVoiceSessionWithCache(ctx, voiceSession.SessionID)
	if err != nil {
		return fmt.Errorf("failed to get voice session: %v", err)
	}
	log.Printf("Retrieved voice session: %s (status: %s)",
		retrievedVoiceSession.SessionID, retrievedVoiceSession.Status)

	// 9. 创建文档
	document := &repository.DocumentModel{
		DocumentID:  "demo-document",
		TenantID:    tenant.TenantID,
		Title:       "Demo Document",
		Content:     "This is a demo document content for testing purposes.",
		ContentType: "text/plain",
		Source:      "demo",
		Status:      "active",
		Metadata: map[string]interface{}{
			"category": "demo",
			"tags":     []string{"test", "demo"},
		},
	}

	documentRepo := pm.GetDocumentRepository()
	if err := documentRepo.Create(ctx, document); err != nil {
		return fmt.Errorf("failed to create document: %v", err)
	}
	log.Printf("Created document: %s", document.Title)

	// 10. 搜索文档
	documents, err := documentRepo.Search(ctx, tenant.TenantID, "demo", 10, 0)
	if err != nil {
		return fmt.Errorf("failed to search documents: %v", err)
	}
	log.Printf("Found %d documents matching 'demo'", len(documents))

	// 11. 获取租户统计
	tenantStats, err := pm.GetTenantRepository().GetStats(ctx, tenant.TenantID)
	if err != nil {
		return fmt.Errorf("failed to get tenant stats: %v", err)
	}
	log.Printf("Tenant stats: users=%d, conversations=%d, messages=%d, documents=%d",
		tenantStats.UserCount, tenantStats.ConversationCount,
		tenantStats.MessageCount, tenantStats.DocumentCount)

	log.Println("Persistence demonstration completed successfully")
	return nil
}
