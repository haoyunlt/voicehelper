-- 聊天机器人数据库架构
-- Version: 1.1.0
-- Sprint 1: 生产就绪

-- 创建数据库
CREATE DATABASE IF NOT EXISTS chatbot_prod;
USE chatbot_prod;

-- 启用必要的扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==================== 租户表 ====================
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    plan VARCHAR(20) DEFAULT 'free', -- free, basic, premium, enterprise
    status VARCHAR(20) DEFAULT 'active', -- active, suspended, deleted
    config JSONB DEFAULT '{}',
    quota JSONB DEFAULT '{"daily_tokens": 100000, "daily_audio_minutes": 60, "max_concurrent": 3}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_tenants_status (status),
    INDEX idx_tenants_plan (plan)
);

-- ==================== 用户表 ====================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(50) UNIQUE NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    openid VARCHAR(100),
    unionid VARCHAR(100),
    username VARCHAR(50),
    nickname VARCHAR(100),
    avatar_url TEXT,
    email VARCHAR(100),
    phone VARCHAR(20),
    role VARCHAR(20) DEFAULT 'user', -- user, admin, super_admin
    status VARCHAR(20) DEFAULT 'active', -- active, inactive, banned
    last_login_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_users_tenant (tenant_id),
    INDEX idx_users_openid (openid),
    INDEX idx_users_status (status),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);

-- ==================== 会话表 ====================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id VARCHAR(100) UNIQUE NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    title VARCHAR(200),
    summary TEXT,
    channel VARCHAR(20) DEFAULT 'web', -- web, wechat, api
    mode VARCHAR(20) DEFAULT 'text', -- text, voice, mixed
    status VARCHAR(20) DEFAULT 'active', -- active, archived, deleted
    metadata JSONB DEFAULT '{}',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_conversations_tenant_user (tenant_id, user_id),
    INDEX idx_conversations_status (status),
    INDEX idx_conversations_started_at (started_at DESC),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ==================== 消息表 ====================
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id VARCHAR(100) UNIQUE NOT NULL,
    conversation_id VARCHAR(100) NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50),
    role VARCHAR(20) NOT NULL, -- user, assistant, system, tool
    content TEXT NOT NULL,
    modality VARCHAR(20) DEFAULT 'text', -- text, audio, asr, tts
    references JSONB DEFAULT '[]', -- RAG引用
    metadata JSONB DEFAULT '{}', -- 包含token使用、延迟等
    parent_message_id VARCHAR(100), -- 用于消息树结构
    version INT DEFAULT 1,
    is_deleted BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_messages_conversation (conversation_id),
    INDEX idx_messages_tenant (tenant_id),
    INDEX idx_messages_created_at (created_at DESC),
    INDEX idx_messages_role (role),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);

-- ==================== 审计日志表 ====================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50),
    action VARCHAR(50) NOT NULL, -- login, logout, query, upload, tool_call, etc
    resource_type VARCHAR(50), -- conversation, message, document, tool
    resource_id VARCHAR(100),
    request_id VARCHAR(100),
    ip_address INET,
    user_agent TEXT,
    status VARCHAR(20), -- success, failed, error
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    duration_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_audit_tenant_user (tenant_id, user_id),
    INDEX idx_audit_action (action),
    INDEX idx_audit_created_at (created_at DESC),
    INDEX idx_audit_request_id (request_id)
);

-- ==================== 文档表 ====================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(100) UNIQUE NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    dataset_id VARCHAR(50) DEFAULT 'default',
    filename VARCHAR(255) NOT NULL,
    file_type VARCHAR(20),
    file_size BIGINT,
    storage_path TEXT,
    checksum VARCHAR(64),
    chunk_count INT DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, indexed, failed
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    created_by VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    indexed_at TIMESTAMP WITH TIME ZONE,
    INDEX idx_documents_tenant_dataset (tenant_id, dataset_id),
    INDEX idx_documents_status (status),
    INDEX idx_documents_created_at (created_at DESC),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);

-- ==================== 工具调用记录表 ====================
CREATE TABLE IF NOT EXISTS tool_calls (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    call_id VARCHAR(100) UNIQUE NOT NULL,
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    conversation_id VARCHAR(100),
    tool_name VARCHAR(50) NOT NULL,
    operation VARCHAR(50) NOT NULL,
    parameters JSONB DEFAULT '{}',
    result JSONB,
    status VARCHAR(20) DEFAULT 'pending', -- pending, running, success, failed
    error_message TEXT,
    duration_ms INT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    INDEX idx_tool_calls_tenant_user (tenant_id, user_id),
    INDEX idx_tool_calls_tool (tool_name),
    INDEX idx_tool_calls_status (status),
    INDEX idx_tool_calls_created_at (created_at DESC),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- ==================== 使用统计表 ====================
CREATE TABLE IF NOT EXISTS usage_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(50) NOT NULL,
    user_id VARCHAR(50),
    date DATE NOT NULL,
    metric_type VARCHAR(50) NOT NULL, -- tokens, audio_minutes, conversations, messages
    metric_value DECIMAL(10, 2) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY unique_usage_stats (tenant_id, user_id, date, metric_type),
    INDEX idx_usage_stats_tenant_date (tenant_id, date),
    INDEX idx_usage_stats_metric_type (metric_type),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);

-- ==================== 缓存表 ====================
CREATE TABLE IF NOT EXISTS cache_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    cache_key VARCHAR(255) UNIQUE NOT NULL,
    cache_type VARCHAR(50) NOT NULL, -- semantic, query, embedding
    tenant_id VARCHAR(50),
    value JSONB NOT NULL,
    ttl INT DEFAULT 3600, -- 秒
    hit_count INT DEFAULT 0,
    last_hit_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_cache_type (cache_type),
    INDEX idx_cache_tenant (tenant_id),
    INDEX idx_cache_expires_at (expires_at)
);

-- ==================== 函数和触发器 ====================

-- 自动更新 updated_at 时间戳
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为所有表添加更新触发器
CREATE TRIGGER update_tenants_updated_at BEFORE UPDATE ON tenants
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_messages_updated_at BEFORE UPDATE ON messages
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==================== 初始数据 ====================

-- 插入默认租户
INSERT INTO tenants (tenant_id, name, plan, status) 
VALUES ('default', 'Default Tenant', 'free', 'active')
ON CONFLICT (tenant_id) DO NOTHING;

-- ==================== 权限设置 ====================

-- 创建只读用户（用于查询）
CREATE USER chatbot_reader WITH PASSWORD 'readonly_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO chatbot_reader;

-- 创建应用用户（用于应用访问）
CREATE USER chatbot_app WITH PASSWORD 'app_password';
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO chatbot_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO chatbot_app;

-- ==================== 性能优化 ====================

-- 分区表（按月分区消息表）
CREATE TABLE messages_2025_01 PARTITION OF messages
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');

CREATE TABLE messages_2025_02 PARTITION OF messages
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');

-- 创建物化视图（热门查询优化）
CREATE MATERIALIZED VIEW mv_daily_usage AS
SELECT 
    tenant_id,
    date,
    SUM(CASE WHEN metric_type = 'tokens' THEN metric_value ELSE 0 END) as total_tokens,
    SUM(CASE WHEN metric_type = 'audio_minutes' THEN metric_value ELSE 0 END) as total_audio_minutes,
    SUM(CASE WHEN metric_type = 'conversations' THEN metric_value ELSE 0 END) as total_conversations,
    SUM(CASE WHEN metric_type = 'messages' THEN metric_value ELSE 0 END) as total_messages
FROM usage_stats
GROUP BY tenant_id, date
WITH DATA;

-- 创建刷新物化视图的定时任务（需要pg_cron扩展）
-- SELECT cron.schedule('refresh-mv-daily-usage', '0 1 * * *', 'REFRESH MATERIALIZED VIEW mv_daily_usage;');

-- ==================== 备份策略 ====================
-- 建议配置：
-- 1. 每日全量备份
-- 2. 每小时增量备份
-- 3. 保留30天备份历史
-- 4. 异地备份到S3/OSS
