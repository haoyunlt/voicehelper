-- VoiceHelper 性能优化索引
-- 基于查询模式和性能分析添加的索引

-- ==================== 用户表索引优化 ====================
-- 复合索引：租户+状态（常见查询组合）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_tenant_status 
ON users (tenant_id, status) WHERE deleted_at IS NULL;

-- 复合索引：租户+角色（权限查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_tenant_role 
ON users (tenant_id, role) WHERE deleted_at IS NULL;

-- 最后登录时间索引（活跃用户分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_last_login 
ON users (last_login) WHERE last_login IS NOT NULL;

-- ==================== 会话表索引优化 ====================
-- 复合索引：用户+最后消息时间（用户会话列表）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_user_last_msg 
ON conversations (user_id, last_msg_at DESC) WHERE deleted_at IS NULL;

-- 复合索引：租户+状态+创建时间（管理查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_tenant_status_created 
ON conversations (tenant_id, status, created_at DESC) WHERE deleted_at IS NULL;

-- 令牌使用量索引（成本分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_token_count 
ON conversations (token_count DESC) WHERE token_count > 0;

-- ==================== 消息表索引优化 ====================
-- 复合索引：会话+创建时间（消息历史查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_conversation_created 
ON messages (conversation_id, created_at DESC);

-- 复合索引：会话+角色（特定角色消息查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_conversation_role 
ON messages (conversation_id, role);

-- 令牌使用量索引（成本分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_token_count 
ON messages (token_count DESC) WHERE token_count > 0;

-- 模态类型索引（语音/文本分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_modality 
ON messages (modality, created_at DESC);

-- ==================== 数据集表索引优化 ====================
-- 复合索引：租户+类型+状态
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_datasets_tenant_type_status 
ON datasets (tenant_id, type, status) WHERE deleted_at IS NULL;

-- 创建者索引（用户数据集查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_datasets_created_by 
ON datasets (created_by, created_at DESC) WHERE deleted_at IS NULL;

-- 文档数量索引（大数据集识别）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_datasets_doc_count 
ON datasets (doc_count DESC) WHERE doc_count > 0;

-- ==================== 文档表索引优化 ====================
-- 复合索引：数据集+状态+更新时间
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_dataset_status_updated 
ON documents (dataset_id, status, updated_at DESC) WHERE deleted_at IS NULL;

-- 文件类型索引（类型统计）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_type 
ON documents (type, created_at DESC) WHERE type IS NOT NULL;

-- 文件大小索引（存储分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_size 
ON documents (size DESC) WHERE size > 0;

-- ==================== API密钥表索引优化 ====================
-- 复合索引：租户+状态+过期时间
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_apikeys_tenant_status_expires 
ON api_keys (tenant_id, status, expires_at) WHERE deleted_at IS NULL;

-- 最后使用时间索引（活跃密钥分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_apikeys_last_used 
ON api_keys (last_used_at DESC) WHERE last_used_at IS NOT NULL;

-- 速率限制索引（限流查询）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_apikeys_rate_limit 
ON api_keys (rate_limit) WHERE rate_limit IS NOT NULL;

-- ==================== 审计日志表索引优化 ====================
-- 复合索引：租户+用户+时间（用户行为分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_tenant_user_created 
ON audit_logs (tenant_id, user_id, created_at DESC);

-- 复合索引：动作+时间（行为统计）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_action_created 
ON audit_logs (action, created_at DESC);

-- 资源类型索引（资源访问分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_resource_type 
ON audit_logs (resource, created_at DESC) WHERE resource IS NOT NULL;

-- IP地址索引（安全分析）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_audit_ip_created 
ON audit_logs (ip_address, created_at DESC) WHERE ip_address IS NOT NULL;

-- ==================== 性能监控索引 ====================
-- 创建时间范围查询优化（所有表）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_range 
ON users (created_at) WHERE created_at >= CURRENT_DATE - INTERVAL '30 days';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_created_range 
ON conversations (created_at) WHERE created_at >= CURRENT_DATE - INTERVAL '7 days';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_created_range 
ON messages (created_at) WHERE created_at >= CURRENT_DATE - INTERVAL '24 hours';

-- ==================== 分区表索引（如果使用分区） ====================
-- 审计日志按月分区的索引模板
-- CREATE INDEX CONCURRENTLY idx_audit_logs_y2024m01_tenant_created 
-- ON audit_logs_y2024m01 (tenant_id, created_at DESC);

-- ==================== 函数索引（高级优化） ====================
-- JSON字段查询优化
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_metadata_gin 
ON conversations USING GIN (metadata) WHERE metadata != '{}';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_references_gin 
ON messages USING GIN (references) WHERE references != '[]';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_datasets_metadata_gin 
ON datasets USING GIN (metadata) WHERE metadata != '{}';

-- 文本搜索索引（如果需要全文搜索）
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_messages_content_fts 
ON messages USING GIN (to_tsvector('chinese', content));

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_conversations_title_fts 
ON conversations USING GIN (to_tsvector('chinese', title)) WHERE title IS NOT NULL;

-- ==================== 唯一约束索引 ====================
-- 确保数据完整性的唯一索引
CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_users_tenant_openid_unique 
ON users (tenant_id, open_id) WHERE deleted_at IS NULL AND open_id IS NOT NULL;

-- ==================== 部分索引（节省空间） ====================
-- 只为活跃记录创建索引
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_conversations 
ON conversations (user_id, updated_at DESC) 
WHERE status = 'active' AND deleted_at IS NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_datasets 
ON datasets (tenant_id, updated_at DESC) 
WHERE status = 'active' AND deleted_at IS NULL;

-- ==================== 索引维护建议 ====================
-- 定期重建索引以保持性能
-- REINDEX INDEX CONCURRENTLY idx_messages_conversation_created;

-- 监控索引使用情况
-- SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch 
-- FROM pg_stat_user_indexes ORDER BY idx_scan DESC;

-- 检查未使用的索引
-- SELECT schemaname, tablename, indexname, idx_scan 
-- FROM pg_stat_user_indexes WHERE idx_scan = 0;
