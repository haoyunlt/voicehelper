-- Dify 数据库初始化脚本

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- 创建用户和数据库 (如果不存在)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'dify') THEN
        CREATE USER dify WITH PASSWORD 'dify123';
    END IF;
END
$$;

-- 授权
GRANT ALL PRIVILEGES ON DATABASE dify TO dify;
GRANT ALL ON SCHEMA public TO dify;

-- 创建基础表结构 (Dify会自动创建完整的表结构)
-- 这里只创建一些必要的初始化数据

-- 插入默认配置
INSERT INTO public.tenant (id, name, created_at, updated_at) 
VALUES ('default', 'Default Tenant', NOW(), NOW()) 
ON CONFLICT (id) DO NOTHING;

-- 创建索引优化
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_datasets_tenant_id ON datasets(tenant_id);

-- 设置默认权限
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO dify;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO dify;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON FUNCTIONS TO dify;
