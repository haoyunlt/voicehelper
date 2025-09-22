# 优化后的项目目录结构

## 整体架构设计理念

基于**分层架构 + 平台化**的设计思想，将项目重新组织为：
- **Core Services**：核心业务服务（backend、algo）
- **Platforms**：各平台客户端（web、mobile、desktop、admin、browser-extension）
- **Shared**：共享资源（SDKs、类型定义、配置）
- **Tools**：开发工具（脚本、部署、测试）

## 优化后的目录结构

```
voicehelper/
├── core/                           # 核心服务层
│   ├── backend/                    # Go / Gin 网关服务
│   │   ├── cmd/
│   │   │   └── gateway/
│   │   │       └── main.go         # 网关入口
│   │   ├── internal/
│   │   │   ├── handler/            # Chat/Voice/Agent handlers
│   │   │   ├── middleware/         # Auth/RateLimit/Tracing
│   │   │   ├── ssews/              # SSE/WS封装
│   │   │   ├── contracts/          # DTO/Envelope/Error
│   │   │   └── service/            # 调用algo服务的client
│   │   ├── pkg/                    # 公共包
│   │   ├── go.mod
│   │   └── Dockerfile
│   └── algo/                       # Python / FastAPI / LangGraph
│       ├── app/
│       │   ├── api.py              # /query /voice /agent /ingest
│       │   ├── main.py
│       │   └── v2_api.py
│       ├── core/
│       │   ├── base/               # 抽象父类 & Mixins
│       │   ├── graph/              # LangGraph图/节点
│       │   ├── tools/              # LangChain工具子类
│       │   ├── asr_tts/            # ASR/TTS 适配器子类
│       │   ├── rag/                # Retriever/Chunker/Embedder
│       │   └── memory/             # 会话/检查点
│       ├── adapters/               # 具体供应商接入
│       ├── services/               # 业务服务
│       ├── requirements.txt
│       └── Dockerfile
│
├── platforms/                      # 平台客户端层
│   ├── web/                        # Web前端 (Next.js)
│   │   ├── src/
│   │   │   ├── api/                # TS SDK (SSE/WS)
│   │   │   ├── audio/              # AudioWorklet & players
│   │   │   ├── pages/              # 页面组件
│   │   │   └── components/         # React组件
│   │   ├── developer-portal/       # 开发者门户
│   │   │   ├── app/
│   │   │   ├── package.json
│   │   │   └── Dockerfile
│   │   ├── package.json
│   │   └── Dockerfile
│   ├── admin/                      # 管理后台 (Flask)
│   │   ├── app.py                  # Flask应用
│   │   ├── templates/              # HTML模板
│   │   ├── static/                 # 静态资源
│   │   └── Dockerfile
│   ├── mobile/                     # 移动端
│   │   ├── android/                # Android应用
│   │   │   └── app/src/main/java/ai/voicehelper/
│   │   ├── ios/                    # iOS应用
│   │   │   └── VoiceHelper/
│   │   ├── src/                    # React Native共享代码
│   │   └── package.json
│   ├── desktop/                    # 桌面端 (Electron)
│   │   ├── src/
│   │   │   ├── main/               # 主进程
│   │   │   └── common/             # 共享代码
│   │   └── package.json
│   └── browser-extension/          # 浏览器插件
│       ├── src/
│       │   └── content/
│       └── manifest.json
│
├── shared/                         # 共享资源层
│   ├── sdks/                       # 客户端SDK
│   │   ├── javascript/             # JS/TS SDK
│   │   │   ├── src/
│   │   │   ├── package.json
│   │   │   └── README.md
│   │   └── python/                 # Python SDK
│   │       ├── voicehelper/
│   │       ├── setup.py
│   │       └── README.md
│   ├── types/                      # 类型定义
│   │   ├── api.d.ts                # API类型
│   │   ├── events.d.ts             # 事件类型
│   │   └── common.d.ts             # 通用类型
│   └── configs/                    # 配置文件
│       ├── env.example             # 环境变量示例
│       ├── env.unified             # 统一配置
│       └── docker-compose.yml      # 本地开发环境
│
├── tools/                          # 开发工具层
│   ├── scripts/                    # 开发脚本
│   │   ├── build.sh                # 构建脚本
│   │   ├── deploy_v2_architecture.sh
│   │   ├── setup_unified_env.sh
│   │   ├── validate_performance.py
│   │   └── ...
│   ├── deployment/                 # 部署工具
│   │   ├── compose/                # Docker Compose配置
│   │   ├── k8s/                    # Kubernetes配置
│   │   ├── helm/                   # Helm Charts
│   │   ├── monitoring/             # 监控配置
│   │   └── scripts/                # 部署脚本
│   └── testing/                    # 测试工具
│       ├── e2e/                    # 端到端测试
│       ├── integration/            # 集成测试
│       ├── performance/            # 性能测试
│       └── fixtures/               # 测试数据
│
├── docs/                           # 文档
│   ├── api/                        # API文档
│   │   └── openapi.yaml
│   ├── architecture/               # 架构文档
│   │   ├── design-latest.md
│   │   └── ARCHITECTURE_DEEP_DIVE.md
│   ├── guides/                     # 使用指南
│   │   ├── QUICK_START.md
│   │   ├── DEPLOYMENT_GUIDE.md
│   │   └── DEVELOPMENT_GUIDE.md
│   └── references/                 # 参考文档
│       ├── TROUBLESHOOTING_GUIDE.md
│       └── BEST_PRACTICES.md
│
├── common/                         # 通用工具（Go）
│   ├── errors/
│   └── logger/
│
├── README.md                       # 项目说明
├── Makefile                        # 构建工具
├── .gitignore                      # Git忽略文件
└── LICENSE                         # 许可证
```

## 优化亮点

### 1. 清晰的分层架构
- **Core**：核心业务逻辑，独立部署
- **Platforms**：各平台客户端，按平台组织
- **Shared**：共享资源，避免重复
- **Tools**：开发工具，统一管理

### 2. 平台化组织
- 所有客户端统一放在 `platforms/` 下
- 每个平台保持独立的构建和部署
- 便于新平台的扩展

### 3. 共享资源管理
- SDKs统一维护，版本同步
- 类型定义共享，保证一致性
- 配置文件集中管理

### 4. 工具链整合
- 脚本工具统一放置
- 部署工具独立组织
- 测试工具分类管理

## 迁移指南

### 阶段1：创建新结构
```bash
# 创建新的目录结构
mkdir -p core/{backend,algo}
mkdir -p platforms/{web,admin,mobile,desktop,browser-extension}
mkdir -p shared/{sdks,types,configs}
mkdir -p tools/{scripts,deployment,testing}
```

### 阶段2：迁移文件
```bash
# 迁移核心服务
mv backend/* core/backend/
mv algo/* core/algo/

# 迁移平台代码
mv frontend/* platforms/web/
mv admin/* platforms/admin/
mv mobile/* platforms/mobile/
mv desktop/* platforms/desktop/
mv browser-extension/* platforms/browser-extension/
mv developer-portal/* platforms/web/developer-portal/

# 迁移共享资源
mv sdks/* shared/sdks/
mv env.* shared/configs/

# 迁移工具
mv scripts/* tools/scripts/
mv deploy/* tools/deployment/
mv tests/* tools/testing/
```

### 阶段3：更新配置
- 更新各平台的构建脚本
- 调整Docker Compose配置
- 修改CI/CD流水线
- 更新文档引用

## 配置文件调整

### Docker Compose
```yaml
# shared/configs/docker-compose.yml
version: '3.8'
services:
  backend:
    build: ../../core/backend
    ports:
      - "8080:8080"
  
  algo:
    build: ../../core/algo
    ports:
      - "8000:8000"
  
  web:
    build: ../../platforms/web
    ports:
      - "3000:3000"
  
  admin:
    build: ../../platforms/admin
    ports:
      - "5001:5001"
```

### Makefile
```makefile
# 根目录 Makefile
.PHONY: build-core build-platforms build-all

build-core:
	cd core/backend && go build
	cd core/algo && pip install -r requirements.txt

build-platforms:
	cd platforms/web && npm install && npm run build
	cd platforms/admin && pip install -r requirements.txt

build-all: build-core build-platforms

deploy:
	cd tools/deployment && ./deploy.sh

test:
	cd tools/testing && ./run-all-tests.sh
```

## 优势总结

1. **模块化清晰**：每个层级职责明确，便于维护
2. **扩展性强**：新增平台或工具都有明确位置
3. **复用性好**：共享资源避免重复开发
4. **管理便捷**：工具链统一，操作标准化
5. **团队协作**：不同团队可专注不同层级

这种结构特别适合多平台、多团队的大型项目，既保持了代码的组织性，又提供了良好的扩展性。
