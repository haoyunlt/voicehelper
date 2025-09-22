# Cursor 性能指南（诊断与优化）

本指南整合了原《CURSOR_PERFORMANCE_DIAGNOSIS.md》与《CURSOR_PERFORMANCE_OPTIMIZATION.md》，统一提供排查流程、推荐配置与验证方法，帮助你快速提升 Cursor 在本项目中的体验。

## 1. 快速诊断
- 现象：响应慢、补全卡顿、CPU/内存占用高。
- 常因：上下文范围过大、未忽略大目录、编辑器特性开太多、工作流不当。
- 自查：是否忽略了 `node_modules/`、`__pycache__/`、`dist/` 等？是否开启全局文件搜索？可用内存是否 > 4GB？

## 2. 必做配置
### 2.1 `.cursor/settings.json`
```json
{
  "cursor.ai.maxTokens": 4000,
  "cursor.ai.contextWindow": "medium",
  "cursor.ai.enableFileSearch": false,
  "cursor.ai.enableSymbolSearch": true,
  "editor.minimap.enabled": false,
  "editor.semanticHighlighting.enabled": false,
  "files.watcherExclude": {
    "**/.git/**": true,
    "**/node_modules/**": true,
    "**/__pycache__/**": true,
    "**/*.pyc": true,
    "**/dist/**": true,
    "**/build/**": true
  }
}
```

### 2.2 `.cursorignore`
```bash
*.pyc
*.pyo
*.pyd
__pycache__/
.pytest_cache/
.mypy_cache/
*.log
*.tmp

**/node_modules/**
frontend/.next/**
frontend/out/**
frontend/dist/**
frontend/build/**

**/coverage/**
**/.nyc_output/**

algo/*/cache/
algo/*/temp/
algo/*/models/

**/dist/**
**/build/**
```

## 3. 使用与工作流
- 用 `@文件路径` 精准限制上下文；避免一次性让 AI 扫描大目录。
- 复杂需求拆小步；对话 30-50 轮开新会话；每 2-3 小时重启 Cursor。
- 关闭无关应用，保持可用内存 > 4GB。

## 4. 监控与验证
- 系统/进程监控：关注 Cursor Helper 内存/CPU。
- 可运行 `scripts/cursor-performance-monitor.sh` 观察实时状态。
- 验证口径：响应时间目标 < 3s；Renderer 内存 < 2.5GB；CPU 峰值 < 80%。

## 5. 常见问答
- 仍卡顿：确认是否有未忽略的大型二进制或报告文件（如巨型 *.json）。
- 搜索变慢：确保已关闭全局文件搜索，并完善 `.cursorignore`。
- 回答跑偏：使用 `@文件` 精准限定上下文。

## 6. 预期收益（项目内实测）
- 响应速度 +30%~50%
- 内存占用 -20%~30%
- CPU 占用 -30%~40%
- 体验更流畅、补全更稳定

## 7. 参考
- 工程层性能验证：`tests/performance/optimization_validator.py`
- 本指南取代：`CURSOR_PERFORMANCE_DIAGNOSIS.md` 与 `CURSOR_PERFORMANCE_OPTIMIZATION.md`
