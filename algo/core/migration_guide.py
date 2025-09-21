"""
代码合并迁移指南

本文件提供从旧API到新统一API的迁移指南和兼容性映射
"""

import warnings
from typing import Dict, Any, List, Optional


class DeprecationHelper:
    """弃用功能帮助器"""
    
    @staticmethod
    def warn_deprecated(old_class: str, new_class: str, version: str = "v1.9.0"):
        """发出弃用警告"""
        warnings.warn(
            f"{old_class} is deprecated since {version}. "
            f"Please use {new_class} from unified_utils instead.",
            DeprecationWarning,
            stacklevel=3
        )


# ==================== 迁移映射 ====================

MIGRATION_MAPPING = {
    # 相似度计算
    "semantic_cache.SimpleSimilarityCalculator": {
        "new_import": "from .unified_utils import get_similarity_calculator",
        "new_usage": "get_similarity_calculator().calculate_similarity(text1, text2, method='hybrid')",
        "old_usage": "SimpleSimilarityCalculator().calculate_similarity(text1, text2)"
    },
    
    # 内容标准化
    "semantic_cache.ContentNormalizer": {
        "new_import": "from .unified_utils import get_content_normalizer",
        "new_usage": "get_content_normalizer().normalize(content)",
        "old_usage": "ContentNormalizer().normalize(content)"
    },
    
    # 语义缓存
    "semantic_cache.SemanticCache": {
        "new_import": "from .unified_utils import get_cache_manager",
        "new_usage": "await get_cache_manager().get(content, model, parameters)",
        "old_usage": "await SemanticCache().get(content, model, parameters)"
    },
    
    # 请求合并
    "request_merger.RequestMerger": {
        "new_import": "from .unified_utils import get_request_processor",
        "new_usage": "get_request_processor().process_requests(requests)",
        "old_usage": "RequestMerger().merge_requests(requests)"
    },
    
    # 批处理去重
    "batch_processor.RequestDeduplicator": {
        "new_import": "from .unified_utils import get_request_processor",
        "new_usage": "get_request_processor().process_requests(unified_requests)",
        "old_usage": "RequestDeduplicator().deduplicate(requests)"
    },
    
    # 热点缓存
    "hotspot_cache.HotspotCache": {
        "new_import": "from .unified_utils import get_cache_manager",
        "new_usage": "get_cache_manager(enable_prewarming=True)",
        "old_usage": "HotspotCache()"
    },
    
    # 缓存预热
    "cache_prewarming.CachePrewarmer": {
        "new_import": "from .unified_utils import get_cache_manager",
        "new_usage": "get_cache_manager(enable_prewarming=True)",
        "old_usage": "CachePrewarmer()"
    },
    
    # 语音处理
    "voice.VoiceService": {
        "new_import": "from .unified_voice import get_voice_service",
        "new_usage": "get_voice_service(retrieve_service, config)",
        "old_usage": "VoiceService(retrieve_service)"
    },
    
    # 语音优化器
    "voice_optimizer.VoiceLatencyOptimizer": {
        "new_import": "from .unified_voice import get_voice_service, VoiceConfig, VoiceProcessingMode",
        "new_usage": "get_voice_service(config=VoiceConfig(mode=VoiceProcessingMode.OPTIMIZED))",
        "old_usage": "VoiceLatencyOptimizer(config)"
    },
    
    # 增强语音优化器
    "enhanced_voice_optimizer.EnhancedVoiceOptimizer": {
        "new_import": "from .unified_voice import get_voice_service, VoiceConfig, VoiceProcessingMode",
        "new_usage": "get_voice_service(config=VoiceConfig(mode=VoiceProcessingMode.ENHANCED))",
        "old_usage": "EnhancedVoiceOptimizer(config)"
    }
}


# ==================== 兼容性包装器 ====================

def create_compatibility_wrapper(old_class_name: str, new_implementation):
    """创建兼容性包装器"""
    
    class CompatibilityWrapper:
        def __init__(self, *args, **kwargs):
            DeprecationHelper.warn_deprecated(
                old_class_name, 
                "unified_utils or unified_voice"
            )
            self._impl = new_implementation(*args, **kwargs)
        
        def __getattr__(self, name):
            return getattr(self._impl, name)
    
    return CompatibilityWrapper


# ==================== 迁移助手函数 ====================

def generate_migration_script(source_files: List[str]) -> str:
    """生成迁移脚本"""
    
    script_lines = [
        "#!/usr/bin/env python3",
        "# 自动生成的迁移脚本",
        "# 将旧API调用替换为新的统一API",
        "",
        "import re",
        "import os",
        "from typing import List",
        "",
        "def migrate_file(file_path: str):",
        "    \"\"\"迁移单个文件\"\"\"",
        "    with open(file_path, 'r', encoding='utf-8') as f:",
        "        content = f.read()",
        "    ",
        "    original_content = content",
        "    ",
    ]
    
    # 添加替换规则
    for old_api, migration_info in MIGRATION_MAPPING.items():
        old_usage = migration_info["old_usage"]
        new_usage = migration_info["new_usage"]
        new_import = migration_info["new_import"]
        
        script_lines.extend([
            f"    # 替换 {old_api}",
            f"    if '{old_usage}' in content:",
            f"        content = content.replace('{old_usage}', '{new_usage}')",
            f"        # 添加新的导入",
            f"        if '{new_import}' not in content:",
            f"            content = '{new_import}\\n' + content",
            "    ",
        ])
    
    script_lines.extend([
        "    # 写回文件",
        "    if content != original_content:",
        "        with open(file_path, 'w', encoding='utf-8') as f:",
        "            f.write(content)",
        "        print(f'Migrated: {file_path}')",
        "    else:",
        "        print(f'No changes: {file_path}')",
        "",
        "def main():",
        "    \"\"\"主函数\"\"\"",
        f"    files = {source_files}",
        "    ",
        "    for file_path in files:",
        "        if os.path.exists(file_path):",
        "            migrate_file(file_path)",
        "        else:",
        "            print(f'File not found: {file_path}')",
        "",
        "if __name__ == '__main__':",
        "    main()"
    ])
    
    return "\n".join(script_lines)


def print_migration_guide():
    """打印迁移指南"""
    
    print("🔄 代码合并迁移指南")
    print("=" * 50)
    print()
    
    print("📋 概述")
    print("为了减少代码重复和提高维护性，我们将相似功能合并到统一的工具类中。")
    print("以下是主要的API变更：")
    print()
    
    for i, (old_api, migration_info) in enumerate(MIGRATION_MAPPING.items(), 1):
        print(f"{i}. {old_api}")
        print(f"   旧用法: {migration_info['old_usage']}")
        print(f"   新用法: {migration_info['new_usage']}")
        print(f"   导入: {migration_info['new_import']}")
        print()
    
    print("🚀 迁移步骤")
    print("1. 更新导入语句")
    print("2. 替换API调用")
    print("3. 测试功能正常")
    print("4. 删除旧的导入")
    print()
    
    print("💡 注意事项")
    print("- 新API提供更好的性能和功能")
    print("- 旧API仍然可用但会显示弃用警告")
    print("- 建议尽快迁移到新API")
    print("- 统一API提供更一致的接口")


def check_deprecated_usage(file_path: str) -> List[Dict[str, Any]]:
    """检查文件中的弃用用法"""
    
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for old_api, migration_info in MIGRATION_MAPPING.items():
                old_usage = migration_info["old_usage"]
                
                # 简化的匹配检查
                if any(part in line for part in old_usage.split('.')):
                    issues.append({
                        'file': file_path,
                        'line': line_num,
                        'content': line.strip(),
                        'old_api': old_api,
                        'suggestion': migration_info["new_usage"]
                    })
    
    except Exception as e:
        print(f"Error checking {file_path}: {e}")
    
    return issues


# ==================== 命令行工具 ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print_migration_guide()
    elif sys.argv[1] == "check":
        if len(sys.argv) < 3:
            print("Usage: python migration_guide.py check <file_path>")
            sys.exit(1)
        
        file_path = sys.argv[2]
        issues = check_deprecated_usage(file_path)
        
        if issues:
            print(f"🔍 发现 {len(issues)} 个需要迁移的用法:")
            for issue in issues:
                print(f"  {issue['file']}:{issue['line']} - {issue['old_api']}")
                print(f"    当前: {issue['content']}")
                print(f"    建议: {issue['suggestion']}")
                print()
        else:
            print("✅ 未发现需要迁移的用法")
    
    elif sys.argv[1] == "generate":
        if len(sys.argv) < 3:
            print("Usage: python migration_guide.py generate <file1> [file2] ...")
            sys.exit(1)
        
        source_files = sys.argv[2:]
        script = generate_migration_script(source_files)
        
        with open("migrate_to_unified_api.py", "w", encoding="utf-8") as f:
            f.write(script)
        
        print("✅ 迁移脚本已生成: migrate_to_unified_api.py")
        print("运行: python migrate_to_unified_api.py")
    
    else:
        print_migration_guide()
