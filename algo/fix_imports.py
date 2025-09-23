#!/usr/bin/env python3
"""
算法服务模块导入路径修复脚本
解决相对导入和PYTHONPATH问题
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_python_path():
    """设置正确的Python路径"""
    # 获取当前脚本目录
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent
    
    # 添加到Python路径
    paths_to_add = [
        str(current_dir),  # algo目录
        str(project_root), # 项目根目录
    ]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    # 设置环境变量
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    new_paths = [p for p in paths_to_add if p not in current_pythonpath]
    
    if new_paths:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = ':'.join(new_paths + [current_pythonpath])
        else:
            os.environ['PYTHONPATH'] = ':'.join(new_paths)
    
    print(f"✅ Python路径已设置:")
    for i, path in enumerate(sys.path[:5]):
        print(f"  {i+1}. {path}")
    
    print(f"✅ PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

def check_imports():
    """检查关键模块导入"""
    print("\n🔍 检查模块导入...")
    
    test_imports = [
        ('core.base', 'StreamCallback'),
        ('core.rag.bge_faiss_retriever', 'BGEFaissRetriever'),
        ('core.graph.chat_voice', 'ChatVoiceAgentGraph'),
        ('core.tools', 'FetchTool'),
        ('core.asr_tts.openai', 'OpenAIAsrAdapter'),
        ('common.logger', None),
        ('common.errors', None),
    ]
    
    success_count = 0
    for module_name, class_name in test_imports:
        try:
            module = __import__(module_name, fromlist=[class_name] if class_name else [])
            if class_name and hasattr(module, class_name):
                print(f"  ✅ {module_name}.{class_name}")
            elif not class_name:
                print(f"  ✅ {module_name}")
            else:
                print(f"  ❌ {module_name}.{class_name} - 类不存在")
                continue
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {module_name} - {e}")
        except Exception as e:
            print(f"  ⚠️  {module_name} - {e}")
    
    print(f"\n📊 导入检查结果: {success_count}/{len(test_imports)} 成功")
    return success_count == len(test_imports)

def create_init_files():
    """创建缺失的__init__.py文件"""
    print("\n📁 创建__init__.py文件...")
    
    current_dir = Path(__file__).parent
    
    # 需要__init__.py的目录
    directories = [
        current_dir / 'core',
        current_dir / 'core' / 'base',
        current_dir / 'core' / 'rag',
        current_dir / 'core' / 'graph',
        current_dir / 'core' / 'tools',
        current_dir / 'core' / 'asr_tts',
        current_dir / 'core' / 'config',
        current_dir / 'core' / 'memory',
        current_dir / 'common',
        current_dir / 'adapters',
        current_dir / 'services',
        current_dir / 'reasoning',
        current_dir / 'app',
        current_dir / 'tests',
    ]
    
    created_count = 0
    for directory in directories:
        if directory.exists() and directory.is_dir():
            init_file = directory / '__init__.py'
            if not init_file.exists():
                init_file.write_text('# Auto-generated __init__.py\n')
                print(f"  ✅ 创建 {init_file.relative_to(current_dir)}")
                created_count += 1
            else:
                print(f"  ✓ 已存在 {init_file.relative_to(current_dir)}")
    
    print(f"\n📊 创建了 {created_count} 个__init__.py文件")

def fix_import_statements():
    """修复导入语句"""
    print("\n🔧 检查导入语句...")
    
    # 这里可以添加自动修复导入语句的逻辑
    # 目前只是检查和报告
    
    current_dir = Path(__file__).parent
    python_files = list(current_dir.rglob('*.py'))
    
    problematic_files = []
    
    for py_file in python_files:
        if py_file.name == __file__.split('/')[-1]:  # 跳过当前脚本
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 检查相对导入
            if 'from .' in content or 'import .' in content:
                problematic_files.append(py_file)
                
        except Exception as e:
            print(f"  ⚠️  无法读取 {py_file}: {e}")
    
    if problematic_files:
        print(f"  ⚠️  发现 {len(problematic_files)} 个文件使用相对导入")
        for file in problematic_files[:5]:  # 只显示前5个
            print(f"    - {file.relative_to(current_dir)}")
        if len(problematic_files) > 5:
            print(f"    ... 还有 {len(problematic_files) - 5} 个文件")
    else:
        print("  ✅ 未发现相对导入问题")

def main():
    """主函数"""
    print("🚀 VoiceHelper 算法服务模块路径修复")
    print("=" * 50)
    
    # 1. 设置Python路径
    setup_python_path()
    
    # 2. 创建__init__.py文件
    create_init_files()
    
    # 3. 检查导入
    imports_ok = check_imports()
    
    # 4. 检查导入语句
    fix_import_statements()
    
    print("\n" + "=" * 50)
    if imports_ok:
        print("🎉 路径修复完成！所有导入都正常工作")
        print("\n📋 下一步:")
        print("1. 运行: source ./activate.sh")
        print("2. 启动服务: python app/v2_api.py")
    else:
        print("⚠️  路径修复完成，但部分导入仍有问题")
        print("请检查缺失的模块文件")
    
    return 0 if imports_ok else 1

if __name__ == '__main__':
    sys.exit(main())
