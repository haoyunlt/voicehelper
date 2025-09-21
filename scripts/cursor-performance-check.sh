#!/bin/bash

# Cursor性能检查脚本
# 用于诊断和监控Cursor的性能状态

echo "🔍 Cursor性能检查报告"
echo "===================="
echo "检查时间: $(date)"
echo ""

# 1. 项目基本信息
echo "📊 项目信息"
echo "----------"
echo "项目路径: $(pwd)"
echo "总文件数: $(find . -type f | wc -l | tr -d ' ')"
echo "项目大小: $(du -sh . | cut -f1)"
echo ""

# 2. 代码统计
echo "📝 代码统计"
echo "----------"
echo "Python文件: $(find . -name "*.py" | wc -l | tr -d ' ')个"
echo "Python代码行数: $(find . -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')行"
echo "Go文件: $(find . -name "*.go" | wc -l | tr -d ' ')个"
echo "TypeScript/JavaScript文件: $(find . -name "*.ts" -o -name "*.tsx" -o -name "*.js" -o -name "*.jsx" | wc -l | tr -d ' ')个"
echo ""

# 3. Git信息
echo "📚 Git信息"
echo "----------"
echo "提交数量: $(git log --oneline | wc -l | tr -d ' ')"
echo ".git目录大小: $(du -sh .git | cut -f1)"
echo ""

# 4. 大文件检查
echo "📁 大文件检查"
echo "------------"
large_files=$(find . -type f -size +100k | head -5)
if [ -z "$large_files" ]; then
    echo "✅ 无大于100KB的文件"
else
    echo "⚠️ 发现大文件:"
    find . -type f -size +100k -exec ls -lh {} + | head -5
fi
echo ""

# 5. Cursor进程检查
echo "🖥️ Cursor进程状态"
echo "----------------"
if pgrep -f "Cursor" > /dev/null; then
    echo "✅ Cursor正在运行"
    
    # 内存使用情况
    cursor_memory=$(ps aux | grep -i cursor | grep -v grep | awk '{sum += $6} END {printf "%.1f", sum/1024}')
    if [ ! -z "$cursor_memory" ]; then
        echo "内存使用: ${cursor_memory}MB"
    fi
    
    # CPU使用情况
    cursor_cpu=$(ps aux | grep -i cursor | grep -v grep | awk '{sum += $3} END {printf "%.1f", sum}')
    if [ ! -z "$cursor_cpu" ]; then
        echo "CPU使用: ${cursor_cpu}%"
    fi
else
    echo "❌ Cursor未运行"
fi
echo ""

# 6. 系统资源
echo "💻 系统资源"
echo "----------"
if command -v vm_stat > /dev/null; then
    # macOS
    free_memory=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
    if [ ! -z "$free_memory" ]; then
        free_mb=$((free_memory * 4096 / 1024 / 1024))
        echo "可用内存: ${free_mb}MB"
    fi
elif command -v free > /dev/null; then
    # Linux
    free_memory=$(free -m | grep "Mem:" | awk '{print $7}')
    echo "可用内存: ${free_memory}MB"
fi

# CPU负载
if command -v uptime > /dev/null; then
    load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    echo "系统负载: $load_avg"
fi
echo ""

# 7. 配置文件检查
echo "⚙️ 配置检查"
echo "----------"
if [ -f ".cursor/settings.json" ]; then
    echo "✅ Cursor设置文件存在"
else
    echo "⚠️ 未找到Cursor设置文件"
fi

if [ -f ".cursorignore" ]; then
    ignore_lines=$(wc -l < .cursorignore)
    echo "✅ .cursorignore存在 (${ignore_lines}行)"
else
    echo "⚠️ 未找到.cursorignore文件"
fi
echo ""

# 8. 性能建议
echo "💡 性能建议"
echo "----------"

# 检查项目大小
project_size_mb=$(du -sm . | cut -f1)
if [ $project_size_mb -gt 100 ]; then
    echo "⚠️ 项目较大 (${project_size_mb}MB)，建议优化.cursorignore"
else
    echo "✅ 项目大小适中 (${project_size_mb}MB)"
fi

# 检查文件数量
file_count=$(find . -type f | wc -l | tr -d ' ')
if [ $file_count -gt 1000 ]; then
    echo "⚠️ 文件数量较多 (${file_count}个)，建议排除不必要的文件"
else
    echo "✅ 文件数量适中 (${file_count}个)"
fi

# 检查Python代码行数
python_lines=$(find . -name "*.py" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')
if [ ! -z "$python_lines" ] && [ $python_lines -gt 50000 ]; then
    echo "⚠️ Python代码量较大 (${python_lines}行)，建议模块化"
else
    echo "✅ 代码量适中"
fi

echo ""
echo "🎯 优化建议"
echo "----------"
echo "1. 确保.cursorignore配置完整"
echo "2. 使用@符号精确指定文件"
echo "3. 避免一次性分析大型目录"
echo "4. 定期重启Cursor释放内存"
echo "5. 保持系统有足够可用内存(>4GB)"
echo ""

echo "✅ 检查完成"
echo "详细报告请查看: docs/CURSOR_PERFORMANCE_DIAGNOSIS.md"