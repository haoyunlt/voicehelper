#!/bin/bash

# Cursor性能诊断脚本 - v1.9.0
# 用于诊断和优化Cursor响应速度问题

echo "🔍 Cursor性能诊断报告"
echo "===================="
echo "时间: $(date)"
echo "项目: VoiceHelper v1.9.0"
echo ""

# 1. 项目基本信息
echo "📊 项目基本信息"
echo "----------------"
echo "项目大小: $(du -sh . 2>/dev/null | cut -f1)"
echo "总文件数: $(find . -type f | wc -l | tr -d ' ')"
echo "源代码文件数: $(find . -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.go" -o -name "*.md" | wc -l | tr -d ' ')"
echo ""

# 2. 大文件检查
echo "📁 大文件检查 (>500KB)"
echo "--------------------"
large_files=$(find . -type f -size +500k 2>/dev/null | head -10)
if [ -z "$large_files" ]; then
    echo "✅ 未发现大文件"
else
    echo "⚠️  发现大文件:"
    find . -type f -size +500k -exec ls -lh {} \; 2>/dev/null | head -10 | awk '{print $5, $9}'
fi
echo ""

# 3. 目录大小分析
echo "📂 目录大小分析 (前10大)"
echo "----------------------"
du -sh */ 2>/dev/null | sort -hr | head -10
echo ""

# 4. 忽略规则效果检查
echo "🚫 .cursorignore效果检查"
echo "----------------------"
echo ".cursorignore规则数: $(wc -l .cursorignore 2>/dev/null | cut -d' ' -f1)"

# 检查是否有应该被忽略但未被忽略的目录
echo ""
echo "检查常见构建目录:"
for dir in node_modules __pycache__ dist build coverage .next; do
    if [ -d "$dir" ]; then
        echo "⚠️  发现 $dir/ 目录 ($(du -sh $dir 2>/dev/null | cut -f1))"
    else
        echo "✅ 未发现 $dir/ 目录"
    fi
done
echo ""

# 5. Git仓库状态
echo "📋 Git仓库状态"
echo "--------------"
echo "提交数量: $(git rev-list --count HEAD 2>/dev/null || echo "未知")"
echo "分支数量: $(git branch -a 2>/dev/null | wc -l | tr -d ' ')"
echo "未跟踪文件: $(git status --porcelain 2>/dev/null | grep '^??' | wc -l | tr -d ' ')"
echo "Git仓库大小: $(du -sh .git 2>/dev/null | cut -f1)"
echo ""

# 6. 文件类型统计
echo "📄 文件类型统计"
echo "--------------"
echo "Python文件: $(find . -name "*.py" | wc -l | tr -d ' ')"
echo "JavaScript/TypeScript: $(find . -name "*.js" -o -name "*.ts" -o -name "*.tsx" | wc -l | tr -d ' ')"
echo "Go文件: $(find . -name "*.go" | wc -l | tr -d ' ')"
echo "Markdown文件: $(find . -name "*.md" | wc -l | tr -d ' ')"
echo "配置文件: $(find . -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.toml" | wc -l | tr -d ' ')"
echo ""

# 7. 最大文件检查
echo "📏 最大源代码文件 (前5)"
echo "--------------------"
echo "Python文件:"
find . -name "*.py" -exec wc -l {} \; 2>/dev/null | sort -nr | head -3 | while read lines file; do
    echo "  $lines 行: $file"
done

echo "JavaScript/TypeScript文件:"
find . -name "*.js" -o -name "*.ts" -o -name "*.tsx" -exec wc -l {} \; 2>/dev/null | sort -nr | head -3 | while read lines file; do
    echo "  $lines 行: $file"
done

echo "Markdown文件:"
find . -name "*.md" -exec wc -l {} \; 2>/dev/null | sort -nr | head -3 | while read lines file; do
    echo "  $lines 行: $file"
done
echo ""

# 8. 性能建议
echo "💡 性能优化建议"
echo "--------------"

# 检查项目大小
project_size_mb=$(du -sm . 2>/dev/null | cut -f1)
if [ "$project_size_mb" -gt 100 ]; then
    echo "⚠️  项目较大 (${project_size_mb}MB)，建议:"
    echo "   - 检查是否有不必要的大文件"
    echo "   - 优化.cursorignore规则"
    echo "   - 考虑使用Git LFS管理大文件"
else
    echo "✅ 项目大小适中 (${project_size_mb}MB)"
fi

# 检查文件数量
total_files=$(find . -type f | wc -l)
if [ "$total_files" -gt 1000 ]; then
    echo "⚠️  文件数量较多 ($total_files)，建议:"
    echo "   - 检查.cursorignore是否正确排除构建产物"
    echo "   - 清理不必要的临时文件"
else
    echo "✅ 文件数量适中 ($total_files)"
fi

# 检查源代码文件大小
echo ""
echo "大型源代码文件建议:"
find . -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.go" | while read file; do
    lines=$(wc -l < "$file" 2>/dev/null)
    if [ "$lines" -gt 800 ]; then
        echo "⚠️  $file ($lines 行) - 考虑拆分为多个文件"
    fi
done

echo ""
echo "🎯 推荐操作:"
echo "1. 重启Cursor以应用新的.cursorignore规则"
echo "2. 清理项目中的临时文件和缓存"
echo "3. 检查Cursor设置中的索引配置"
echo "4. 如果问题持续，考虑禁用部分Cursor功能"
echo ""

echo "✅ 诊断完成！"
