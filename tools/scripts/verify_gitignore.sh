#!/bin/bash

# Git忽略规则验证脚本
# 用于验证敏感文件和缓存目录是否被正确忽略

echo "🔍 验证 .gitignore 和 .cursorignore 规则..."
echo ""

# 测试文件列表
test_files=(
    ".env"
    ".env.local"
    ".env.backup.20241201"
    ".env.bak"
    "api_keys.txt"
    "credentials.json"
    "config.secret"
    "algo/data/vectors.pkl"
    "algo/storage/embeddings.h5"
    ".cache/huggingface/model.bin"
    "sentence_transformers_cache/model.pt"
    "temp_cleanup.sh"
    "remove_milvus_refs.sh"
    "config.milvus_backup"
)

echo "📋 测试忽略规则:"
ignored_count=0
total_count=${#test_files[@]}

for file in "${test_files[@]}"; do
    if git check-ignore "$file" >/dev/null 2>&1; then
        echo "  ✅ $file - 已忽略"
        ((ignored_count++))
    else
        echo "  ❌ $file - 未忽略"
    fi
done

echo ""
echo "📊 测试结果:"
echo "  总计: $total_count 个测试文件"
echo "  已忽略: $ignored_count 个"
echo "  未忽略: $((total_count - ignored_count)) 个"

if [ $ignored_count -eq $total_count ]; then
    echo ""
    echo "🎉 所有测试通过！.gitignore 规则配置正确。"
    exit 0
else
    echo ""
    echo "⚠️  部分文件未被忽略，请检查 .gitignore 配置。"
    exit 1
fi
