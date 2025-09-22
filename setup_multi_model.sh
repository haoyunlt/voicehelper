#!/bin/bash

# 多模型配置部署脚本
# 用于快速配置和部署国内大模型服务

set -e

echo "🚀 VoiceHelper 多模型配置部署脚本"
echo "=================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 函数：打印彩色消息
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# 检查必要的命令
check_requirements() {
    print_info "检查系统要求..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装，请先安装 Docker"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose 未安装，请先安装 Docker Compose"
        exit 1
    fi
    
    print_success "系统要求检查通过"
}

# 显示模型选择菜单
show_model_menu() {
    echo ""
    print_info "请选择您要配置的大模型方案："
    echo ""
    echo "1) 🥇 成本优化方案 (推荐)"
    echo "   - 主模型: 豆包 Lite (0.9元/百万tokens)"
    echo "   - 备用: GLM-4 Flash (0.2元/百万tokens)"
    echo "   - 适合: 个人项目、初创公司"
    echo ""
    echo "2) 🥈 性能平衡方案 (推荐)"
    echo "   - 主模型: 豆包 Pro (2.8元/百万tokens)"
    echo "   - 备用: GLM-4 Flash (0.2元/百万tokens)"
    echo "   - 适合: 中小企业、生产环境"
    echo ""
    echo "3) 🥉 企业全功能方案"
    echo "   - 主模型: 豆包 Pro"
    echo "   - 备用: 通义千问 + GLM-4"
    echo "   - 适合: 大企业、高可用需求"
    echo ""
    echo "4) 🔧 自定义配置"
    echo "   - 手动选择模型组合"
    echo ""
    echo "0) 退出"
    echo ""
}

# 配置成本优化方案
setup_cost_optimized() {
    print_info "配置成本优化方案..."
    
    # 复制统一配置文件
    cp env.example .env
    
    # 更新为成本优化配置
    sed -i.bak 's/PRIMARY_MODEL=doubao-pro-4k/PRIMARY_MODEL=doubao-lite-4k/' .env
    sed -i.bak 's/ARK_MODEL=doubao-pro-4k/ARK_MODEL=doubao-lite-4k/' .env
    rm -f .env.bak
    
    print_success "成本优化方案配置完成"
    print_warning "请编辑 .env 文件，填入真实的 API 密钥"
}

# 配置性能平衡方案
setup_balanced() {
    print_info "配置性能平衡方案..."
    
    # 复制统一配置文件 (默认就是性能平衡方案)
    cp env.example .env
    
    print_success "性能平衡方案配置完成"
    print_warning "请编辑 .env 文件，填入真实的 API 密钥"
}

# 配置企业方案
setup_enterprise() {
    print_info "配置企业全功能方案..."
    
    # 复制统一配置文件 (已包含所有企业级配置)
    cp env.example .env
    
    print_success "企业全功能方案配置完成"
    print_info "统一配置文件已包含所有模型配置，请根据需要填入相应的 API 密钥"
    print_warning "请编辑 .env 文件，填入真实的 API 密钥"
}

# 自定义配置
setup_custom() {
    print_info "开始自定义配置..."
    
    # 复制统一配置文件
    cp env.example .env
    
    print_success "已复制统一配置文件到 .env"
    print_info "配置文件包含所有可用模型，请根据需要编辑并启用所需的模型"
    print_warning "请编辑 .env 文件，填入相应的 API 密钥"
}

# 测试API连接
test_api_connections() {
    print_info "测试 API 连接..."
    
    if [ ! -f .env ]; then
        print_error ".env 文件不存在，请先运行配置"
        return 1
    fi
    
    # 加载环境变量
    source .env
    
    # 重启算法服务
    print_info "重启算法服务..."
    docker-compose -f docker-compose.local.yml build algo-service
    docker-compose -f docker-compose.local.yml up algo-service -d
    
    # 等待服务启动
    print_info "等待服务启动..."
    sleep 10
    
    # 测试健康检查
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "算法服务启动成功"
    else
        print_error "算法服务启动失败"
        return 1
    fi
    
    # 测试模型调用
    print_info "测试模型调用..."
    
    response=$(curl -s -X POST "http://localhost:8000/query" \
        -H "Content-Type: application/json" \
        -d '{
            "messages": [{"role": "user", "content": "你好，请简单回复"}],
            "dataset_id": "test",
            "max_tokens": 50
        }')
    
    if echo "$response" | grep -q "choices"; then
        print_success "模型调用测试成功"
        echo "响应: $response"
    else
        print_warning "模型调用测试失败，可能需要配置有效的 API 密钥"
        echo "响应: $response"
    fi
}

# 显示配置指南
show_api_guide() {
    echo ""
    print_info "API 密钥获取指南："
    echo ""
    echo "🔑 豆包大模型 (推荐首选):"
    echo "   1. 访问: https://console.volcengine.com/"
    echo "   2. 注册/登录火山引擎账户"
    echo "   3. 开通豆包大模型服务"
    echo "   4. 在 API 管理页面创建密钥"
    echo ""
    echo "🔑 GLM-4 (推荐备用):"
    echo "   1. 访问: https://open.bigmodel.cn/"
    echo "   2. 注册智谱AI账户"
    echo "   3. 在控制台创建 API Key"
    echo ""
    echo "🔑 通义千问 (企业选择):"
    echo "   1. 访问: https://dashscope.console.aliyun.com/"
    echo "   2. 开通阿里云账户和DashScope服务"
    echo "   3. 创建 API Key"
    echo ""
    echo "🔑 文心一言:"
    echo "   1. 访问: https://console.bce.baidu.com/qianfan/"
    echo "   2. 开通百度智能云千帆服务"
    echo "   3. 获取 Access Token"
    echo ""
}

# 显示成本对比
show_cost_comparison() {
    echo ""
    print_info "成本对比 (每百万 tokens):"
    echo ""
    printf "%-20s %-15s %-15s %-15s\n" "模型" "输入成本" "输出成本" "总成本(1:1)"
    echo "================================================================"
    printf "%-20s %-15s %-15s %-15s\n" "GLM-4 Flash" "0.1元" "0.1元" "0.2元 🏆"
    printf "%-20s %-15s %-15s %-15s\n" "豆包 Lite" "0.3元" "0.6元" "0.9元"
    printf "%-20s %-15s %-15s %-15s\n" "豆包 Pro" "0.8元" "2.0元" "2.8元 ⭐"
    printf "%-20s %-15s %-15s %-15s\n" "文心 Lite" "0.8元" "2.0元" "2.8元"
    printf "%-20s %-15s %-15s %-15s\n" "混元 Lite" "1.0元" "2.0元" "3.0元"
    printf "%-20s %-15s %-15s %-15s\n" "通义千问 Turbo" "2.0元" "6.0元" "8.0元"
    echo ""
    print_info "🏆 = 最便宜  ⭐ = 推荐 (性价比最佳)"
}

# 主菜单循环
main_menu() {
    while true; do
        show_model_menu
        read -p "请选择 (0-4): " choice
        
        case $choice in
            1)
                setup_cost_optimized
                break
                ;;
            2)
                setup_balanced
                break
                ;;
            3)
                setup_enterprise
                break
                ;;
            4)
                setup_custom
                break
                ;;
            0)
                print_info "退出配置"
                exit 0
                ;;
            *)
                print_error "无效选择，请重新输入"
                ;;
        esac
    done
}

# 主函数
main() {
    check_requirements
    
    echo ""
    print_info "欢迎使用 VoiceHelper 多模型配置工具！"
    
    # 显示成本对比
    show_cost_comparison
    
    # 主菜单
    main_menu
    
    # 显示API获取指南
    show_api_guide
    
    # 询问是否测试
    echo ""
    read -p "是否现在测试 API 连接？(y/N): " test_choice
    if [[ $test_choice =~ ^[Yy]$ ]]; then
        test_api_connections
    fi
    
    echo ""
    print_success "配置完成！"
    print_info "下一步："
    echo "  1. 编辑 .env 文件，填入真实的 API 密钥"
    echo "  2. 运行: docker-compose -f docker-compose.local.yml up -d"
    echo "  3. 测试: curl http://localhost:8000/health"
    echo ""
    print_info "查看详细文档: docs/DOMESTIC_LLM_RESEARCH.md"
}

# 运行主函数
main "$@"
