package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"voicehelper/backend/pkg/discovery"
)

func main() {
	fmt.Println("🔍 服务发现功能演示")
	fmt.Println(strings.Repeat("=", 50))

	// 1. 基本服务发现演示
	demonstrateServiceDiscovery()

	// 2. 服务注册表演示
	demonstrateServiceRegistry()

	// 3. 负载均衡演示
	demonstrateLoadBalancing()

	// 4. 健康检查演示
	demonstrateHealthChecks()

	fmt.Println("\n✅ 服务发现功能演示完成")
}

// demonstrateServiceDiscovery 演示基本服务发现功能
func demonstrateServiceDiscovery() {
	fmt.Println("\n1. 🔍 服务发现演示")
	fmt.Println(strings.Repeat("-", 30))

	// 创建服务发现实例
	serviceDiscovery := discovery.NewServiceDiscovery()

	// 演示发现公开的API服务
	publicAPIs := []string{
		"https://httpbin.org",
		"https://jsonplaceholder.typicode.com",
	}

	for _, apiURL := range publicAPIs {
		fmt.Printf("正在发现服务: %s\n", apiURL)

		request := &discovery.DiscoveryRequest{
			URL:              apiURL,
			Timeout:          10 * time.Second,
			FollowRedirects:  true,
			DiscoveryMethods: []string{"openapi", "swagger", "health", "info", "root"},
		}

		ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
		result, err := serviceDiscovery.DiscoverService(ctx, request)
		cancel()

		if err != nil {
			fmt.Printf("  ❌ 发现失败: %v\n", err)
			continue
		}

		fmt.Printf("  ✅ 发现成功:\n")
		fmt.Printf("     服务名称: %s\n", result.ServiceName)
		fmt.Printf("     版本: %s\n", result.Version)
		fmt.Printf("     描述: %s\n", result.Description)
		fmt.Printf("     能力: %v\n", result.Capabilities)
		fmt.Printf("     端点数量: %d\n", len(result.Endpoints))
		fmt.Printf("     状态: %s\n", result.Status)

		if result.HealthCheck != nil {
			fmt.Printf("     健康检查: %s %s\n", result.HealthCheck.Method, result.HealthCheck.Path)
		}
	}

	// 演示缓存功能
	fmt.Println("\n缓存的服务:")
	cachedServices := serviceDiscovery.ListCachedServices()
	for i, serviceURL := range cachedServices {
		fmt.Printf("  %d. %s\n", i+1, serviceURL)
	}
}

// demonstrateServiceRegistry 演示服务注册表功能
func demonstrateServiceRegistry() {
	fmt.Println("\n2. 📋 服务注册表演示")
	fmt.Println(strings.Repeat("-", 30))

	// 创建服务注册表
	registry := discovery.NewServiceRegistry(&discovery.RegistryConfig{
		HealthCheckEnabled:  false, // 演示中禁用自动健康检查
		HealthCheckInterval: 30 * time.Second,
		HealthCheckTimeout:  5 * time.Second,
	})

	// 注册示例服务
	services := []*discovery.RegisteredService{
		{
			ID:           "user-service",
			Name:         "用户服务",
			URL:          "https://jsonplaceholder.typicode.com",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthHealthy,
			Weight:       100,
			Priority:     1,
			Tags:         []string{"users", "authentication"},
		},
		{
			ID:           "post-service",
			Name:         "文章服务",
			URL:          "https://jsonplaceholder.typicode.com",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthHealthy,
			Weight:       150,
			Priority:     2,
			Tags:         []string{"posts", "content"},
		},
		{
			ID:           "comment-service",
			Name:         "评论服务",
			URL:          "https://jsonplaceholder.typicode.com",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthUnhealthy,
			Weight:       80,
			Priority:     1,
			Tags:         []string{"comments", "social"},
		},
	}

	ctx := context.Background()
	for _, service := range services {
		err := registry.RegisterService(ctx, service)
		if err != nil {
			fmt.Printf("  ❌ 注册服务失败 %s: %v\n", service.ID, err)
		} else {
			fmt.Printf("  ✅ 注册服务成功: %s (%s)\n", service.Name, service.ID)
		}
	}

	// 列出所有服务
	fmt.Println("\n注册的服务列表:")
	allServices := registry.ListServices()
	for i, service := range allServices {
		fmt.Printf("  %d. %s (%s) - %s - 权重:%d 优先级:%d\n",
			i+1, service.Name, service.ID, service.HealthStatus, service.Weight, service.Priority)
	}

	// 按类别列出服务
	fmt.Println("\nAPI类别的服务:")
	apiServices := registry.ListServicesByCategory("api")
	for i, service := range apiServices {
		fmt.Printf("  %d. %s - %s\n", i+1, service.Name, service.HealthStatus)
	}

	// 列出健康的服务
	fmt.Println("\n健康的服务:")
	healthyServices := registry.ListHealthyServices()
	for i, service := range healthyServices {
		fmt.Printf("  %d. %s\n", i+1, service.Name)
	}

	// 更新服务
	fmt.Println("\n更新服务权重...")
	err := registry.UpdateService("user-service", map[string]interface{}{
		"weight": 200,
		"tags":   []string{"users", "authentication", "updated"},
	})
	if err != nil {
		fmt.Printf("  ❌ 更新失败: %v\n", err)
	} else {
		fmt.Printf("  ✅ 更新成功\n")
	}

	// 获取服务统计
	fmt.Println("\n服务统计:")
	for _, service := range allServices {
		// 模拟一些请求统计
		registry.RecordServiceRequest(service.ID, 100*time.Millisecond, true)
		registry.RecordServiceRequest(service.ID, 150*time.Millisecond, true)
		registry.RecordServiceRequest(service.ID, 200*time.Millisecond, false)

		stats, err := registry.GetServiceStats(service.ID)
		if err == nil {
			fmt.Printf("  %s: 请求数=%v, 错误数=%v, 平均延迟=%v\n",
				service.Name, stats["request_count"], stats["error_count"], stats["average_latency"])
		}
	}
}

// demonstrateLoadBalancing 演示负载均衡功能
func demonstrateLoadBalancing() {
	fmt.Println("\n3. ⚖️ 负载均衡演示")
	fmt.Println(strings.Repeat("-", 30))

	// 创建服务注册表和负载均衡器
	registry := discovery.NewServiceRegistry(nil)
	loadBalancer := discovery.NewLoadBalancer(registry, &discovery.LoadBalancerConfig{
		Strategy:             discovery.StrategyHealthyFirst,
		FailureThreshold:     3,
		RecoveryTimeout:      10 * time.Second,
		EnableCircuitBreaker: true,
	})

	// 注册测试服务
	testServices := []*discovery.RegisteredService{
		{
			ID:           "service-a",
			Name:         "服务A",
			URL:          "http://service-a:8080",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthHealthy,
			Weight:       100,
			Priority:     1,
		},
		{
			ID:           "service-b",
			Name:         "服务B",
			URL:          "http://service-b:8080",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthHealthy,
			Weight:       200,
			Priority:     2,
		},
		{
			ID:           "service-c",
			Name:         "服务C",
			URL:          "http://service-c:8080",
			Category:     "api",
			Status:       discovery.StatusActive,
			HealthStatus: discovery.HealthUnhealthy,
			Weight:       150,
			Priority:     1,
		},
	}

	ctx := context.Background()
	for _, service := range testServices {
		registry.RegisterService(ctx, service)
	}

	// 演示不同的负载均衡策略
	strategies := []discovery.LoadBalancingStrategy{
		discovery.StrategyHealthyFirst,
		discovery.StrategyWeightedRandom,
		discovery.StrategyPriority,
	}

	for _, strategy := range strategies {
		fmt.Printf("\n使用策略: %s\n", strategy)

		// 临时更改策略（在实际应用中应该重新创建负载均衡器）
		selections := make(map[string]int)

		// 进行多次选择以观察分布
		for i := 0; i < 10; i++ {
			selection, err := loadBalancer.SelectService("api", nil)
			if err != nil {
				fmt.Printf("  ❌ 选择失败: %v\n", err)
				continue
			}
			selections[selection.Service.Name]++
		}

		fmt.Println("  选择分布:")
		for serviceName, count := range selections {
			fmt.Printf("    %s: %d次\n", serviceName, count)
		}
	}

	// 演示熔断器功能
	fmt.Println("\n熔断器演示:")

	// 模拟服务A连续失败
	fmt.Println("  模拟服务A连续失败...")
	for i := 0; i < 5; i++ {
		loadBalancer.RecordFailure("service-a")
	}

	// 检查熔断器状态
	circuitStatus := loadBalancer.GetCircuitBreakerStatus()
	fmt.Printf("  熔断器状态: %v\n", circuitStatus)

	// 尝试选择服务（应该避开服务A）
	fmt.Println("  熔断器打开后的服务选择:")
	for i := 0; i < 5; i++ {
		selection, err := loadBalancer.SelectService("api", nil)
		if err != nil {
			fmt.Printf("    ❌ 选择失败: %v\n", err)
		} else {
			fmt.Printf("    选择了: %s (%s)\n", selection.Service.Name, selection.Reason)
		}
	}

	// 重置熔断器
	fmt.Println("  重置服务A的熔断器...")
	loadBalancer.ResetCircuitBreaker("service-a")

	// 再次检查状态
	circuitStatus = loadBalancer.GetCircuitBreakerStatus()
	fmt.Printf("  重置后熔断器状态: %v\n", circuitStatus)

	// 演示多服务选择
	fmt.Println("\n多服务选择演示（用于冗余）:")
	selections, err := loadBalancer.SelectMultipleServices("api", 2, nil)
	if err != nil {
		fmt.Printf("  ❌ 多服务选择失败: %v\n", err)
	} else {
		fmt.Printf("  选择了%d个服务:\n", len(selections))
		for i, selection := range selections {
			fmt.Printf("    %d. %s (%s)\n", i+1, selection.Service.Name, selection.Reason)
		}
	}
}

// demonstrateHealthChecks 演示健康检查功能
func demonstrateHealthChecks() {
	fmt.Println("\n4. 🏥 健康检查演示")
	fmt.Println(strings.Repeat("-", 30))

	// 创建一个启用健康检查的注册表
	registry := discovery.NewServiceRegistry(&discovery.RegistryConfig{
		HealthCheckEnabled:  true,
		HealthCheckInterval: 5 * time.Second,
		HealthCheckTimeout:  3 * time.Second,
	})

	// 注册一个真实的可访问服务
	healthyService := &discovery.RegisteredService{
		ID:           "httpbin-service",
		Name:         "HTTPBin服务",
		URL:          "https://httpbin.org",
		Category:     "api",
		Status:       discovery.StatusActive,
		HealthStatus: discovery.HealthUnknown,
	}

	ctx := context.Background()
	err := registry.RegisterService(ctx, healthyService)
	if err != nil {
		fmt.Printf("  ❌ 注册服务失败: %v\n", err)
		return
	}

	fmt.Printf("  ✅ 注册服务: %s\n", healthyService.Name)

	// 手动执行健康检查
	fmt.Println("\n执行手动健康检查...")
	health, err := registry.CheckServiceHealth(ctx, "httpbin-service")
	if err != nil {
		fmt.Printf("  ❌ 健康检查失败: %v\n", err)
	} else {
		fmt.Printf("  健康状态: %s\n", health)
	}

	// 等待自动健康检查运行
	fmt.Println("\n等待自动健康检查运行...")
	time.Sleep(6 * time.Second)

	// 检查健康服务列表
	healthyServices := registry.ListHealthyServices()
	fmt.Printf("  健康服务数量: %d\n", len(healthyServices))
	for _, service := range healthyServices {
		fmt.Printf("    - %s: %s (最后检查: %v)\n",
			service.Name, service.HealthStatus, service.LastHealthCheck.Format("15:04:05"))
	}

	// 获取服务统计
	stats, err := registry.GetServiceStats("httpbin-service")
	if err == nil {
		fmt.Println("\n服务统计:")
		statsJSON, _ := json.MarshalIndent(stats, "    ", "  ")
		fmt.Printf("    %s\n", string(statsJSON))
	}
}

// 创建一个简单的HTTP服务器用于测试
func startTestServer(port int, healthy bool) {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		if healthy {
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]string{"status": "healthy"})
		} else {
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(map[string]string{"status": "unhealthy"})
		}
	})

	mux.HandleFunc("/info", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"name":        fmt.Sprintf("Test Service %d", port),
			"version":     "1.0.0",
			"description": "Test service for discovery demo",
			"port":        port,
		})
	})

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, "Test Service %d is running", port)
	})

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: mux,
	}

	go func() {
		log.Printf("Test server starting on port %d (healthy: %v)", port, healthy)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("Test server error: %v", err)
		}
	}()
}
