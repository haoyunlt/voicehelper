package errors

import (
	"fmt"
	"net/http"
)

// ErrorCode 错误码类型
type ErrorCode int

// 错误码定义 - 采用6位数字编码
// 格式: XYZABC
// X: 服务类型 (1:Gateway, 2:Auth, 3:Chat, 4:Voice, 5:RAG, 6:Storage, 7:Integration, 8:Monitor, 9:Common)
// Y: 模块类型 (0:通用, 1:API, 2:Service, 3:Database, 4:Cache, 5:Network, 6:File, 7:Config, 8:Security, 9:Performance)
// Z: 错误类型 (0:成功, 1:客户端错误, 2:服务端错误, 3:网络错误, 4:数据错误, 5:权限错误, 6:配置错误, 7:性能错误, 8:安全错误, 9:未知错误)
// ABC: 具体错误序号 (001-999)

const (
	// 成功码
	Success ErrorCode = 000000

	// ========== Gateway服务错误码 (1xxxxx) ==========
	// Gateway通用错误 (10xxxx)
	GatewayInternalError      ErrorCode = 102001 // Gateway内部错误
	GatewayServiceUnavailable ErrorCode = 102002 // Gateway服务不可用
	GatewayTimeout            ErrorCode = 102003 // Gateway超时

	// Gateway API错误 (11xxxx)
	GatewayInvalidRequest    ErrorCode = 111001 // 无效请求
	GatewayMissingParameter  ErrorCode = 111002 // 缺少参数
	GatewayInvalidParameter  ErrorCode = 111003 // 参数无效
	GatewayRequestTooLarge   ErrorCode = 111004 // 请求体过大
	GatewayRateLimitExceeded ErrorCode = 111005 // 请求频率超限

	// Gateway网络错误 (13xxxx)
	GatewayNetworkError     ErrorCode = 133001 // 网络错误
	GatewayConnectionFailed ErrorCode = 133002 // 连接失败
	GatewayDNSResolveFailed ErrorCode = 133003 // DNS解析失败

	// ========== 认证服务错误码 (2xxxxx) ==========
	// 认证通用错误 (20xxxx)
	AuthInternalError      ErrorCode = 202001 // 认证服务内部错误
	AuthServiceUnavailable ErrorCode = 202002 // 认证服务不可用

	// 认证API错误 (21xxxx)
	AuthInvalidCredentials ErrorCode = 211001 // 无效凭证
	AuthTokenExpired       ErrorCode = 211002 // Token过期
	AuthTokenInvalid       ErrorCode = 211003 // Token无效
	AuthPermissionDenied   ErrorCode = 211004 // 权限不足
	AuthUserNotFound       ErrorCode = 211005 // 用户不存在
	AuthUserDisabled       ErrorCode = 211006 // 用户已禁用

	// 认证安全错误 (28xxxx)
	AuthSecurityViolation  ErrorCode = 281001 // 安全违规
	AuthBruteForceDetected ErrorCode = 281002 // 检测到暴力破解
	AuthSuspiciousActivity ErrorCode = 281003 // 可疑活动

	// ========== 聊天服务错误码 (3xxxxx) ==========
	// 聊天通用错误 (30xxxx)
	ChatInternalError      ErrorCode = 302001 // 聊天服务内部错误
	ChatServiceUnavailable ErrorCode = 302002 // 聊天服务不可用

	// 聊天API错误 (31xxxx)
	ChatInvalidMessage       ErrorCode = 311001 // 无效消息
	ChatMessageTooLong       ErrorCode = 311002 // 消息过长
	ChatSessionNotFound      ErrorCode = 311003 // 会话不存在
	ChatSessionExpired       ErrorCode = 311004 // 会话过期
	ChatContextLimitExceeded ErrorCode = 311005 // 上下文长度超限

	// 聊天性能错误 (37xxxx)
	ChatResponseTimeout  ErrorCode = 371001 // 响应超时
	ChatQueueFull        ErrorCode = 371002 // 队列已满
	ChatConcurrencyLimit ErrorCode = 371003 // 并发限制

	// ========== 语音服务错误码 (4xxxxx) ==========
	// 语音通用错误 (40xxxx)
	VoiceInternalError      ErrorCode = 402001 // 语音服务内部错误
	VoiceServiceUnavailable ErrorCode = 402002 // 语音服务不可用

	// 语音API错误 (41xxxx)
	VoiceInvalidFormat         ErrorCode = 411001 // 音频格式无效
	VoiceFileTooLarge          ErrorCode = 411002 // 音频文件过大
	VoiceProcessingFailed      ErrorCode = 411003 // 音频处理失败
	VoiceASRFailed             ErrorCode = 411004 // 语音识别失败
	VoiceTTSFailed             ErrorCode = 411005 // 语音合成失败
	VoiceEmotionAnalysisFailed ErrorCode = 411006 // 情感分析失败

	// 语音文件错误 (46xxxx)
	VoiceFileNotFound  ErrorCode = 461001 // 音频文件不存在
	VoiceFileCorrupted ErrorCode = 461002 // 音频文件损坏
	VoiceStorageFailed ErrorCode = 461003 // 音频存储失败

	// ========== RAG服务错误码 (5xxxxx) ==========
	// RAG通用错误 (50xxxx)
	RAGInternalError      ErrorCode = 502001 // RAG服务内部错误
	RAGServiceUnavailable ErrorCode = 502002 // RAG服务不可用

	// RAG API错误 (51xxxx)
	RAGInvalidQuery     ErrorCode = 511001 // 无效查询
	RAGDocumentNotFound ErrorCode = 511002 // 文档不存在
	RAGIndexingFailed   ErrorCode = 511003 // 索引失败
	RAGRetrievalFailed  ErrorCode = 511004 // 检索失败
	RAGEmbeddingFailed  ErrorCode = 511005 // 向量化失败
	RAGRerankingFailed  ErrorCode = 511006 // 重排序失败

	// RAG数据库错误 (53xxxx)
	RAGVectorDBError            ErrorCode = 533001 // 向量数据库错误
	RAGVectorDBConnectionFailed ErrorCode = 533002 // 向量数据库连接失败
	RAGCollectionNotFound       ErrorCode = 533003 // 集合不存在

	// ========== 存储服务错误码 (6xxxxx) ==========
	// 存储通用错误 (60xxxx)
	StorageInternalError      ErrorCode = 602001 // 存储服务内部错误
	StorageServiceUnavailable ErrorCode = 602002 // 存储服务不可用

	// 存储文件错误 (66xxxx)
	StorageFileNotFound      ErrorCode = 661001 // 文件不存在
	StorageFileAccessDenied  ErrorCode = 661002 // 文件访问被拒绝
	StorageFileCorrupted     ErrorCode = 661003 // 文件损坏
	StorageInsufficientSpace ErrorCode = 661004 // 存储空间不足
	StorageUploadFailed      ErrorCode = 661005 // 文件上传失败
	StorageDownloadFailed    ErrorCode = 661006 // 文件下载失败

	// ========== 集成服务错误码 (7xxxxx) ==========
	// 集成通用错误 (70xxxx)
	IntegrationInternalError      ErrorCode = 702001 // 集成服务内部错误
	IntegrationServiceUnavailable ErrorCode = 702002 // 集成服务不可用

	// 集成API错误 (71xxxx)
	IntegrationInvalidConfig    ErrorCode = 711001 // 无效配置
	IntegrationConnectionFailed ErrorCode = 711002 // 连接失败
	IntegrationAuthFailed       ErrorCode = 711003 // 认证失败
	IntegrationAPILimitExceeded ErrorCode = 711004 // API调用限制超出
	IntegrationDataSyncFailed   ErrorCode = 711005 // 数据同步失败

	// ========== 监控服务错误码 (8xxxxx) ==========
	// 监控通用错误 (80xxxx)
	MonitorInternalError      ErrorCode = 802001 // 监控服务内部错误
	MonitorServiceUnavailable ErrorCode = 802002 // 监控服务不可用

	// 监控API错误 (81xxxx)
	MonitorMetricNotFound     ErrorCode = 811001 // 指标不存在
	MonitorInvalidTimeRange   ErrorCode = 811002 // 无效时间范围
	MonitorQueryFailed        ErrorCode = 811003 // 查询失败
	MonitorAlertConfigInvalid ErrorCode = 811004 // 告警配置无效

	// ========== 通用错误码 (9xxxxx) ==========
	// 通用系统错误 (90xxxx)
	SystemInternalError   ErrorCode = 902001 // 系统内部错误
	SystemMaintenanceMode ErrorCode = 902002 // 系统维护模式
	SystemOverloaded      ErrorCode = 902003 // 系统过载

	// 通用配置错误 (96xxxx)
	ConfigNotFound   ErrorCode = 961001 // 配置不存在
	ConfigInvalid    ErrorCode = 961002 // 配置无效
	ConfigLoadFailed ErrorCode = 961003 // 配置加载失败

	// 通用网络错误 (93xxxx)
	NetworkTimeout           ErrorCode = 933001 // 网络超时
	NetworkConnectionRefused ErrorCode = 933002 // 连接被拒绝
	NetworkHostUnreachable   ErrorCode = 933003 // 主机不可达
)

// ErrorInfo 错误信息结构
type ErrorInfo struct {
	Code        ErrorCode `json:"code"`
	Message     string    `json:"message"`
	Description string    `json:"description"`
	HTTPStatus  int       `json:"http_status"`
	Category    string    `json:"category"`
	Service     string    `json:"service"`
}

// errorInfoMap 错误码信息映射
var errorInfoMap = map[ErrorCode]ErrorInfo{
	Success: {Success, "Success", "操作成功", http.StatusOK, "Success", "Common"},

	// Gateway错误
	GatewayInternalError:      {GatewayInternalError, "Gateway Internal Error", "网关内部错误", http.StatusInternalServerError, "Gateway", "Gateway"},
	GatewayServiceUnavailable: {GatewayServiceUnavailable, "Gateway Service Unavailable", "网关服务不可用", http.StatusServiceUnavailable, "Gateway", "Gateway"},
	GatewayTimeout:            {GatewayTimeout, "Gateway Timeout", "网关超时", http.StatusGatewayTimeout, "Gateway", "Gateway"},
	GatewayInvalidRequest:     {GatewayInvalidRequest, "Invalid Request", "无效请求", http.StatusBadRequest, "Gateway", "Gateway"},
	GatewayMissingParameter:   {GatewayMissingParameter, "Missing Parameter", "缺少参数", http.StatusBadRequest, "Gateway", "Gateway"},
	GatewayInvalidParameter:   {GatewayInvalidParameter, "Invalid Parameter", "参数无效", http.StatusBadRequest, "Gateway", "Gateway"},
	GatewayRequestTooLarge:    {GatewayRequestTooLarge, "Request Too Large", "请求体过大", http.StatusRequestEntityTooLarge, "Gateway", "Gateway"},
	GatewayRateLimitExceeded:  {GatewayRateLimitExceeded, "Rate Limit Exceeded", "请求频率超限", http.StatusTooManyRequests, "Gateway", "Gateway"},
	GatewayNetworkError:       {GatewayNetworkError, "Network Error", "网络错误", http.StatusBadGateway, "Gateway", "Gateway"},
	GatewayConnectionFailed:   {GatewayConnectionFailed, "Connection Failed", "连接失败", http.StatusBadGateway, "Gateway", "Gateway"},
	GatewayDNSResolveFailed:   {GatewayDNSResolveFailed, "DNS Resolve Failed", "DNS解析失败", http.StatusBadGateway, "Gateway", "Gateway"},

	// 认证错误
	AuthInternalError:      {AuthInternalError, "Auth Internal Error", "认证服务内部错误", http.StatusInternalServerError, "Auth", "Auth"},
	AuthServiceUnavailable: {AuthServiceUnavailable, "Auth Service Unavailable", "认证服务不可用", http.StatusServiceUnavailable, "Auth", "Auth"},
	AuthInvalidCredentials: {AuthInvalidCredentials, "Invalid Credentials", "无效凭证", http.StatusUnauthorized, "Auth", "Auth"},
	AuthTokenExpired:       {AuthTokenExpired, "Token Expired", "Token过期", http.StatusUnauthorized, "Auth", "Auth"},
	AuthTokenInvalid:       {AuthTokenInvalid, "Token Invalid", "Token无效", http.StatusUnauthorized, "Auth", "Auth"},
	AuthPermissionDenied:   {AuthPermissionDenied, "Permission Denied", "权限不足", http.StatusForbidden, "Auth", "Auth"},
	AuthUserNotFound:       {AuthUserNotFound, "User Not Found", "用户不存在", http.StatusNotFound, "Auth", "Auth"},
	AuthUserDisabled:       {AuthUserDisabled, "User Disabled", "用户已禁用", http.StatusForbidden, "Auth", "Auth"},
	AuthSecurityViolation:  {AuthSecurityViolation, "Security Violation", "安全违规", http.StatusForbidden, "Auth", "Auth"},
	AuthBruteForceDetected: {AuthBruteForceDetected, "Brute Force Detected", "检测到暴力破解", http.StatusTooManyRequests, "Auth", "Auth"},
	AuthSuspiciousActivity: {AuthSuspiciousActivity, "Suspicious Activity", "可疑活动", http.StatusForbidden, "Auth", "Auth"},

	// 聊天错误
	ChatInternalError:        {ChatInternalError, "Chat Internal Error", "聊天服务内部错误", http.StatusInternalServerError, "Chat", "Chat"},
	ChatServiceUnavailable:   {ChatServiceUnavailable, "Chat Service Unavailable", "聊天服务不可用", http.StatusServiceUnavailable, "Chat", "Chat"},
	ChatInvalidMessage:       {ChatInvalidMessage, "Invalid Message", "无效消息", http.StatusBadRequest, "Chat", "Chat"},
	ChatMessageTooLong:       {ChatMessageTooLong, "Message Too Long", "消息过长", http.StatusBadRequest, "Chat", "Chat"},
	ChatSessionNotFound:      {ChatSessionNotFound, "Session Not Found", "会话不存在", http.StatusNotFound, "Chat", "Chat"},
	ChatSessionExpired:       {ChatSessionExpired, "Session Expired", "会话过期", http.StatusGone, "Chat", "Chat"},
	ChatContextLimitExceeded: {ChatContextLimitExceeded, "Context Limit Exceeded", "上下文长度超限", http.StatusBadRequest, "Chat", "Chat"},
	ChatResponseTimeout:      {ChatResponseTimeout, "Response Timeout", "响应超时", http.StatusRequestTimeout, "Chat", "Chat"},
	ChatQueueFull:            {ChatQueueFull, "Queue Full", "队列已满", http.StatusServiceUnavailable, "Chat", "Chat"},
	ChatConcurrencyLimit:     {ChatConcurrencyLimit, "Concurrency Limit", "并发限制", http.StatusTooManyRequests, "Chat", "Chat"},

	// 语音错误
	VoiceInternalError:         {VoiceInternalError, "Voice Internal Error", "语音服务内部错误", http.StatusInternalServerError, "Voice", "Voice"},
	VoiceServiceUnavailable:    {VoiceServiceUnavailable, "Voice Service Unavailable", "语音服务不可用", http.StatusServiceUnavailable, "Voice", "Voice"},
	VoiceInvalidFormat:         {VoiceInvalidFormat, "Invalid Audio Format", "音频格式无效", http.StatusBadRequest, "Voice", "Voice"},
	VoiceFileTooLarge:          {VoiceFileTooLarge, "Audio File Too Large", "音频文件过大", http.StatusRequestEntityTooLarge, "Voice", "Voice"},
	VoiceProcessingFailed:      {VoiceProcessingFailed, "Audio Processing Failed", "音频处理失败", http.StatusInternalServerError, "Voice", "Voice"},
	VoiceASRFailed:             {VoiceASRFailed, "ASR Failed", "语音识别失败", http.StatusInternalServerError, "Voice", "Voice"},
	VoiceTTSFailed:             {VoiceTTSFailed, "TTS Failed", "语音合成失败", http.StatusInternalServerError, "Voice", "Voice"},
	VoiceEmotionAnalysisFailed: {VoiceEmotionAnalysisFailed, "Emotion Analysis Failed", "情感分析失败", http.StatusInternalServerError, "Voice", "Voice"},
	VoiceFileNotFound:          {VoiceFileNotFound, "Audio File Not Found", "音频文件不存在", http.StatusNotFound, "Voice", "Voice"},
	VoiceFileCorrupted:         {VoiceFileCorrupted, "Audio File Corrupted", "音频文件损坏", http.StatusBadRequest, "Voice", "Voice"},
	VoiceStorageFailed:         {VoiceStorageFailed, "Audio Storage Failed", "音频存储失败", http.StatusInternalServerError, "Voice", "Voice"},

	// RAG错误
	RAGInternalError:            {RAGInternalError, "RAG Internal Error", "RAG服务内部错误", http.StatusInternalServerError, "RAG", "RAG"},
	RAGServiceUnavailable:       {RAGServiceUnavailable, "RAG Service Unavailable", "RAG服务不可用", http.StatusServiceUnavailable, "RAG", "RAG"},
	RAGInvalidQuery:             {RAGInvalidQuery, "Invalid Query", "无效查询", http.StatusBadRequest, "RAG", "RAG"},
	RAGDocumentNotFound:         {RAGDocumentNotFound, "Document Not Found", "文档不存在", http.StatusNotFound, "RAG", "RAG"},
	RAGIndexingFailed:           {RAGIndexingFailed, "Indexing Failed", "索引失败", http.StatusInternalServerError, "RAG", "RAG"},
	RAGRetrievalFailed:          {RAGRetrievalFailed, "Retrieval Failed", "检索失败", http.StatusInternalServerError, "RAG", "RAG"},
	RAGEmbeddingFailed:          {RAGEmbeddingFailed, "Embedding Failed", "向量化失败", http.StatusInternalServerError, "RAG", "RAG"},
	RAGRerankingFailed:          {RAGRerankingFailed, "Reranking Failed", "重排序失败", http.StatusInternalServerError, "RAG", "RAG"},
	RAGVectorDBError:            {RAGVectorDBError, "Vector DB Error", "向量数据库错误", http.StatusInternalServerError, "RAG", "RAG"},
	RAGVectorDBConnectionFailed: {RAGVectorDBConnectionFailed, "Vector DB Connection Failed", "向量数据库连接失败", http.StatusServiceUnavailable, "RAG", "RAG"},
	RAGCollectionNotFound:       {RAGCollectionNotFound, "Collection Not Found", "集合不存在", http.StatusNotFound, "RAG", "RAG"},

	// 存储错误
	StorageInternalError:      {StorageInternalError, "Storage Internal Error", "存储服务内部错误", http.StatusInternalServerError, "Storage", "Storage"},
	StorageServiceUnavailable: {StorageServiceUnavailable, "Storage Service Unavailable", "存储服务不可用", http.StatusServiceUnavailable, "Storage", "Storage"},
	StorageFileNotFound:       {StorageFileNotFound, "File Not Found", "文件不存在", http.StatusNotFound, "Storage", "Storage"},
	StorageFileAccessDenied:   {StorageFileAccessDenied, "File Access Denied", "文件访问被拒绝", http.StatusForbidden, "Storage", "Storage"},
	StorageFileCorrupted:      {StorageFileCorrupted, "File Corrupted", "文件损坏", http.StatusBadRequest, "Storage", "Storage"},
	StorageInsufficientSpace:  {StorageInsufficientSpace, "Insufficient Space", "存储空间不足", http.StatusInsufficientStorage, "Storage", "Storage"},
	StorageUploadFailed:       {StorageUploadFailed, "Upload Failed", "文件上传失败", http.StatusInternalServerError, "Storage", "Storage"},
	StorageDownloadFailed:     {StorageDownloadFailed, "Download Failed", "文件下载失败", http.StatusInternalServerError, "Storage", "Storage"},

	// 集成错误
	IntegrationInternalError:      {IntegrationInternalError, "Integration Internal Error", "集成服务内部错误", http.StatusInternalServerError, "Integration", "Integration"},
	IntegrationServiceUnavailable: {IntegrationServiceUnavailable, "Integration Service Unavailable", "集成服务不可用", http.StatusServiceUnavailable, "Integration", "Integration"},
	IntegrationInvalidConfig:      {IntegrationInvalidConfig, "Invalid Config", "无效配置", http.StatusBadRequest, "Integration", "Integration"},
	IntegrationConnectionFailed:   {IntegrationConnectionFailed, "Connection Failed", "连接失败", http.StatusBadGateway, "Integration", "Integration"},
	IntegrationAuthFailed:         {IntegrationAuthFailed, "Auth Failed", "认证失败", http.StatusUnauthorized, "Integration", "Integration"},
	IntegrationAPILimitExceeded:   {IntegrationAPILimitExceeded, "API Limit Exceeded", "API调用限制超出", http.StatusTooManyRequests, "Integration", "Integration"},
	IntegrationDataSyncFailed:     {IntegrationDataSyncFailed, "Data Sync Failed", "数据同步失败", http.StatusInternalServerError, "Integration", "Integration"},

	// 监控错误
	MonitorInternalError:      {MonitorInternalError, "Monitor Internal Error", "监控服务内部错误", http.StatusInternalServerError, "Monitor", "Monitor"},
	MonitorServiceUnavailable: {MonitorServiceUnavailable, "Monitor Service Unavailable", "监控服务不可用", http.StatusServiceUnavailable, "Monitor", "Monitor"},
	MonitorMetricNotFound:     {MonitorMetricNotFound, "Metric Not Found", "指标不存在", http.StatusNotFound, "Monitor", "Monitor"},
	MonitorInvalidTimeRange:   {MonitorInvalidTimeRange, "Invalid Time Range", "无效时间范围", http.StatusBadRequest, "Monitor", "Monitor"},
	MonitorQueryFailed:        {MonitorQueryFailed, "Query Failed", "查询失败", http.StatusInternalServerError, "Monitor", "Monitor"},
	MonitorAlertConfigInvalid: {MonitorAlertConfigInvalid, "Alert Config Invalid", "告警配置无效", http.StatusBadRequest, "Monitor", "Monitor"},

	// 通用错误
	SystemInternalError:      {SystemInternalError, "System Internal Error", "系统内部错误", http.StatusInternalServerError, "System", "Common"},
	SystemMaintenanceMode:    {SystemMaintenanceMode, "System Maintenance Mode", "系统维护模式", http.StatusServiceUnavailable, "System", "Common"},
	SystemOverloaded:         {SystemOverloaded, "System Overloaded", "系统过载", http.StatusServiceUnavailable, "System", "Common"},
	ConfigNotFound:           {ConfigNotFound, "Config Not Found", "配置不存在", http.StatusNotFound, "Config", "Common"},
	ConfigInvalid:            {ConfigInvalid, "Config Invalid", "配置无效", http.StatusBadRequest, "Config", "Common"},
	ConfigLoadFailed:         {ConfigLoadFailed, "Config Load Failed", "配置加载失败", http.StatusInternalServerError, "Config", "Common"},
	NetworkTimeout:           {NetworkTimeout, "Network Timeout", "网络超时", http.StatusRequestTimeout, "Network", "Common"},
	NetworkConnectionRefused: {NetworkConnectionRefused, "Connection Refused", "连接被拒绝", http.StatusBadGateway, "Network", "Common"},
	NetworkHostUnreachable:   {NetworkHostUnreachable, "Host Unreachable", "主机不可达", http.StatusBadGateway, "Network", "Common"},
}

// GetErrorInfo 获取错误信息
func GetErrorInfo(code ErrorCode) ErrorInfo {
	if info, exists := errorInfoMap[code]; exists {
		return info
	}
	return ErrorInfo{
		Code:        code,
		Message:     "Unknown Error",
		Description: fmt.Sprintf("未知错误码: %d", code),
		HTTPStatus:  http.StatusInternalServerError,
		Category:    "Unknown",
		Service:     "Unknown",
	}
}

// String 返回错误码的字符串表示
func (e ErrorCode) String() string {
	info := GetErrorInfo(e)
	return fmt.Sprintf("[%d] %s: %s", e, info.Message, info.Description)
}

// HTTPStatus 返回对应的HTTP状态码
func (e ErrorCode) HTTPStatus() int {
	return GetErrorInfo(e).HTTPStatus
}

// Message 返回错误消息
func (e ErrorCode) Message() string {
	return GetErrorInfo(e).Message
}

// Description 返回错误描述
func (e ErrorCode) Description() string {
	return GetErrorInfo(e).Description
}

// Category 返回错误分类
func (e ErrorCode) Category() string {
	return GetErrorInfo(e).Category
}

// Service 返回所属服务
func (e ErrorCode) Service() string {
	return GetErrorInfo(e).Service
}
