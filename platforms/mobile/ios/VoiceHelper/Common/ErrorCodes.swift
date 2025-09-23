/**
 * VoiceHelper iOS 错误码系统
 * 与其他平台保持一致的错误码体系
 */

import Foundation

// MARK: - Error Codes
enum ErrorCode: Int, CaseIterable {
    // 成功码
    case success = 0
    
    // ========== Gateway服务错误码 (1xxxxx) ==========
    case gatewayInternalError = 102001
    case gatewayServiceUnavailable = 102002
    case gatewayTimeout = 102003
    case gatewayInvalidRequest = 111001
    case gatewayMissingParameter = 111002
    case gatewayInvalidParameter = 111003
    case gatewayRequestTooLarge = 111004
    case gatewayRateLimitExceeded = 111005
    case gatewayNetworkError = 133001
    case gatewayConnectionFailed = 133002
    case gatewayDNSResolveFailed = 133003
    
    // ========== 认证服务错误码 (2xxxxx) ==========
    case authInternalError = 202001
    case authServiceUnavailable = 202002
    case authInvalidCredentials = 211001
    case authTokenExpired = 211002
    case authTokenInvalid = 211003
    case authPermissionDenied = 211004
    case authUserNotFound = 211005
    case authUserDisabled = 211006
    case authSecurityViolation = 281001
    case authBruteForceDetected = 281002
    case authSuspiciousActivity = 281003
    
    // ========== 聊天服务错误码 (3xxxxx) ==========
    case chatInternalError = 302001
    case chatServiceUnavailable = 302002
    case chatInvalidMessage = 311001
    case chatMessageTooLong = 311002
    case chatSessionNotFound = 311003
    case chatSessionExpired = 311004
    case chatContextLimitExceeded = 311005
    case chatResponseTimeout = 371001
    case chatQueueFull = 371002
    case chatConcurrencyLimit = 371003
    
    // ========== 语音服务错误码 (4xxxxx) ==========
    case voiceInternalError = 402001
    case voiceServiceUnavailable = 402002
    case voiceInvalidFormat = 411001
    case voiceFileTooLarge = 411002
    case voiceProcessingFailed = 411003
    case voiceASRFailed = 411004
    case voiceTTSFailed = 411005
    case voiceEmotionAnalysisFailed = 411006
    case voiceFileNotFound = 461001
    case voiceFileCorrupted = 461002
    case voiceStorageFailed = 461003
    
    // ========== RAG服务错误码 (5xxxxx) ==========
    case ragInternalError = 502001
    case ragServiceUnavailable = 502002
    case ragInvalidQuery = 511001
    case ragDocumentNotFound = 511002
    case ragIndexingFailed = 511003
    case ragRetrievalFailed = 511004
    case ragEmbeddingFailed = 511005
    case ragRerankingFailed = 511006
    case ragVectorDBError = 533001
    case ragVectorDBConnectionFailed = 533002
    case ragCollectionNotFound = 533003
    
    // ========== 存储服务错误码 (6xxxxx) ==========
    case storageInternalError = 602001
    case storageServiceUnavailable = 602002
    case storageFileNotFound = 661001
    case storageFileAccessDenied = 661002
    case storageFileCorrupted = 661003
    case storageInsufficientSpace = 661004
    case storageUploadFailed = 661005
    case storageDownloadFailed = 661006
    
    // ========== iOS应用特有错误码 (8xxxxx) ==========
    // iOS应用通用错误 (80xxxx)
    case iOSInternalError = 802001
    case iOSInitializationFailed = 802002
    case iOSUpdateFailed = 802003
    
    // iOS UI错误 (81xxxx)
    case iOSUIError = 811001
    case iOSViewControllerError = 811002
    case iOSNavigationError = 811003
    case iOSAnimationError = 811004
    
    // iOS权限错误 (82xxxx)
    case iOSPermissionDenied = 821001
    case iOSMicrophonePermissionDenied = 821002
    case iOSCameraPermissionDenied = 821003
    case iOSLocationPermissionDenied = 821004
    case iOSNotificationPermissionDenied = 821005
    
    // iOS存储错误 (86xxxx)
    case iOSKeychainError = 861001
    case iOSUserDefaultsError = 861002
    case iOSCoreDataError = 861003
    case iOSFileManagerError = 861004
    
    // iOS网络错误 (83xxxx)
    case iOSNetworkError = 831001
    case iOSURLSessionError = 831002
    case iOSReachabilityError = 831003
    
    // iOS系统错误 (87xxxx)
    case iOSSystemError = 871001
    case iOSBackgroundTaskError = 871002
    case iOSMemoryWarning = 871003
    
    // ========== 通用错误码 (9xxxxx) ==========
    case systemInternalError = 902001
    case systemMaintenanceMode = 902002
    case systemOverloaded = 902003
    case configNotFound = 961001
    case configInvalid = 961002
    case configLoadFailed = 961003
    case networkTimeout = 933001
    case networkConnectionRefused = 933002
    case networkHostUnreachable = 933003
}

// MARK: - Error Info
struct ErrorInfo {
    let code: ErrorCode
    let message: String
    let description: String
    let category: String
    let service: String
    
    init(code: ErrorCode, message: String, description: String, category: String, service: String) {
        self.code = code
        self.message = message
        self.description = description
        self.category = category
        self.service = service
    }
}

// MARK: - Error Info Mapping
extension ErrorCode {
    var errorInfo: ErrorInfo {
        switch self {
        case .success:
            return ErrorInfo(code: .success, message: "Success", description: "操作成功", category: "Success", service: "Common")
            
        // iOS错误
        case .iOSInternalError:
            return ErrorInfo(code: .iOSInternalError, message: "iOS Internal Error", description: "iOS应用内部错误", category: "iOS", service: "iOS")
        case .iOSInitializationFailed:
            return ErrorInfo(code: .iOSInitializationFailed, message: "iOS Initialization Failed", description: "iOS应用初始化失败", category: "iOS", service: "iOS")
        case .iOSUIError:
            return ErrorInfo(code: .iOSUIError, message: "iOS UI Error", description: "iOS界面错误", category: "iOS", service: "iOS")
        case .iOSPermissionDenied:
            return ErrorInfo(code: .iOSPermissionDenied, message: "Permission Denied", description: "权限被拒绝", category: "iOS", service: "iOS")
        case .iOSMicrophonePermissionDenied:
            return ErrorInfo(code: .iOSMicrophonePermissionDenied, message: "Microphone Permission Denied", description: "麦克风权限被拒绝", category: "iOS", service: "iOS")
        case .iOSKeychainError:
            return ErrorInfo(code: .iOSKeychainError, message: "Keychain Error", description: "钥匙串错误", category: "iOS", service: "iOS")
        case .iOSNetworkError:
            return ErrorInfo(code: .iOSNetworkError, message: "iOS Network Error", description: "iOS网络错误", category: "iOS", service: "iOS")
        case .iOSSystemError:
            return ErrorInfo(code: .iOSSystemError, message: "iOS System Error", description: "iOS系统错误", category: "iOS", service: "iOS")
            
        // 通用错误
        case .systemInternalError:
            return ErrorInfo(code: .systemInternalError, message: "System Internal Error", description: "系统内部错误", category: "System", service: "Common")
        case .networkTimeout:
            return ErrorInfo(code: .networkTimeout, message: "Network Timeout", description: "网络超时", category: "Network", service: "Common")
            
        // 其他错误的默认处理
        default:
            return ErrorInfo(code: self, message: "Unknown Error", description: "未知错误码: \(self.rawValue)", category: "Unknown", service: "Unknown")
        }
    }
}

// MARK: - VoiceHelper Error
class VoiceHelperError: NSError {
    let errorCode: ErrorCode
    let errorInfo: ErrorInfo
    let details: [String: Any]?
    
    init(code: ErrorCode, message: String? = nil, details: [String: Any]? = nil) {
        self.errorCode = code
        self.errorInfo = code.errorInfo
        self.details = details
        
        let userInfo: [String: Any] = [
            NSLocalizedDescriptionKey: message ?? errorInfo.message,
            NSLocalizedFailureReasonErrorKey: errorInfo.description,
            "errorCode": code.rawValue,
            "category": errorInfo.category,
            "service": errorInfo.service,
            "details": details ?? [:]
        ]
        
        super.init(domain: "com.voicehelper.error", code: code.rawValue, userInfo: userInfo)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    var toDictionary: [String: Any] {
        return [
            "code": errorCode.rawValue,
            "message": errorInfo.message,
            "description": errorInfo.description,
            "category": errorInfo.category,
            "service": errorInfo.service,
            "details": details ?? [:]
        ]
    }
}
