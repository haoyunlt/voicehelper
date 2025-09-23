/**
 * VoiceHelper Android 错误码系统
 * 与其他平台保持一致的错误码体系
 */

package com.voicehelper.common

import android.content.Context
import com.google.gson.Gson
import com.google.gson.annotations.SerializedName

// MARK: - Error Codes
enum class ErrorCode(val code: Int) {
    // 成功码
    SUCCESS(0),
    
    // ========== Gateway服务错误码 (1xxxxx) ==========
    GATEWAY_INTERNAL_ERROR(102001),
    GATEWAY_SERVICE_UNAVAILABLE(102002),
    GATEWAY_TIMEOUT(102003),
    GATEWAY_INVALID_REQUEST(111001),
    GATEWAY_MISSING_PARAMETER(111002),
    GATEWAY_INVALID_PARAMETER(111003),
    GATEWAY_REQUEST_TOO_LARGE(111004),
    GATEWAY_RATE_LIMIT_EXCEEDED(111005),
    GATEWAY_NETWORK_ERROR(133001),
    GATEWAY_CONNECTION_FAILED(133002),
    GATEWAY_DNS_RESOLVE_FAILED(133003),
    
    // ========== 认证服务错误码 (2xxxxx) ==========
    AUTH_INTERNAL_ERROR(202001),
    AUTH_SERVICE_UNAVAILABLE(202002),
    AUTH_INVALID_CREDENTIALS(211001),
    AUTH_TOKEN_EXPIRED(211002),
    AUTH_TOKEN_INVALID(211003),
    AUTH_PERMISSION_DENIED(211004),
    AUTH_USER_NOT_FOUND(211005),
    AUTH_USER_DISABLED(211006),
    AUTH_SECURITY_VIOLATION(281001),
    AUTH_BRUTE_FORCE_DETECTED(281002),
    AUTH_SUSPICIOUS_ACTIVITY(281003),
    
    // ========== 聊天服务错误码 (3xxxxx) ==========
    CHAT_INTERNAL_ERROR(302001),
    CHAT_SERVICE_UNAVAILABLE(302002),
    CHAT_INVALID_MESSAGE(311001),
    CHAT_MESSAGE_TOO_LONG(311002),
    CHAT_SESSION_NOT_FOUND(311003),
    CHAT_SESSION_EXPIRED(311004),
    CHAT_CONTEXT_LIMIT_EXCEEDED(311005),
    CHAT_RESPONSE_TIMEOUT(371001),
    CHAT_QUEUE_FULL(371002),
    CHAT_CONCURRENCY_LIMIT(371003),
    
    // ========== 语音服务错误码 (4xxxxx) ==========
    VOICE_INTERNAL_ERROR(402001),
    VOICE_SERVICE_UNAVAILABLE(402002),
    VOICE_INVALID_FORMAT(411001),
    VOICE_FILE_TOO_LARGE(411002),
    VOICE_PROCESSING_FAILED(411003),
    VOICE_ASR_FAILED(411004),
    VOICE_TTS_FAILED(411005),
    VOICE_EMOTION_ANALYSIS_FAILED(411006),
    VOICE_FILE_NOT_FOUND(461001),
    VOICE_FILE_CORRUPTED(461002),
    VOICE_STORAGE_FAILED(461003),
    
    // ========== RAG服务错误码 (5xxxxx) ==========
    RAG_INTERNAL_ERROR(502001),
    RAG_SERVICE_UNAVAILABLE(502002),
    RAG_INVALID_QUERY(511001),
    RAG_DOCUMENT_NOT_FOUND(511002),
    RAG_INDEXING_FAILED(511003),
    RAG_RETRIEVAL_FAILED(511004),
    RAG_EMBEDDING_FAILED(511005),
    RAG_RERANKING_FAILED(511006),
    RAG_VECTOR_DB_ERROR(533001),
    RAG_VECTOR_DB_CONNECTION_FAILED(533002),
    RAG_COLLECTION_NOT_FOUND(533003),
    
    // ========== 存储服务错误码 (6xxxxx) ==========
    STORAGE_INTERNAL_ERROR(602001),
    STORAGE_SERVICE_UNAVAILABLE(602002),
    STORAGE_FILE_NOT_FOUND(661001),
    STORAGE_FILE_ACCESS_DENIED(661002),
    STORAGE_FILE_CORRUPTED(661003),
    STORAGE_INSUFFICIENT_SPACE(661004),
    STORAGE_UPLOAD_FAILED(661005),
    STORAGE_DOWNLOAD_FAILED(661006),
    
    // ========== Android应用特有错误码 (8xxxxx) ==========
    // Android应用通用错误 (80xxxx)
    ANDROID_INTERNAL_ERROR(802001),
    ANDROID_INITIALIZATION_FAILED(802002),
    ANDROID_UPDATE_FAILED(802003),
    
    // Android UI错误 (81xxxx)
    ANDROID_UI_ERROR(811001),
    ANDROID_ACTIVITY_ERROR(811002),
    ANDROID_FRAGMENT_ERROR(811003),
    ANDROID_VIEW_ERROR(811004),
    
    // Android权限错误 (82xxxx)
    ANDROID_PERMISSION_DENIED(821001),
    ANDROID_MICROPHONE_PERMISSION_DENIED(821002),
    ANDROID_CAMERA_PERMISSION_DENIED(821003),
    ANDROID_LOCATION_PERMISSION_DENIED(821004),
    ANDROID_STORAGE_PERMISSION_DENIED(821005),
    ANDROID_NOTIFICATION_PERMISSION_DENIED(821006),
    
    // Android存储错误 (86xxxx)
    ANDROID_SHARED_PREFERENCES_ERROR(861001),
    ANDROID_DATABASE_ERROR(861002),
    ANDROID_FILE_SYSTEM_ERROR(861003),
    ANDROID_EXTERNAL_STORAGE_ERROR(861004),
    
    // Android网络错误 (83xxxx)
    ANDROID_NETWORK_ERROR(831001),
    ANDROID_HTTP_CLIENT_ERROR(831002),
    ANDROID_CONNECTIVITY_ERROR(831003),
    
    // Android系统错误 (87xxxx)
    ANDROID_SYSTEM_ERROR(871001),
    ANDROID_SERVICE_ERROR(871002),
    ANDROID_BROADCAST_ERROR(871003),
    ANDROID_INTENT_ERROR(871004),
    
    // ========== 通用错误码 (9xxxxx) ==========
    SYSTEM_INTERNAL_ERROR(902001),
    SYSTEM_MAINTENANCE_MODE(902002),
    SYSTEM_OVERLOADED(902003),
    CONFIG_NOT_FOUND(961001),
    CONFIG_INVALID(961002),
    CONFIG_LOAD_FAILED(961003),
    NETWORK_TIMEOUT(933001),
    NETWORK_CONNECTION_REFUSED(933002),
    NETWORK_HOST_UNREACHABLE(933003);
    
    companion object {
        fun fromCode(code: Int): ErrorCode? {
            return values().find { it.code == code }
        }
    }
}

// MARK: - Error Info
data class ErrorInfo(
    @SerializedName("code")
    val code: ErrorCode,
    @SerializedName("message")
    val message: String,
    @SerializedName("description")
    val description: String,
    @SerializedName("category")
    val category: String,
    @SerializedName("service")
    val service: String
)

// MARK: - Error Info Extension
fun ErrorCode.getErrorInfo(): ErrorInfo {
    return when (this) {
        ErrorCode.SUCCESS -> ErrorInfo(
            code = ErrorCode.SUCCESS,
            message = "Success",
            description = "操作成功",
            category = "Success",
            service = "Common"
        )
        
        // Android错误
        ErrorCode.ANDROID_INTERNAL_ERROR -> ErrorInfo(
            code = ErrorCode.ANDROID_INTERNAL_ERROR,
            message = "Android Internal Error",
            description = "Android应用内部错误",
            category = "Android",
            service = "Android"
        )
        ErrorCode.ANDROID_INITIALIZATION_FAILED -> ErrorInfo(
            code = ErrorCode.ANDROID_INITIALIZATION_FAILED,
            message = "Android Initialization Failed",
            description = "Android应用初始化失败",
            category = "Android",
            service = "Android"
        )
        ErrorCode.ANDROID_UI_ERROR -> ErrorInfo(
            code = ErrorCode.ANDROID_UI_ERROR,
            message = "Android UI Error",
            description = "Android界面错误",
            category = "Android",
            service = "Android"
        )
        ErrorCode.ANDROID_PERMISSION_DENIED -> ErrorInfo(
            code = ErrorCode.ANDROID_PERMISSION_DENIED,
            message = "Permission Denied",
            description = "权限被拒绝",
            category = "Android",
            service = "Android"
        )
        ErrorCode.ANDROID_MICROPHONE_PERMISSION_DENIED -> ErrorInfo(
            code = ErrorCode.ANDROID_MICROPHONE_PERMISSION_DENIED,
            message = "Microphone Permission Denied",
            description = "麦克风权限被拒绝",
            category = "Android",
            service = "Android"
        )
        ErrorCode.ANDROID_SHARED_PREFERENCES_ERROR -> ErrorInfo(
            code = ErrorCode.ANDROID_SHARED_PREFERENCES_ERROR,
            message = "SharedPreferences Error",
            description = "SharedPreferences错误",
            category = "Android",
            service = "Android"
        )
        ErrorCode.ANDROID_NETWORK_ERROR -> ErrorInfo(
            code = ErrorCode.ANDROID_NETWORK_ERROR,
            message = "Android Network Error",
            description = "Android网络错误",
            category = "Android",
            service = "Android"
        )
        ErrorCode.ANDROID_SYSTEM_ERROR -> ErrorInfo(
            code = ErrorCode.ANDROID_SYSTEM_ERROR,
            message = "Android System Error",
            description = "Android系统错误",
            category = "Android",
            service = "Android"
        )
        
        // 通用错误
        ErrorCode.SYSTEM_INTERNAL_ERROR -> ErrorInfo(
            code = ErrorCode.SYSTEM_INTERNAL_ERROR,
            message = "System Internal Error",
            description = "系统内部错误",
            category = "System",
            service = "Common"
        )
        ErrorCode.NETWORK_TIMEOUT -> ErrorInfo(
            code = ErrorCode.NETWORK_TIMEOUT,
            message = "Network Timeout",
            description = "网络超时",
            category = "Network",
            service = "Common"
        )
        
        // 其他错误的默认处理
        else -> ErrorInfo(
            code = this,
            message = "Unknown Error",
            description = "未知错误码: ${this.code}",
            category = "Unknown",
            service = "Unknown"
        )
    }
}

// MARK: - VoiceHelper Exception
class VoiceHelperException(
    val errorCode: ErrorCode,
    message: String? = null,
    val details: Map<String, Any>? = null,
    cause: Throwable? = null
) : Exception(message ?: errorCode.getErrorInfo().message, cause) {
    
    val errorInfo: ErrorInfo = errorCode.getErrorInfo()
    
    fun toMap(): Map<String, Any> {
        return mapOf(
            "code" to errorCode.code,
            "message" to errorInfo.message,
            "description" to errorInfo.description,
            "category" to errorInfo.category,
            "service" to errorInfo.service,
            "customMessage" to (message?.takeIf { it != errorInfo.message }),
            "details" to (details ?: emptyMap<String, Any>())
        ).filterValues { it != null }
    }
    
    fun toJson(): String {
        return Gson().toJson(toMap())
    }
}
