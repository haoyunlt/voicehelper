/**
 * VoiceHelper Android 日志系统
 * 提供结构化日志记录，包含设备信息和性能指标
 */

package com.voicehelper.common

import android.app.ActivityManager
import android.content.Context
import android.content.pm.PackageManager
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.Build
import android.os.Environment
import android.os.StatFs
import android.provider.Settings
import android.telephony.TelephonyManager
import android.util.Log
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.annotations.SerializedName
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.io.File
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ConcurrentHashMap

// MARK: - Log Level
enum class LogLevel(val value: String) {
    DEBUG("debug"),
    INFO("info"),
    WARNING("warning"),
    ERROR("error"),
    CRITICAL("critical");
    
    val androidLogLevel: Int
        get() = when (this) {
            DEBUG -> Log.DEBUG
            INFO -> Log.INFO
            WARNING -> Log.WARN
            ERROR -> Log.ERROR
            CRITICAL -> Log.ERROR
        }
}

// MARK: - Log Type
enum class LogType(val value: String) {
    STARTUP("startup"),
    SHUTDOWN("shutdown"),
    ACTIVITY("activity"),
    FRAGMENT("fragment"),
    SERVICE("service"),
    BROADCAST("broadcast"),
    NETWORK("network"),
    PERMISSION("permission"),
    DATABASE("database"),
    SHARED_PREFERENCES("shared_preferences"),
    FILE_SYSTEM("file_system"),
    ERROR("error"),
    DEBUG("debug"),
    PERFORMANCE("performance"),
    SECURITY("security"),
    BUSINESS("business"),
    SYSTEM("system"),
    BACKGROUND("background")
}

// MARK: - Device Info
data class DeviceInfo(
    @SerializedName("manufacturer")
    val manufacturer: String,
    @SerializedName("model")
    val model: String,
    @SerializedName("device")
    val device: String,
    @SerializedName("product")
    val product: String,
    @SerializedName("board")
    val board: String,
    @SerializedName("hardware")
    val hardware: String,
    @SerializedName("androidVersion")
    val androidVersion: String,
    @SerializedName("apiLevel")
    val apiLevel: Int,
    @SerializedName("buildNumber")
    val buildNumber: String,
    @SerializedName("appVersion")
    val appVersion: String,
    @SerializedName("appVersionCode")
    val appVersionCode: Long,
    @SerializedName("packageName")
    val packageName: String,
    @SerializedName("screenDensity")
    val screenDensity: Float,
    @SerializedName("screenWidth")
    val screenWidth: Int,
    @SerializedName("screenHeight")
    val screenHeight: Int,
    @SerializedName("totalMemory")
    val totalMemory: Long,
    @SerializedName("availableMemory")
    val availableMemory: Long,
    @SerializedName("totalStorage")
    val totalStorage: Long,
    @SerializedName("availableStorage")
    val availableStorage: Long,
    @SerializedName("batteryLevel")
    val batteryLevel: Int?,
    @SerializedName("isCharging")
    val isCharging: Boolean?,
    @SerializedName("networkType")
    val networkType: String,
    @SerializedName("carrierName")
    val carrierName: String?,
    @SerializedName("isRooted")
    val isRooted: Boolean,
    @SerializedName("timeZone")
    val timeZone: String,
    @SerializedName("locale")
    val locale: String,
    @SerializedName("androidId")
    val androidId: String
) {
    companion object {
        fun current(context: Context): DeviceInfo {
            val packageManager = context.packageManager
            val packageInfo = try {
                packageManager.getPackageInfo(context.packageName, 0)
            } catch (e: PackageManager.NameNotFoundException) {
                null
            }
            
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)
            
            val resources = context.resources
            val displayMetrics = resources.displayMetrics
            
            val statFs = StatFs(Environment.getDataDirectory().path)
            val totalStorage = statFs.blockCountLong * statFs.blockSizeLong
            val availableStorage = statFs.availableBlocksLong * statFs.blockSizeLong
            
            return DeviceInfo(
                manufacturer = Build.MANUFACTURER,
                model = Build.MODEL,
                device = Build.DEVICE,
                product = Build.PRODUCT,
                board = Build.BOARD,
                hardware = Build.HARDWARE,
                androidVersion = Build.VERSION.RELEASE,
                apiLevel = Build.VERSION.SDK_INT,
                buildNumber = Build.DISPLAY,
                appVersion = packageInfo?.versionName ?: "Unknown",
                appVersionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    packageInfo?.longVersionCode ?: 0L
                } else {
                    @Suppress("DEPRECATION")
                    packageInfo?.versionCode?.toLong() ?: 0L
                },
                packageName = context.packageName,
                screenDensity = displayMetrics.density,
                screenWidth = displayMetrics.widthPixels,
                screenHeight = displayMetrics.heightPixels,
                totalMemory = memoryInfo.totalMem,
                availableMemory = memoryInfo.availMem,
                totalStorage = totalStorage,
                availableStorage = availableStorage,
                batteryLevel = getBatteryLevel(context),
                isCharging = isCharging(context),
                networkType = getNetworkType(context),
                carrierName = getCarrierName(context),
                isRooted = isRooted(),
                timeZone = TimeZone.getDefault().id,
                locale = Locale.getDefault().toString(),
                androidId = Settings.Secure.getString(context.contentResolver, Settings.Secure.ANDROID_ID)
            )
        }
        
        private fun getBatteryLevel(context: Context): Int? {
            return try {
                val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
                batteryManager.getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
            } catch (e: Exception) {
                null
            }
        }
        
        private fun isCharging(context: Context): Boolean? {
            return try {
                val batteryManager = context.getSystemService(Context.BATTERY_SERVICE) as android.os.BatteryManager
                batteryManager.isCharging
            } catch (e: Exception) {
                null
            }
        }
        
        private fun getNetworkType(context: Context): String {
            val connectivityManager = context.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            val network = connectivityManager.activeNetwork ?: return "none"
            val networkCapabilities = connectivityManager.getNetworkCapabilities(network) ?: return "unknown"
            
            return when {
                networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) -> "wifi"
                networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR) -> "cellular"
                networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET) -> "ethernet"
                networkCapabilities.hasTransport(NetworkCapabilities.TRANSPORT_BLUETOOTH) -> "bluetooth"
                else -> "other"
            }
        }
        
        private fun getCarrierName(context: Context): String? {
            return try {
                val telephonyManager = context.getSystemService(Context.TELEPHONY_SERVICE) as TelephonyManager
                telephonyManager.networkOperatorName?.takeIf { it.isNotEmpty() }
            } catch (e: Exception) {
                null
            }
        }
        
        private fun isRooted(): Boolean {
            val rootPaths = arrayOf(
                "/system/app/Superuser.apk",
                "/sbin/su",
                "/system/bin/su",
                "/system/xbin/su",
                "/data/local/xbin/su",
                "/data/local/bin/su",
                "/system/sd/xbin/su",
                "/system/bin/failsafe/su",
                "/data/local/su",
                "/su/bin/su"
            )
            
            for (path in rootPaths) {
                if (File(path).exists()) {
                    return true
                }
            }
            
            return false
        }
    }
}

// MARK: - Performance Info
data class PerformanceInfo(
    @SerializedName("cpuUsage")
    val cpuUsage: Double?,
    @SerializedName("memoryUsage")
    val memoryUsage: Long?,
    @SerializedName("batteryLevel")
    val batteryLevel: Int?,
    @SerializedName("networkLatency")
    val networkLatency: Double?,
    @SerializedName("heapSize")
    val heapSize: Long?,
    @SerializedName("heapUsed")
    val heapUsed: Long?
)

// MARK: - Log Entry
data class LogEntry(
    @SerializedName("timestamp")
    val timestamp: String,
    @SerializedName("level")
    val level: LogLevel,
    @SerializedName("type")
    val type: LogType,
    @SerializedName("service")
    val service: String,
    @SerializedName("module")
    val module: String,
    @SerializedName("message")
    val message: String,
    @SerializedName("errorCode")
    val errorCode: Int? = null,
    @SerializedName("device")
    val device: DeviceInfo? = null,
    @SerializedName("context")
    val context: Map<String, Any>? = null,
    @SerializedName("stack")
    val stack: String? = null,
    @SerializedName("durationMs")
    val durationMs: Double? = null,
    @SerializedName("activityName")
    val activityName: String? = null,
    @SerializedName("fragmentName")
    val fragmentName: String? = null,
    @SerializedName("networkUrl")
    val networkUrl: String? = null,
    @SerializedName("permissionType")
    val permissionType: String? = null,
    @SerializedName("filePath")
    val filePath: String? = null,
    @SerializedName("performance")
    val performance: PerformanceInfo? = null
)

// MARK: - VoiceHelper Logger
class VoiceHelperLogger private constructor(
    private val context: Context,
    private val service: String,
    private val module: String
) {
    private val baseContext = ConcurrentHashMap<String, Any>()
    private val gson: Gson = GsonBuilder().setPrettyPrinting().create()
    private val logScope = CoroutineScope(Dispatchers.IO)
    private val logFile: File
    
    companion object {
        private const val TAG = "VoiceHelperLogger"
        private var defaultLogger: VoiceHelperLogger? = null
        
        fun initLogger(context: Context, service: String = "voicehelper-android"): VoiceHelperLogger {
            defaultLogger = VoiceHelperLogger(context.applicationContext, service, "")
            return defaultLogger!!
        }
        
        fun getLogger(module: String = ""): VoiceHelperLogger {
            val logger = defaultLogger ?: throw IllegalStateException("Logger not initialized. Call initLogger() first.")
            return if (module.isNotEmpty()) {
                logger.withModule(module)
            } else {
                logger
            }
        }
    }
    
    init {
        // 设置日志文件路径
        val logsDir = File(context.filesDir, "logs")
        if (!logsDir.exists()) {
            logsDir.mkdirs()
        }
        
        val dateFormat = SimpleDateFormat("yyyy-MM-dd", Locale.getDefault())
        val logFileName = "voicehelper-${dateFormat.format(Date())}.log"
        logFile = File(logsDir, logFileName)
        
        // 初始化时记录启动日志
        startup("Android日志系统初始化", mapOf(
            "service" to service,
            "module" to module,
            "logFilePath" to logFile.absolutePath
        ))
    }
    
    private fun buildLogEntry(
        level: LogLevel,
        type: LogType,
        message: String,
        errorCode: ErrorCode? = null,
        context: Map<String, Any>? = null,
        durationMs: Double? = null,
        activityName: String? = null,
        fragmentName: String? = null,
        networkUrl: String? = null,
        permissionType: String? = null,
        filePath: String? = null,
        includeStack: Boolean = false,
        includeDevice: Boolean = false,
        includePerformance: Boolean = false
    ): LogEntry {
        
        val timestamp = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.getDefault()).apply {
            timeZone = TimeZone.getTimeZone("UTC")
        }.format(Date())
        
        val mergedContext = if (context != null) {
            HashMap(baseContext).apply { putAll(context) }
        } else {
            HashMap(baseContext)
        }
        
        val performanceInfo = if (includePerformance || type == LogType.PERFORMANCE) {
            getPerformanceInfo()
        } else null
        
        return LogEntry(
            timestamp = timestamp,
            level = level,
            type = type,
            service = service,
            module = module,
            message = message,
            errorCode = errorCode?.code,
            device = if (includeDevice || type == LogType.STARTUP || type == LogType.SYSTEM) {
                DeviceInfo.current(context)
            } else null,
            context = mergedContext.takeIf { it.isNotEmpty() },
            stack = if (includeStack || level == LogLevel.ERROR || level == LogLevel.CRITICAL) {
                getStackTrace()
            } else null,
            durationMs = durationMs,
            activityName = activityName,
            fragmentName = fragmentName,
            networkUrl = networkUrl,
            permissionType = permissionType,
            filePath = filePath,
            performance = performanceInfo
        )
    }
    
    private fun log(entry: LogEntry) {
        // Android 系统日志
        Log.println(entry.level.androidLogLevel, TAG, entry.message)
        
        // 异步写入文件日志
        logScope.launch {
            writeToFile(entry)
        }
        
        // 发送到远程服务 (可选)
        logScope.launch {
            sendToRemote(entry)
        }
    }
    
    private fun writeToFile(entry: LogEntry) {
        try {
            val jsonString = gson.toJson(entry)
            FileWriter(logFile, true).use { writer ->
                writer.append(jsonString)
                writer.append("\n")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write log to file", e)
        }
    }
    
    private fun sendToRemote(entry: LogEntry) {
        // 实现远程日志发送逻辑
        // 可以发送到服务器或第三方日志服务
    }
    
    private fun getStackTrace(): String {
        return Thread.currentThread().stackTrace.joinToString("\n") { it.toString() }
    }
    
    private fun getPerformanceInfo(): PerformanceInfo {
        val runtime = Runtime.getRuntime()
        val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val memoryInfo = ActivityManager.MemoryInfo()
        activityManager.getMemoryInfo(memoryInfo)
        
        return PerformanceInfo(
            cpuUsage = null, // CPU使用率需要更复杂的计算
            memoryUsage = memoryInfo.totalMem - memoryInfo.availMem,
            batteryLevel = DeviceInfo.getBatteryLevel(context),
            networkLatency = null, // 网络延迟需要实际网络请求测量
            heapSize = runtime.totalMemory(),
            heapUsed = runtime.totalMemory() - runtime.freeMemory()
        )
    }
    
    // MARK: - Public Logging Methods
    
    fun debug(message: String, context: Map<String, Any>? = null) {
        val entry = buildLogEntry(LogLevel.DEBUG, LogType.DEBUG, message, context = context)
        log(entry)
    }
    
    fun info(message: String, context: Map<String, Any>? = null) {
        val entry = buildLogEntry(LogLevel.INFO, LogType.SYSTEM, message, context = context)
        log(entry)
    }
    
    fun warning(message: String, context: Map<String, Any>? = null) {
        val entry = buildLogEntry(LogLevel.WARNING, LogType.SYSTEM, message, context = context)
        log(entry)
    }
    
    fun error(message: String, context: Map<String, Any>? = null) {
        val entry = buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, context = context, includeStack = true, includeDevice = true)
        log(entry)
    }
    
    fun critical(message: String, context: Map<String, Any>? = null) {
        val entry = buildLogEntry(LogLevel.CRITICAL, LogType.ERROR, message, context = context, includeStack = true, includeDevice = true)
        log(entry)
    }
    
    // MARK: - Specialized Logging Methods
    
    fun startup(message: String, context: Map<String, Any>? = null) {
        val entry = buildLogEntry(LogLevel.INFO, LogType.STARTUP, message, context = context, includeDevice = true, includePerformance = true)
        log(entry)
    }
    
    fun shutdown(message: String, context: Map<String, Any>? = null) {
        val entry = buildLogEntry(LogLevel.INFO, LogType.SHUTDOWN, message, context = context, includeDevice = true, includePerformance = true)
        log(entry)
    }
    
    fun activity(action: String, activityName: String, context: Map<String, Any>? = null) {
        val message = "Activity $action: $activityName"
        val entry = buildLogEntry(LogLevel.INFO, LogType.ACTIVITY, message, context = context, activityName = activityName)
        log(entry)
    }
    
    fun fragment(action: String, fragmentName: String, context: Map<String, Any>? = null) {
        val message = "Fragment $action: $fragmentName"
        val entry = buildLogEntry(LogLevel.INFO, LogType.FRAGMENT, message, context = context, fragmentName = fragmentName)
        log(entry)
    }
    
    fun network(method: String, url: String, context: Map<String, Any>? = null) {
        val message = "$method $url"
        val entry = buildLogEntry(LogLevel.DEBUG, LogType.NETWORK, message, context = context, networkUrl = url)
        log(entry)
    }
    
    fun permission(permissionType: String, granted: Boolean, context: Map<String, Any>? = null) {
        val message = "Permission $permissionType: ${if (granted) "granted" else "denied"}"
        val level = if (granted) LogLevel.INFO else LogLevel.WARNING
        val entry = buildLogEntry(level, LogType.PERMISSION, message, context = context, permissionType = permissionType)
        log(entry)
    }
    
    fun errorWithCode(errorCode: ErrorCode, message: String, context: Map<String, Any>? = null) {
        val errorInfo = errorCode.getErrorInfo()
        val mergedContext = HashMap<String, Any>().apply {
            context?.let { putAll(it) }
            put("errorInfo", mapOf(
                "code" to errorInfo.code.code,
                "message" to errorInfo.message,
                "description" to errorInfo.description,
                "category" to errorInfo.category,
                "service" to errorInfo.service
            ))
        }
        val entry = buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, errorCode = errorCode, context = mergedContext, includeStack = true, includeDevice = true)
        log(entry)
    }
    
    fun performance(operation: String, durationMs: Double, context: Map<String, Any>? = null) {
        val message = "Performance: $operation"
        val mergedContext = HashMap<String, Any>().apply {
            context?.let { putAll(it) }
            put("operation", operation)
        }
        val entry = buildLogEntry(LogLevel.INFO, LogType.PERFORMANCE, message, context = mergedContext, durationMs = durationMs, includePerformance = true)
        log(entry)
    }
    
    fun security(event: String, context: Map<String, Any>? = null) {
        val message = "Security Event: $event"
        val mergedContext = HashMap<String, Any>().apply {
            context?.let { putAll(it) }
            put("event", event)
        }
        val entry = buildLogEntry(LogLevel.WARNING, LogType.SECURITY, message, context = mergedContext, includeDevice = true)
        log(entry)
    }
    
    fun business(event: String, context: Map<String, Any>? = null) {
        val message = "Business Event: $event"
        val mergedContext = HashMap<String, Any>().apply {
            context?.let { putAll(it) }
            put("event", event)
        }
        val entry = buildLogEntry(LogLevel.INFO, LogType.BUSINESS, message, context = mergedContext)
        log(entry)
    }
    
    fun exception(message: String, throwable: Throwable, context: Map<String, Any>? = null) {
        var errorCode: ErrorCode? = null
        if (throwable is VoiceHelperException) {
            errorCode = throwable.errorCode
        }
        
        val mergedContext = HashMap<String, Any>().apply {
            context?.let { putAll(it) }
            put("exceptionClass", throwable.javaClass.simpleName)
            put("exceptionMessage", throwable.message ?: "")
            put("exceptionCause", throwable.cause?.message ?: "")
        }
        
        val entry = buildLogEntry(LogLevel.ERROR, LogType.ERROR, message, errorCode = errorCode, context = mergedContext, includeStack = true, includeDevice = true)
        log(entry)
    }
    
    // MARK: - Context Management
    
    fun setContext(context: Map<String, Any>) {
        baseContext.putAll(context)
    }
    
    fun withModule(module: String): VoiceHelperLogger {
        val newLogger = VoiceHelperLogger(context, service, module)
        newLogger.baseContext.putAll(baseContext)
        return newLogger
    }
    
    fun withContext(context: Map<String, Any>): VoiceHelperLogger {
        val newLogger = VoiceHelperLogger(this.context, service, module)
        newLogger.baseContext.putAll(baseContext)
        newLogger.baseContext.putAll(context)
        return newLogger
    }
    
    // MARK: - Log File Management
    
    fun getLogFile(): File = logFile
    
    fun getLogFiles(): List<File> {
        val logsDir = logFile.parentFile ?: return emptyList()
        return logsDir.listFiles { file ->
            file.isFile && file.name.startsWith("voicehelper-") && file.name.endsWith(".log")
        }?.sortedBy { it.name } ?: emptyList()
    }
    
    fun cleanOldLogs(daysToKeep: Int = 7) {
        val cutoffTime = System.currentTimeMillis() - (daysToKeep * 24 * 60 * 60 * 1000L)
        val logFiles = getLogFiles()
        
        for (logFile in logFiles) {
            if (logFile.lastModified() < cutoffTime) {
                if (logFile.delete()) {
                    info("删除过期日志文件", mapOf("logFile" to logFile.path, "age" to daysToKeep))
                } else {
                    warning("删除过期日志文件失败", mapOf("logFile" to logFile.path))
                }
            }
        }
    }
}

// MARK: - Convenience Functions
fun debug(message: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().debug(message, context)
}

fun info(message: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().info(message, context)
}

fun warning(message: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().warning(message, context)
}

fun error(message: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().error(message, context)
}

fun critical(message: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().critical(message, context)
}

fun startup(message: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().startup(message, context)
}

fun shutdown(message: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().shutdown(message, context)
}

fun errorWithCode(errorCode: ErrorCode, message: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().errorWithCode(errorCode, message, context)
}

fun performance(operation: String, durationMs: Double, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().performance(operation, durationMs, context)
}

fun security(event: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().security(event, context)
}

fun business(event: String, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().business(event, context)
}

fun exception(message: String, throwable: Throwable, context: Map<String, Any>? = null) {
    VoiceHelperLogger.getLogger().exception(message, throwable, context)
}
