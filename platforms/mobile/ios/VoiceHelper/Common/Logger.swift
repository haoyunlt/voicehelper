/**
 * VoiceHelper iOS 日志系统
 * 提供结构化日志记录，包含设备信息和性能指标
 */

import Foundation
import UIKit
import os.log
import Network

// MARK: - Log Level
enum LogLevel: String, CaseIterable {
    case debug = "debug"
    case info = "info"
    case warning = "warning"
    case error = "error"
    case critical = "critical"
    
    var osLogType: OSLogType {
        switch self {
        case .debug:
            return .debug
        case .info:
            return .info
        case .warning:
            return .default
        case .error:
            return .error
        case .critical:
            return .fault
        }
    }
}

// MARK: - Log Type
enum LogType: String, CaseIterable {
    case startup = "startup"
    case shutdown = "shutdown"
    case viewController = "view_controller"
    case network = "network"
    case permission = "permission"
    case keychain = "keychain"
    case userDefaults = "user_defaults"
    case coreData = "core_data"
    case error = "error"
    case debug = "debug"
    case performance = "performance"
    case security = "security"
    case business = "business"
    case system = "system"
    case background = "background"
}

// MARK: - Device Info
struct DeviceInfo {
    let deviceModel: String
    let systemName: String
    let systemVersion: String
    let appVersion: String
    let buildNumber: String
    let bundleIdentifier: String
    let deviceName: String
    let screenSize: CGSize
    let screenScale: CGFloat
    let batteryLevel: Float
    let batteryState: UIDevice.BatteryState
    let isJailbroken: Bool
    let memoryUsage: UInt64
    let diskSpace: UInt64
    let freeDiskSpace: UInt64
    let networkType: String
    let carrierName: String?
    let timeZone: String
    let locale: String
    
    static func current() -> DeviceInfo {
        let device = UIDevice.current
        let screen = UIScreen.main
        let bundle = Bundle.main
        
        return DeviceInfo(
            deviceModel: device.model,
            systemName: device.systemName,
            systemVersion: device.systemVersion,
            appVersion: bundle.infoDictionary?["CFBundleShortVersionString"] as? String ?? "Unknown",
            buildNumber: bundle.infoDictionary?["CFBundleVersion"] as? String ?? "Unknown",
            bundleIdentifier: bundle.bundleIdentifier ?? "Unknown",
            deviceName: device.name,
            screenSize: screen.bounds.size,
            screenScale: screen.scale,
            batteryLevel: device.batteryLevel,
            batteryState: device.batteryState,
            isJailbroken: DeviceInfo.checkJailbreak(),
            memoryUsage: DeviceInfo.getMemoryUsage(),
            diskSpace: DeviceInfo.getDiskSpace(),
            freeDiskSpace: DeviceInfo.getFreeDiskSpace(),
            networkType: DeviceInfo.getNetworkType(),
            carrierName: DeviceInfo.getCarrierName(),
            timeZone: TimeZone.current.identifier,
            locale: Locale.current.identifier
        )
    }
    
    private static func checkJailbreak() -> Bool {
        let jailbreakPaths = [
            "/Applications/Cydia.app",
            "/Library/MobileSubstrate/MobileSubstrate.dylib",
            "/bin/bash",
            "/usr/sbin/sshd",
            "/etc/apt",
            "/private/var/lib/apt/"
        ]
        
        for path in jailbreakPaths {
            if FileManager.default.fileExists(atPath: path) {
                return true
            }
        }
        
        return false
    }
    
    private static func getMemoryUsage() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return info.resident_size
        } else {
            return 0
        }
    }
    
    private static func getDiskSpace() -> UInt64 {
        do {
            let systemAttributes = try FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory())
            return systemAttributes[.systemSize] as? UInt64 ?? 0
        } catch {
            return 0
        }
    }
    
    private static func getFreeDiskSpace() -> UInt64 {
        do {
            let systemAttributes = try FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory())
            return systemAttributes[.systemFreeSize] as? UInt64 ?? 0
        } catch {
            return 0
        }
    }
    
    private static func getNetworkType() -> String {
        let monitor = NWPathMonitor()
        let queue = DispatchQueue(label: "NetworkMonitor")
        var networkType = "unknown"
        
        monitor.pathUpdateHandler = { path in
            if path.usesInterfaceType(.wifi) {
                networkType = "wifi"
            } else if path.usesInterfaceType(.cellular) {
                networkType = "cellular"
            } else if path.usesInterfaceType(.wiredEthernet) {
                networkType = "ethernet"
            } else {
                networkType = "other"
            }
        }
        
        monitor.start(queue: queue)
        return networkType
    }
    
    private static func getCarrierName() -> String? {
        // Note: CTTelephonyNetworkInfo is deprecated in iOS 16+
        // This is a simplified implementation
        return nil
    }
}

// MARK: - Log Entry
struct LogEntry: Codable {
    let timestamp: String
    let level: LogLevel
    let type: LogType
    let service: String
    let module: String
    let message: String
    let errorCode: Int?
    let device: DeviceInfo?
    let context: [String: AnyCodable]?
    let stack: String?
    let durationMs: Double?
    let viewController: String?
    let networkUrl: String?
    let permissionType: String?
    let filePath: String?
    let performance: PerformanceInfo?
    
    struct PerformanceInfo: Codable {
        let cpuUsage: Double?
        let memoryUsage: UInt64?
        let batteryLevel: Float?
        let networkLatency: Double?
    }
}

// MARK: - AnyCodable for flexible JSON encoding
struct AnyCodable: Codable {
    let value: Any
    
    init(_ value: Any) {
        self.value = value
    }
    
    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        
        if let intValue = try? container.decode(Int.self) {
            value = intValue
        } else if let doubleValue = try? container.decode(Double.self) {
            value = doubleValue
        } else if let stringValue = try? container.decode(String.self) {
            value = stringValue
        } else if let boolValue = try? container.decode(Bool.self) {
            value = boolValue
        } else if let arrayValue = try? container.decode([AnyCodable].self) {
            value = arrayValue.map { $0.value }
        } else if let dictValue = try? container.decode([String: AnyCodable].self) {
            value = dictValue.mapValues { $0.value }
        } else {
            value = NSNull()
        }
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        
        switch value {
        case let intValue as Int:
            try container.encode(intValue)
        case let doubleValue as Double:
            try container.encode(doubleValue)
        case let stringValue as String:
            try container.encode(stringValue)
        case let boolValue as Bool:
            try container.encode(boolValue)
        case let arrayValue as [Any]:
            try container.encode(arrayValue.map { AnyCodable($0) })
        case let dictValue as [String: Any]:
            try container.encode(dictValue.mapValues { AnyCodable($0) })
        default:
            try container.encodeNil()
        }
    }
}

// MARK: - VoiceHelper Logger
class VoiceHelperLogger {
    private let service: String
    private let module: String
    private let osLog: OSLog
    private var baseContext: [String: Any] = [:]
    private let logQueue = DispatchQueue(label: "com.voicehelper.logger", qos: .utility)
    private let fileManager = FileManager.default
    private let logFileURL: URL
    
    init(service: String = "voicehelper-ios", module: String = "") {
        self.service = service
        self.module = module
        self.osLog = OSLog(subsystem: service, category: module.isEmpty ? "default" : module)
        
        // 设置日志文件路径
        let documentsPath = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        let logsDirectory = documentsPath.appendingPathComponent("logs")
        
        // 创建日志目录
        try? fileManager.createDirectory(at: logsDirectory, withIntermediateDirectories: true, attributes: nil)
        
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd"
        let logFileName = "voicehelper-\(dateFormatter.string(from: Date())).log"
        self.logFileURL = logsDirectory.appendingPathComponent(logFileName)
        
        // 初始化时记录启动日志
        startup("iOS日志系统初始化", context: [
            "service": service,
            "module": module,
            "logFileURL": logFileURL.path
        ])
    }
    
    private func buildLogEntry(
        level: LogLevel,
        type: LogType,
        message: String,
        errorCode: ErrorCode? = nil,
        context: [String: Any]? = nil,
        durationMs: Double? = nil,
        viewController: String? = nil,
        networkUrl: String? = nil,
        permissionType: String? = nil,
        filePath: String? = nil,
        includeStack: Bool = false,
        includeDevice: Bool = false,
        includePerformance: Bool = false
    ) -> LogEntry {
        
        let timestamp = ISO8601DateFormatter().string(from: Date())
        
        var contextDict: [String: AnyCodable]?
        if let mergedContext = mergeDictionaries(baseContext, context) {
            contextDict = mergedContext.mapValues { AnyCodable($0) }
        }
        
        var performanceInfo: LogEntry.PerformanceInfo?
        if includePerformance || type == .performance {
            let device = UIDevice.current
            performanceInfo = LogEntry.PerformanceInfo(
                cpuUsage: getCPUUsage(),
                memoryUsage: DeviceInfo.getMemoryUsage(),
                batteryLevel: device.batteryLevel,
                networkLatency: nil // 可以根据需要实现网络延迟测量
            )
        }
        
        return LogEntry(
            timestamp: timestamp,
            level: level,
            type: type,
            service: service,
            module: module,
            message: message,
            errorCode: errorCode?.rawValue,
            device: includeDevice || type == .startup || type == .system ? DeviceInfo.current() : nil,
            context: contextDict,
            stack: includeStack || level == .error || level == .critical ? Thread.callStackSymbols.joined(separator: "\n") : nil,
            durationMs: durationMs,
            viewController: viewController,
            networkUrl: networkUrl,
            permissionType: permissionType,
            filePath: filePath,
            performance: performanceInfo
        )
    }
    
    private func log(_ entry: LogEntry) {
        logQueue.async { [weak self] in
            // 系统日志
            os_log("%{public}@", log: self?.osLog ?? OSLog.default, type: entry.level.osLogType, entry.message)
            
            // 文件日志
            self?.writeToFile(entry)
            
            // 远程日志 (可选)
            self?.sendToRemote(entry)
        }
    }
    
    private func writeToFile(_ entry: LogEntry) {
        do {
            let jsonData = try JSONEncoder().encode(entry)
            let jsonString = String(data: jsonData, encoding: .utf8) ?? ""
            let logLine = jsonString + "\n"
            
            if fileManager.fileExists(atPath: logFileURL.path) {
                let fileHandle = try FileHandle(forWritingTo: logFileURL)
                fileHandle.seekToEndOfFile()
                fileHandle.write(logLine.data(using: .utf8) ?? Data())
                fileHandle.closeFile()
            } else {
                try logLine.write(to: logFileURL, atomically: true, encoding: .utf8)
            }
        } catch {
            os_log("Failed to write log to file: %{public}@", log: osLog, type: .error, error.localizedDescription)
        }
    }
    
    private func sendToRemote(_ entry: LogEntry) {
        // 实现远程日志发送逻辑
        // 可以发送到服务器或第三方日志服务
    }
    
    private func mergeDictionaries(_ dict1: [String: Any]?, _ dict2: [String: Any]?) -> [String: Any]? {
        guard let d1 = dict1 else { return dict2 }
        guard let d2 = dict2 else { return d1 }
        return d1.merging(d2) { _, new in new }
    }
    
    private func getCPUUsage() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size)/4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        
        if kerr == KERN_SUCCESS {
            return Double(info.resident_size) / 1024.0 / 1024.0 // MB
        } else {
            return 0.0
        }
    }
    
    // MARK: - Public Logging Methods
    
    func debug(_ message: String, context: [String: Any]? = nil) {
        let entry = buildLogEntry(level: .debug, type: .debug, message: message, context: context)
        log(entry)
    }
    
    func info(_ message: String, context: [String: Any]? = nil) {
        let entry = buildLogEntry(level: .info, type: .system, message: message, context: context)
        log(entry)
    }
    
    func warning(_ message: String, context: [String: Any]? = nil) {
        let entry = buildLogEntry(level: .warning, type: .system, message: message, context: context)
        log(entry)
    }
    
    func error(_ message: String, context: [String: Any]? = nil) {
        let entry = buildLogEntry(level: .error, type: .error, message: message, context: context, includeStack: true, includeDevice: true)
        log(entry)
    }
    
    func critical(_ message: String, context: [String: Any]? = nil) {
        let entry = buildLogEntry(level: .critical, type: .error, message: message, context: context, includeStack: true, includeDevice: true)
        log(entry)
    }
    
    // MARK: - Specialized Logging Methods
    
    func startup(_ message: String, context: [String: Any]? = nil) {
        let entry = buildLogEntry(level: .info, type: .startup, message: message, context: context, includeDevice: true, includePerformance: true)
        log(entry)
    }
    
    func shutdown(_ message: String, context: [String: Any]? = nil) {
        let entry = buildLogEntry(level: .info, type: .shutdown, message: message, context: context, includeDevice: true, includePerformance: true)
        log(entry)
    }
    
    func viewController(_ action: String, viewController: String, context: [String: Any]? = nil) {
        let message = "ViewController \(action): \(viewController)"
        let entry = buildLogEntry(level: .info, type: .viewController, message: message, context: context, viewController: viewController)
        log(entry)
    }
    
    func network(_ method: String, url: String, context: [String: Any]? = nil) {
        let message = "\(method) \(url)"
        let entry = buildLogEntry(level: .debug, type: .network, message: message, context: context, networkUrl: url)
        log(entry)
    }
    
    func permission(_ permissionType: String, granted: Bool, context: [String: Any]? = nil) {
        let message = "Permission \(permissionType): \(granted ? "granted" : "denied")"
        let level: LogLevel = granted ? .info : .warning
        let entry = buildLogEntry(level: level, type: .permission, message: message, context: context, permissionType: permissionType)
        log(entry)
    }
    
    func errorWithCode(_ errorCode: ErrorCode, message: String, context: [String: Any]? = nil) {
        let errorInfo = errorCode.errorInfo
        let mergedContext = mergeDictionaries(context, [
            "errorInfo": [
                "code": errorInfo.code.rawValue,
                "message": errorInfo.message,
                "description": errorInfo.description,
                "category": errorInfo.category,
                "service": errorInfo.service
            ]
        ])
        let entry = buildLogEntry(level: .error, type: .error, message: message, errorCode: errorCode, context: mergedContext, includeStack: true, includeDevice: true)
        log(entry)
    }
    
    func performance(_ operation: String, durationMs: Double, context: [String: Any]? = nil) {
        let message = "Performance: \(operation)"
        let mergedContext = mergeDictionaries(context, ["operation": operation])
        let entry = buildLogEntry(level: .info, type: .performance, message: message, context: mergedContext, durationMs: durationMs, includePerformance: true)
        log(entry)
    }
    
    func security(_ event: String, context: [String: Any]? = nil) {
        let message = "Security Event: \(event)"
        let mergedContext = mergeDictionaries(context, ["event": event])
        let entry = buildLogEntry(level: .warning, type: .security, message: message, context: mergedContext, includeDevice: true)
        log(entry)
    }
    
    func business(_ event: String, context: [String: Any]? = nil) {
        let message = "Business Event: \(event)"
        let mergedContext = mergeDictionaries(context, ["event": event])
        let entry = buildLogEntry(level: .info, type: .business, message: message, context: mergedContext)
        log(entry)
    }
    
    func exception(_ message: String, error: Error, context: [String: Any]? = nil) {
        var errorCode: ErrorCode?
        if let voiceHelperError = error as? VoiceHelperError {
            errorCode = voiceHelperError.errorCode
        }
        
        let mergedContext = mergeDictionaries(context, [
            "errorDomain": (error as NSError).domain,
            "errorCode": (error as NSError).code,
            "errorDescription": error.localizedDescription
        ])
        
        let entry = buildLogEntry(level: .error, type: .error, message: message, errorCode: errorCode, context: mergedContext, includeStack: true, includeDevice: true)
        log(entry)
    }
    
    // MARK: - Context Management
    
    func setContext(_ context: [String: Any]) {
        baseContext = baseContext.merging(context) { _, new in new }
    }
    
    func withModule(_ module: String) -> VoiceHelperLogger {
        let newLogger = VoiceHelperLogger(service: service, module: module)
        newLogger.baseContext = baseContext
        return newLogger
    }
    
    func withContext(_ context: [String: Any]) -> VoiceHelperLogger {
        let newLogger = VoiceHelperLogger(service: service, module: module)
        newLogger.baseContext = baseContext.merging(context) { _, new in new }
        return newLogger
    }
    
    // MARK: - Log File Management
    
    func getLogFileURL() -> URL {
        return logFileURL
    }
    
    func getLogFiles() -> [URL] {
        let logsDirectory = logFileURL.deletingLastPathComponent()
        do {
            let files = try fileManager.contentsOfDirectory(at: logsDirectory, includingPropertiesForKeys: nil)
            return files.filter { $0.pathExtension == "log" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        } catch {
            exception("获取日志文件列表失败", error: error)
            return []
        }
    }
    
    func cleanOldLogs(daysToKeep: Int = 7) {
        let cutoffDate = Calendar.current.date(byAdding: .day, value: -daysToKeep, to: Date()) ?? Date()
        let logFiles = getLogFiles()
        
        for logFile in logFiles {
            do {
                let attributes = try fileManager.attributesOfItem(atPath: logFile.path)
                if let modificationDate = attributes[.modificationDate] as? Date,
                   modificationDate < cutoffDate {
                    try fileManager.removeItem(at: logFile)
                    info("删除过期日志文件", context: ["logFile": logFile.path, "age": daysToKeep])
                }
            } catch {
                exception("清理过期日志失败", error: error, context: ["logFile": logFile.path])
            }
        }
    }
}

// MARK: - Global Logger
private var defaultLogger: VoiceHelperLogger?

public func initLogger(service: String = "voicehelper-ios") -> VoiceHelperLogger {
    defaultLogger = VoiceHelperLogger(service: service)
    return defaultLogger!
}

public func getLogger(module: String = "") -> VoiceHelperLogger {
    guard let logger = defaultLogger else {
        return initLogger()
    }
    
    if !module.isEmpty {
        return logger.withModule(module)
    }
    
    return logger
}

// MARK: - Convenience Functions
public func debug(_ message: String, context: [String: Any]? = nil) {
    getLogger().debug(message, context: context)
}

public func info(_ message: String, context: [String: Any]? = nil) {
    getLogger().info(message, context: context)
}

public func warning(_ message: String, context: [String: Any]? = nil) {
    getLogger().warning(message, context: context)
}

public func error(_ message: String, context: [String: Any]? = nil) {
    getLogger().error(message, context: context)
}

public func critical(_ message: String, context: [String: Any]? = nil) {
    getLogger().critical(message, context: context)
}

public func startup(_ message: String, context: [String: Any]? = nil) {
    getLogger().startup(message, context: context)
}

public func shutdown(_ message: String, context: [String: Any]? = nil) {
    getLogger().shutdown(message, context: context)
}

public func errorWithCode(_ errorCode: ErrorCode, message: String, context: [String: Any]? = nil) {
    getLogger().errorWithCode(errorCode, message: message, context: context)
}

public func performance(_ operation: String, durationMs: Double, context: [String: Any]? = nil) {
    getLogger().performance(operation, durationMs: durationMs, context: context)
}

public func security(_ event: String, context: [String: Any]? = nil) {
    getLogger().security(event, context: context)
}

public func business(_ event: String, context: [String: Any]? = nil) {
    getLogger().business(event, context: context)
}

public func exception(_ message: String, error: Error, context: [String: Any]? = nil) {
    getLogger().exception(message, error: error, context: context)
}
