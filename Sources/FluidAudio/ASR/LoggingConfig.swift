import Foundation
import OSLog

/// Configuration for controlling FluidAudio logging behavior
public struct FluidAudioLogging {
    /// Global flag to enable/disable all FluidAudio logging
    public static var isEnabled: Bool = true
    
    /// Logging level control
    public enum Level: Int {
        case none = 0
        case error = 1
        case warning = 2
        case info = 3
        case debug = 4
    }
    
    /// Current logging level (only messages at or below this level will be logged)
    public static var level: Level = .info
    
    /// Convenience method to disable all logging
    public static func disable() {
        isEnabled = false
    }
    
    /// Convenience method to enable logging with a specific level
    public static func enable(level: Level = .info) {
        isEnabled = true
        self.level = level
    }
}

/// Wrapper for Logger that respects FluidAudioLogging configuration
internal struct FluidLogger {
    private let logger: Logger
    
    init(subsystem: String, category: String) {
        self.logger = Logger(subsystem: subsystem, category: category)
    }
    
    func debug(_ message: String) {
        guard FluidAudioLogging.isEnabled,
              FluidAudioLogging.level.rawValue >= FluidAudioLogging.Level.debug.rawValue else { return }
        logger.debug("\(message)")
    }
    
    func info(_ message: String) {
        guard FluidAudioLogging.isEnabled,
              FluidAudioLogging.level.rawValue >= FluidAudioLogging.Level.info.rawValue else { return }
        logger.info("\(message)")
    }
    
    func warning(_ message: String) {
        guard FluidAudioLogging.isEnabled,
              FluidAudioLogging.level.rawValue >= FluidAudioLogging.Level.warning.rawValue else { return }
        logger.warning("\(message)")
    }
    
    func error(_ message: String) {
        guard FluidAudioLogging.isEnabled,
              FluidAudioLogging.level.rawValue >= FluidAudioLogging.Level.error.rawValue else { return }
        logger.error("\(message)")
    }
}