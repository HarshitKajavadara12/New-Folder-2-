"""
GIGA SYSTEM - Logger Configuration
Greek Intelligence for Global Analysis

Structured logging system with performance-aware configuration.
Provides colored console output, file rotation, and structured JSON logs
for production environments.

Features:
- Colored console output with emoji indicators
- File rotation with compression
- JSON structured logging for analysis
- Performance metrics logging
- Context-aware log filtering
- Integration with profiling tools
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import logging.handlers

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging as logger


class GigaFormatter(logging.Formatter):
    """Custom formatter for GIGA System logs."""
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'ENDC': '\033[0m'       # End color
    }
    
    # Emoji indicators for log levels
    EMOJIS = {
        'DEBUG': ' ',
        'INFO': ' ', 
        'WARNING': ' ️',
        'ERROR': ' ',
        'CRITICAL': ' '
    }
    
    def format(self, record):
        """Format log record with colors and emojis."""
        # Add emoji and color
        emoji = self.EMOJIS.get(record.levelname, ' ')
        color = self.COLORS.get(record.levelname, '')
        end_color = self.COLORS['ENDC']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Format message
        if hasattr(record, 'module_name'):
            module = f"[{record.module_name}]"
        else:
            module = f"[{record.name}]"
        
        formatted = f"{color}{emoji} {timestamp} {module} {record.getMessage()}{end_color}"
        
        # Add performance metrics if available
        if hasattr(record, 'execution_time'):
            formatted += f" ({record.execution_time:.3f}ms)"
        
        return formatted


class PerformanceFilter(logging.Filter):
    """Filter to add performance context to log records."""
    
    def filter(self, record):
        """Add performance metrics to log record."""
        # Add module context
        if hasattr(record, 'funcName'):
            record.module_name = f"{record.module}.{record.funcName}"
        else:
            record.module_name = record.module
        
        return True


def setup_logging(
    level: str = "INFO",
    log_to_file: bool = True,
    log_file_path: Optional[Path] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_logging: bool = False
) -> logging.Logger:
    """
    Setup comprehensive logging for GIGA System.
    
    Args:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_to_file: Enable file logging
        log_file_path: Custom log file path
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        json_logging: Enable JSON structured logging
        
    Returns:
        Configured logger instance
    """
    
    if LOGURU_AVAILABLE:
        return _setup_loguru_logging(level, log_to_file, log_file_path)
    else:
        return _setup_standard_logging(level, log_to_file, log_file_path, 
                                     max_file_size, backup_count, json_logging)


def _setup_loguru_logging(level: str, log_to_file: bool, log_file_path: Optional[Path]):
    """Setup logging using loguru (preferred)."""
    
    # Remove default handler
    logger.remove()
    
    # Console handler with colors and formatting
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=console_format,
        level=level,
        colorize=True,
        enqueue=True
    )
    
    # File handler if requested
    if log_to_file:
        if log_file_path is None:
            log_file_path = Path("logs/giga_system_{time}.log")
        
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # File format without colors
        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )
        
        logger.add(
            str(log_file_path),
            format=file_format,
            level=level,
            rotation="10 MB",
            retention="30 days",
            compression="gz",
            enqueue=True,
            backtrace=True,
            diagnose=True
        )
    
    return logger


def _setup_standard_logging(level: str, log_to_file: bool, log_file_path: Optional[Path],
                          max_file_size: int, backup_count: int, json_logging: bool):
    """Setup logging using standard library."""
    
    # Create root logger
    root_logger = logging.getLogger("giga_system")
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(GigaFormatter())
    console_handler.addFilter(PerformanceFilter())
    root_logger.addHandler(console_handler)
    
    # File handler if requested
    if log_to_file:
        if log_file_path is None:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            log_file_path = log_dir / "giga_system.log"
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        
        if json_logging:
            file_formatter = JsonFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        root_logger.addHandler(file_handler)
    
    return root_logger


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Add performance metrics if available
        if hasattr(record, 'execution_time'):
            log_entry['execution_time_ms'] = record.execution_time
        
        if hasattr(record, 'memory_usage'):
            log_entry['memory_usage_mb'] = record.memory_usage
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def get_logger(name: str = "giga_system") -> logging.Logger:
    """
    Get logger instance for a specific module.
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        Logger instance
    """
    if LOGURU_AVAILABLE:
        return logger.bind(name=name)
    else:
        return logging.getLogger(name)


def log_performance(func_name: str, execution_time: float, memory_usage: Optional[float] = None):
    """
    Log performance metrics for a function.
    
    Args:
        func_name: Name of the function
        execution_time: Execution time in milliseconds
        memory_usage: Memory usage in MB (optional)
    """
    perf_logger = get_logger("performance")
    
    if LOGURU_AVAILABLE:
        perf_logger.info(
            f"Function {func_name} executed in {execution_time:.3f}ms"
            + (f" using {memory_usage:.2f}MB" if memory_usage else "")
        )
    else:
        extra = {'execution_time': execution_time}
        if memory_usage:
            extra['memory_usage'] = memory_usage
        
        perf_logger.info(
            f"Function {func_name} executed",
            extra=extra
        )


def log_greek_calculation(symbol: str, greeks: Dict[str, float], calculation_time: float):
    """
    Log Greek calculation results with performance metrics.
    
    Args:
        symbol: Symbol for which Greeks were calculated
        greeks: Dictionary of Greek values
        calculation_time: Time taken for calculation (ms)
    """
    greeks_logger = get_logger("greeks")
    
    message = f"Greeks calculated for {symbol}: "
    message += ", ".join([f"{k}={v:.4f}" for k, v in greeks.items()])
    
    if LOGURU_AVAILABLE:
        greeks_logger.debug(message, execution_time=calculation_time)
    else:
        greeks_logger.debug(message, extra={'execution_time': calculation_time})


def log_trade_execution(trade_details: Dict[str, Any]):
    """
    Log trade execution details.
    
    Args:
        trade_details: Dictionary with trade information
    """
    trade_logger = get_logger("trading")
    
    message = (
        f"Trade executed: {trade_details.get('side')} "
        f"{trade_details.get('quantity')} {trade_details.get('symbol')} "
        f"@ {trade_details.get('price')}"
    )
    
    if 'execution_time_ms' in trade_details:
        message += f" (latency: {trade_details['execution_time_ms']:.1f}ms)"
    
    trade_logger.info(message)


def log_strategy_performance(strategy_name: str, metrics: Dict[str, float]):
    """
    Log strategy performance metrics.
    
    Args:
        strategy_name: Name of the trading strategy
        metrics: Dictionary of performance metrics
    """
    strategy_logger = get_logger("strategy")
    
    message = f"Strategy {strategy_name} performance: "
    message += ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    
    strategy_logger.info(message)


# Initialize default logging lazy wrapper
_default_logger: Optional[logging.Logger] = None

def get_default_logger() -> logging.Logger:
    """Get or create the default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logging()
    return _default_logger


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, **context):
        self.context = context
        self.original_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        """Enter context - add context to all log records."""
        def record_factory(*args, **kwargs):
            record = self.original_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original factory."""
        logging.setLogRecordFactory(self.original_factory)


# Usage examples:
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging(level="DEBUG", log_to_file=True)
    
    # Basic logging
    logger.info("GIGA System starting up")
    logger.debug("Debug message with details")
    logger.warning("Warning message")
    
    # Performance logging
    log_performance("black_scholes_calculation", 0.052, 1.2)
    
    # Greek calculation logging
    greeks = {"delta": 0.6234, "gamma": 0.0156, "theta": -0.0234}
    log_greek_calculation("AAPL", greeks, 0.045)
    
    # Trade execution logging
    trade = {
        "side": "BUY",
        "quantity": 100,
        "symbol": "AAPL",
        "price": 150.25,
        "execution_time_ms": 2.3
    }
    log_trade_execution(trade)
    
    # Context logging
    with LogContext(strategy="delta_neutral", user_id=123):
        logger.info("Processing strategy with context")