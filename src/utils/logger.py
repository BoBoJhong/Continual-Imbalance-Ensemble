"""Logging utility for experiments."""
import logging
import sys
from pathlib import Path
from datetime import datetime


def _flush_after_emit(handler_cls):
    """讓 FileHandler / StreamHandler 每次寫入後 flush，重導向 stdout 到檔案時可即時 tail log。"""

    class _Flushing(handler_cls):
        def emit(self, record):
            super().emit(record)
            try:
                self.flush()
            except Exception:
                pass

    _Flushing.__name__ = handler_cls.__name__ + "Flushing"
    return _Flushing


_FlushStreamHandler = _flush_after_emit(logging.StreamHandler)
_FlushFileHandler = _flush_after_emit(logging.FileHandler)


class ExperimentLogger:
    """Logger for experiment tracking."""
    
    def __init__(
        self,
        name: str = "experiment",
        log_dir: str = "logs",
        console: bool = True,
        file: bool = True,
        level: str = "INFO",
        log_filename: str | None = None,
        log_file_mode: str = "w",
    ):
        """
        Initialize experiment logger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            console: Whether to log to console
            file: Whether to log to file
            level: Logging level
            log_filename: 若指定則寫入此檔名（相對於 log_dir），便於長實驗固定路徑 tail；
                          未指定則維持 {name}_{timestamp}.log
            log_file_mode: 與 log_filename 搭配之開檔模式，預設 "w" 每次覆寫
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if console:
            console_handler = _FlushStreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if file:
            if log_filename:
                log_file = self.log_dir / log_filename
                file_handler = _FlushFileHandler(
                    log_file, mode=log_file_mode, encoding="utf-8"
                )
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = self.log_dir / f"{name}_{timestamp}.log"
                file_handler = _FlushFileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
        
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
        
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
        
    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)
        
    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)


def get_logger(name: str = "experiment", **kwargs) -> ExperimentLogger:
    """
    Get an experiment logger instance.
    
    Args:
        name: Logger name
        **kwargs: Additional arguments for ExperimentLogger
        
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(name=name, **kwargs)
