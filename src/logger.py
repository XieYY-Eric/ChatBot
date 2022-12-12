import logging

class Logger(logging.Logger):
    
    DEBUG_COLOR = "\x1b[38;20m"
    WARN_COLOR = "\x1b[33;20m"
    ERROR_COLOR = "\x1b[31;20m"
    EXCEPTION_COLOR = "\x1b[31;1m"
    END_COLOR = "\x1b[0m"
    
    def __init__(self, lvl = logging.DEBUG):
        logging.Logger.__init__(self, __name__, lvl)
        
        # TODO(Sean) formatting (time, etc)
        # TODO(Sean) file logging
        console = logging.StreamHandler()
        self.addHandler(console)
        
    def log_debug(self, msg: str, *args) -> None:
        """Print a debug log message."""
        self.debug(Logger.DEBUG_COLOR + f"[DEBUG]: {msg}" + Logger.END_COLOR, *args)
    
    def log_info(self, msg: str, *args) -> None:
        """Print an info log message."""
        self.info(f"[INFO]: {msg}", *args)
        
    def log_warn(self, msg: str, *args) -> None:
        """Print a warning log message."""
        self.warning(Logger.WARN_COLOR + f"[WARN]: {msg}" + Logger.END_COLOR, *args)
        
    def log_error(self, msg: str, *args) -> None:
        """Print an error log message."""
        self.error(Logger.ERROR_COLOR + f"[ERROR]: {msg}" + Logger.END_COLOR, *args)
    
    def log_exception(self, msg: str, *args) -> None:
        """Print an exception log message."""
        self.exception(Logger.EXCEPTION_COLOR + f"[EXCEPTION]: {msg}" + Logger.END_COLOR, *args)
