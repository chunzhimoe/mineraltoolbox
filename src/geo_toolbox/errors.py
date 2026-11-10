class ToolboxError(Exception):
    pass


class MissingBackendError(ToolboxError, ImportError):
    pass


class DataValidationError(ToolboxError, ValueError):
    pass


class BackendExecutionError(ToolboxError, RuntimeError):
    pass