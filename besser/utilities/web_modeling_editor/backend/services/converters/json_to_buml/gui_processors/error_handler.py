"""
Centralized error handling for GUI generation pipeline.

Provides:
- Custom exception hierarchy
- Contextual error messages
- Error recovery strategies
- User-friendly error formatting
"""

from typing import Any, Dict, List, Optional
import traceback
import logging

logger = logging.getLogger(__name__)


class GUIGenerationError(Exception):
    """Base exception for GUI generation errors."""
    
    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recovery_suggestion: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.context = context or {}
        self.recovery_suggestion = recovery_suggestion
        self.original_exception = original_exception
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with context and suggestions."""
        lines = [f"GUI Generation Error: {self.message}"]
        
        if self.context:
            lines.append("\nContext:")
            for key, value in self.context.items():
                lines.append(f"  {key}: {value}")
        
        if self.recovery_suggestion:
            lines.append(f"\n💡 Suggestion: {self.recovery_suggestion}")
        
        if self.original_exception:
            lines.append(f"\nOriginal error: {str(self.original_exception)}")
        
        return "\n".join(lines)


class ComponentParsingError(GUIGenerationError):
    """Error during component parsing."""
    
    def __init__(
        self,
        component_type: str,
        component_id: Optional[str] = None,
        message: str = "",
        **kwargs
    ):
        context = {
            "component_type": component_type,
            "component_id": component_id or "unknown"
        }
        context.update(kwargs.get("context", {}))
        
        full_message = message or f"Failed to parse {component_type} component"
        
        super().__init__(
            message=full_message,
            context=context,
            recovery_suggestion=kwargs.get("recovery_suggestion", 
                "Check component structure and required properties"),
            original_exception=kwargs.get("original_exception")
        )


class StyleParsingError(GUIGenerationError):
    """Error during style parsing."""
    
    def __init__(
        self,
        selector: Optional[str] = None,
        property_name: Optional[str] = None,
        message: str = "",
        **kwargs
    ):
        context = {
            "selector": selector or "unknown",
            "property": property_name or "unknown"
        }
        context.update(kwargs.get("context", {}))
        
        full_message = message or f"Failed to parse style for selector '{selector}'"
        
        super().__init__(
            message=full_message,
            context=context,
            recovery_suggestion=kwargs.get("recovery_suggestion",
                "Check CSS property syntax and values"),
            original_exception=kwargs.get("original_exception")
        )


class DataBindingError(GUIGenerationError):
    """Error during data binding resolution."""
    
    def __init__(
        self,
        binding_path: str,
        component_name: Optional[str] = None,
        message: str = "",
        **kwargs
    ):
        context = {
            "binding_path": binding_path,
            "component": component_name or "unknown"
        }
        context.update(kwargs.get("context", {}))
        
        full_message = message or f"Failed to resolve data binding '{binding_path}'"
        
        super().__init__(
            message=full_message,
            context=context,
            recovery_suggestion=kwargs.get("recovery_suggestion",
                "Verify data binding path and ensure the entity exists"),
            original_exception=kwargs.get("original_exception")
        )


class LayoutError(GUIGenerationError):
    """Error during layout processing."""
    
    def __init__(
        self,
        layout_type: str,
        message: str = "",
        **kwargs
    ):
        context = {"layout_type": layout_type}
        context.update(kwargs.get("context", {}))
        
        full_message = message or f"Failed to process {layout_type} layout"
        
        super().__init__(
            message=full_message,
            context=context,
            recovery_suggestion=kwargs.get("recovery_suggestion",
                "Check layout properties and child component structure"),
            original_exception=kwargs.get("original_exception")
        )


class CodeGenerationError(GUIGenerationError):
    """Error during code generation phase."""
    
    def __init__(
        self,
        generation_phase: str,
        output_file: Optional[str] = None,
        message: str = "",
        **kwargs
    ):
        context = {
            "phase": generation_phase,
            "output_file": output_file or "unknown"
        }
        context.update(kwargs.get("context", {}))
        
        full_message = message or f"Failed during {generation_phase} phase"
        
        super().__init__(
            message=full_message,
            context=context,
            recovery_suggestion=kwargs.get("recovery_suggestion",
                "Check template syntax and model data"),
            original_exception=kwargs.get("original_exception")
        )


class ErrorHandler:
    """Centralized error handling and recovery."""
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize error handler.
        
        Args:
            strict_mode: If True, raise all errors. If False, attempt recovery.
        """
        self.strict_mode = strict_mode
        self.errors: List[GUIGenerationError] = []
        self.warnings: List[str] = []
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        recovery_action: Optional[callable] = None
    ) -> Any:
        """
        Handle an error with optional recovery.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            recovery_action: Optional function to call for recovery
            
        Returns:
            Result of recovery_action if provided and successful, None otherwise
            
        Raises:
            GUIGenerationError: If strict_mode is True or recovery fails
        """
        # Wrap the error if it's not already a GUIGenerationError
        if not isinstance(error, GUIGenerationError):
            wrapped_error = GUIGenerationError(
                message=str(error),
                context=context,
                original_exception=error
            )
        else:
            wrapped_error = error
        
        # Log the error
        logger.error(wrapped_error.format_message())
        logger.debug("Stack trace:", exc_info=True)
        
        # Store the error
        self.errors.append(wrapped_error)
        
        # In strict mode, always raise
        if self.strict_mode:
            raise wrapped_error
        
        # Attempt recovery if provided
        if recovery_action:
            try:
                logger.info("Attempting error recovery...")
                result = recovery_action()
                logger.info("Error recovery successful")
                return result
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")
                raise wrapped_error
        
        # No recovery possible
        logger.warning("No recovery action provided, continuing with degraded functionality")
        return None
    
    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get a summary of all errors and warnings."""
        return {
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [error.format_message() for error in self.errors],
            "warnings": self.warnings
        }
    
    def has_errors(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings occurred."""
        return len(self.warnings) > 0
    
    def clear(self):
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()


class ErrorRecovery:
    """Common error recovery strategies."""
    
    @staticmethod
    def use_default_value(default: Any):
        """Recovery strategy: Use a default value."""
        return lambda: default
    
    @staticmethod
    def skip_component():
        """Recovery strategy: Skip the problematic component."""
        return lambda: None
    
    @staticmethod
    def use_fallback_parser(fallback_fn: callable, *args, **kwargs):
        """Recovery strategy: Use a fallback parsing function."""
        return lambda: fallback_fn(*args, **kwargs)
    
    @staticmethod
    def log_and_continue(message: str):
        """Recovery strategy: Log a message and continue."""
        def recovery():
            logger.info(f"Recovery: {message}")
            return None
        return recovery


def safe_parse(
    parse_fn: callable,
    error_handler: ErrorHandler,
    error_type: type = ComponentParsingError,
    context: Optional[Dict[str, Any]] = None,
    recovery_action: Optional[callable] = None,
    *args,
    **kwargs
) -> Any:
    """
    Safely execute a parsing function with error handling.
    
    Args:
        parse_fn: The parsing function to execute
        error_handler: ErrorHandler instance
        error_type: Type of error to raise on failure
        context: Additional context for error reporting
        recovery_action: Optional recovery function
        *args, **kwargs: Arguments to pass to parse_fn
        
    Returns:
        Result of parse_fn or recovery_action
    """
    try:
        return parse_fn(*args, **kwargs)
    except Exception as e:
        error = error_type(
            message=str(e),
            context=context,
            original_exception=e
        ) if not isinstance(e, GUIGenerationError) else e
        
        return error_handler.handle_error(error, context, recovery_action)


def format_error_report(error_handler: ErrorHandler) -> str:
    """
    Format a comprehensive error report.
    
    Args:
        error_handler: ErrorHandler with collected errors/warnings
        
    Returns:
        Formatted error report string
    """
    summary = error_handler.get_error_summary()
    
    lines = ["=" * 80]
    lines.append("GUI GENERATION ERROR REPORT")
    lines.append("=" * 80)
    lines.append(f"Total Errors: {summary['error_count']}")
    lines.append(f"Total Warnings: {summary['warning_count']}")
    lines.append("")
    
    if summary['errors']:
        lines.append("ERRORS:")
        lines.append("-" * 80)
        for idx, error in enumerate(summary['errors'], 1):
            lines.append(f"\n{idx}. {error}")
        lines.append("")
    
    if summary['warnings']:
        lines.append("WARNINGS:")
        lines.append("-" * 80)
        for idx, warning in enumerate(summary['warnings'], 1):
            lines.append(f"{idx}. {warning}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

