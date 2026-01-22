"""
Utility functions and helpers for Stock Analysis Dashboard.
"""

import logging
from functools import wraps
from typing import Callable, Any


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def handle_errors(func: Callable) -> Callable:
    """
    Decorator for error handling.
    
    Args:
        func: Function to wrap with error handling
    
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            import streamlit as st
            st.error(f"An error occurred in {func.__name__}: {str(e)}")
            return None
    return wrapper

