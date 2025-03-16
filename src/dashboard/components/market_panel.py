"""
Market Panel Component for the Trading Dashboard

This module is maintained for backward compatibility and re-exports
the functionality from the new modular market panel components.

For new code, please use the modular imports directly from:
src.dashboard.components.market
"""

# Re-export from modular structure
from src.dashboard.components.market import (
    create_market_panel,
    register_market_callbacks,
)

# For backward compatibility
__all__ = ["create_market_panel", "register_market_callbacks"]
