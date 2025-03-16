"""
Dashboard Update Service Module (DEPRECATED)

This module is being refactored into smaller, more maintainable modules.
Please use the new modules in the src/dashboard/services/update_service/ package:

- service.py: Core UpdateService class
- handlers.py: Functions for registering and managing update handlers
- utils.py: Utility functions for the update service
- config.py: Configuration defaults and settings

This file is kept for backward compatibility but will be removed in a future version.
"""

import warnings

# Add deprecation warning
warnings.warn(
    "The update_service.py module is deprecated and will be removed in a future version. "
    "Please use the update_service package instead. "
    "Import from src.dashboard.services.update_service directly, which now refers to the package.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export all update service functions from the new package
from src.dashboard.services.update_service import *
