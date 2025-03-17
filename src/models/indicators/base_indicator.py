"""
Base Indicator Class for the Algorithmic Trading System

This module defines the base class for all technical indicators used in the system.
Each specific indicator will inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np


class BaseIndicator(ABC):
    """Base class for all technical indicators."""
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the indicator with parameters.
        
        Args:
            params: Dictionary of indicator parameters
        """
        self.params = params or {}
        self.name = self.__class__.__name__
        
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the indicator values for the given data.
        
        Args:
            data: Pandas DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicator columns
        """
        pass
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the input data has the required columns.
        
        Args:
            data: Pandas DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for column in required_columns:
            if column not in data.columns:
                return False
        return True
    
    def get_column_names(self) -> List[str]:
        """
        Get the names of the columns added by this indicator.
        
        Returns:
            List of column names
        """
        # Default implementation, should be overridden by child classes
        return [f"{self.name.lower()}"]
    
    @staticmethod
    def normalize_value(value: float, min_val: float, max_val: float) -> float:
        """
        Normalize a value to a 0-1 range.
        
        Args:
            value: The value to normalize
            min_val: Minimum value in the range
            max_val: Maximum value in the range
            
        Returns:
            Normalized value between 0 and 1
        """
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    @staticmethod
    def moving_average(data: pd.Series, window: int, ma_type: str = 'sma') -> pd.Series:
        """
        Calculate moving average of a series.
        
        Args:
            data: Data series
            window: Window size
            ma_type: Type of moving average ('sma', 'ema', 'wma')
            
        Returns:
            Series with moving average values
        """
        if ma_type.lower() == 'sma':
            return data.rolling(window=window).mean()
        elif ma_type.lower() == 'ema':
            return data.ewm(span=window, adjust=False).mean()
        elif ma_type.lower() == 'wma':
            # Weighted moving average with linearly decreasing weights
            weights = np.arange(1, window + 1)
            wma = data.rolling(window=window).apply(
                lambda x: np.sum(weights * x) / weights.sum(), raw=True
            )
            return wma
        else:
            raise ValueError(f"Unsupported moving average type: {ma_type}") 