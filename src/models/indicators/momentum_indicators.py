"""
Momentum Indicators for the Algorithmic Trading System

This module implements momentum indicators such as RSI, MACD, and Stochastic Oscillator.
"""

from typing import Dict, Any, List

import pandas as pd
import numpy as np

from src.indicators.base_indicator import BaseIndicator


class RSI(BaseIndicator):
    """Relative Strength Index indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the RSI indicator.
        
        Args:
            params: Dictionary with RSI parameters:
                - period: Look-back period (default 14)
        """
        default_params = {'period': 14}
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added RSI column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        df = data.copy()
        
        # Calculate price changes
        delta = df['close'].diff()
        
        # Create gain and loss series
        gain = delta.copy()
        loss = delta.copy()
        
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = -loss  # Make loss positive
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Handle division by zero
        avg_loss = avg_loss.replace(0, np.finfo(float).eps)
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # Add RSI to the dataframe
        df['rsi'] = rsi
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['rsi']


class MACD(BaseIndicator):
    """Moving Average Convergence Divergence indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the MACD indicator.
        
        Args:
            params: Dictionary with MACD parameters:
                - fast_period: Fast EMA period (default 12)
                - slow_period: Slow EMA period (default 26)
                - signal_period: Signal line period (default 9)
        """
        default_params = {
            'fast_period': 12,
            'slow_period': 26,
            'signal_period': 9
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MACD values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added MACD columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        signal_period = self.params['signal_period']
        
        df = data.copy()
        
        # Calculate fast and slow EMAs
        fast_ema = df['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line and signal line
        df['macd_line'] = fast_ema - slow_ema
        df['macd_signal'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['macd_line', 'macd_signal', 'macd_histogram']


class StochasticOscillator(BaseIndicator):
    """Stochastic Oscillator indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Stochastic Oscillator.
        
        Args:
            params: Dictionary with Stochastic parameters:
                - k_period: %K period (default 14)
                - k_slowing: %K slowing period (default 3)
                - d_period: %D period (default 3)
        """
        default_params = {
            'k_period': 14,
            'k_slowing': 3,
            'd_period': 3
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Stochastic columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        k_period = self.params['k_period']
        k_slowing = self.params['k_slowing']
        d_period = self.params['d_period']
        
        df = data.copy()
        
        # Calculate %K
        # Formula: %K = 100 * (close - lowest low) / (highest high - lowest low)
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        # Handle case where high_max equals low_min
        denom = high_max - low_min
        denom = denom.replace(0, np.finfo(float).eps)
        
        raw_k = 100 * (df['close'] - low_min) / denom
        
        # Apply slowing period to %K
        df['stoch_k'] = raw_k.rolling(window=k_slowing).mean()
        
        # Calculate %D (SMA of %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['stoch_k', 'stoch_d']


class MomentumIndex(BaseIndicator):
    """Momentum indicator measuring the rate of price change."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Momentum indicator.
        
        Args:
            params: Dictionary with Momentum parameters:
                - period: Look-back period (default 10)
                - normalize: Whether to normalize the values (default False)
        """
        default_params = {
            'period': 10,
            'normalize': False
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Momentum values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Momentum column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        normalize = self.params['normalize']
        
        df = data.copy()
        
        # Calculate momentum
        df['momentum'] = df['close'].diff(period)
        
        # Normalize if requested
        if normalize:
            df['momentum'] = df['momentum'] / df['close'].shift(period) * 100
            
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['momentum'] 