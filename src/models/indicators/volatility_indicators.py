"""
Volatility Indicators for the Algorithmic Trading System

This module implements volatility indicators such as Bollinger Bands, ATR, and Keltner Channels.
"""

from typing import Dict, Any, List

import pandas as pd
import numpy as np

from src.indicators.base_indicator import BaseIndicator


class BollingerBands(BaseIndicator):
    """Bollinger Bands indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Bollinger Bands indicator.
        
        Args:
            params: Dictionary with parameters:
                - period: Look-back period (default 20)
                - std_dev: Standard deviation multiplier (default 2.0)
                - source: Price source column (default 'close')
        """
        default_params = {
            'period': 20,
            'std_dev': 2.0,
            'source': 'close'
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bollinger Bands values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        std_dev = self.params['std_dev']
        source = self.params['source']
        
        if source not in data.columns:
            raise ValueError(f"Source column '{source}' not found in data")
        
        df = data.copy()
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df[source].rolling(window=period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df[source].rolling(window=period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * std_dev)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * std_dev)
        
        # Calculate bandwidth and %B
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Calculate %B (position within the bands)
        df['bb_pct_b'] = (df[source] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Drop intermediate calculation
        df = df.drop(['bb_std'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['bb_middle', 'bb_upper', 'bb_lower', 'bb_bandwidth', 'bb_pct_b']


class ATR(BaseIndicator):
    """Average True Range indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the ATR indicator.
        
        Args:
            params: Dictionary with parameters:
                - period: Look-back period (default 14)
                - ema_smoothing: Whether to use EMA smoothing (default False)
        """
        default_params = {
            'period': 14,
            'ema_smoothing': False
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added ATR columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        ema_smoothing = self.params['ema_smoothing']
        
        df = data.copy()
        
        # Calculate True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        if ema_smoothing:
            df['atr'] = df['tr'].ewm(span=period, adjust=False).mean()
        else:
            df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Calculate normalized ATR (ATR as percentage of price)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # Drop intermediate calculations
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['atr', 'atr_pct']


class KeltnerChannels(BaseIndicator):
    """Keltner Channels indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Keltner Channels indicator.
        
        Args:
            params: Dictionary with parameters:
                - ema_period: EMA period for middle line (default 20)
                - atr_period: ATR period (default 10)
                - atr_multiplier: ATR multiplier for channel width (default 2.0)
                - use_sma: Whether to use SMA instead of EMA for middle line (default False)
        """
        default_params = {
            'ema_period': 20,
            'atr_period': 10,
            'atr_multiplier': 2.0,
            'use_sma': False
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Keltner Channels values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Keltner Channels columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        ema_period = self.params['ema_period']
        atr_period = self.params['atr_period']
        atr_multiplier = self.params['atr_multiplier']
        use_sma = self.params['use_sma']
        
        df = data.copy()
        
        # Calculate middle line (EMA or SMA of typical price)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        if use_sma:
            df['kc_middle'] = typical_price.rolling(window=ema_period).mean()
        else:
            df['kc_middle'] = typical_price.ewm(span=ema_period, adjust=False).mean()
        
        # Calculate ATR
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=atr_period).mean()
        
        # Calculate upper and lower bands
        df['kc_upper'] = df['kc_middle'] + (df['atr'] * atr_multiplier)
        df['kc_lower'] = df['kc_middle'] - (df['atr'] * atr_multiplier)
        
        # Calculate bandwidth
        df['kc_bandwidth'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
        
        # Drop intermediate calculations
        df = df.drop(['tr1', 'tr2', 'tr3', 'tr'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['kc_middle', 'kc_upper', 'kc_lower', 'atr', 'kc_bandwidth']


class HistoricalVolatility(BaseIndicator):
    """Historical Volatility indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Historical Volatility indicator.
        
        Args:
            params: Dictionary with parameters:
                - period: Look-back period (default 20)
                - trading_periods: Number of trading periods in a year (default 252)
                - use_log_returns: Whether to use log returns (default True)
        """
        default_params = {
            'period': 20,
            'trading_periods': 252,  # Trading days in a year
            'use_log_returns': True
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Historical Volatility values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Historical Volatility column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        trading_periods = self.params['trading_periods']
        use_log_returns = self.params['use_log_returns']
        
        df = data.copy()
        
        # Calculate returns
        if use_log_returns:
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
        else:
            df['returns'] = df['close'].pct_change()
        
        # Calculate historical volatility
        # Annualized standard deviation of returns
        df['hist_vol'] = df['returns'].rolling(window=period).std() * np.sqrt(trading_periods) * 100
        
        # Drop intermediate calculations
        df = df.drop(['returns'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['hist_vol']


class ChaikinVolatility(BaseIndicator):
    """Chaikin Volatility indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Chaikin Volatility indicator.
        
        Args:
            params: Dictionary with parameters:
                - ema_period: EMA smoothing period (default 10)
                - change_period: Rate of change period (default 10)
        """
        default_params = {
            'ema_period': 10,
            'change_period': 10
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Chaikin Volatility values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Chaikin Volatility column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        ema_period = self.params['ema_period']
        change_period = self.params['change_period']
        
        df = data.copy()
        
        # Calculate high-low range
        df['hl_range'] = df['high'] - df['low']
        
        # Apply EMA smoothing to the high-low range
        df['hl_range_ema'] = df['hl_range'].ewm(span=ema_period, adjust=False).mean()
        
        # Calculate the rate of change of the smoothed range
        df['chaikin_vol'] = (
            (df['hl_range_ema'] - df['hl_range_ema'].shift(change_period)) / 
            df['hl_range_ema'].shift(change_period)
        ) * 100
        
        # Drop intermediate calculations
        df = df.drop(['hl_range', 'hl_range_ema'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['chaikin_vol'] 