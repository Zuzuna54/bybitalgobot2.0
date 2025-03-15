"""
Trend Indicators for the Algorithmic Trading System

This module implements trend indicators such as Moving Averages, ADX, and Ichimoku Cloud.
"""

from typing import Dict, Any, List

import pandas as pd
import numpy as np

from src.indicators.base_indicator import BaseIndicator


class MovingAverage(BaseIndicator):
    """Moving Average indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Moving Average indicator.
        
        Args:
            params: Dictionary with MA parameters:
                - period: MA period (default 20)
                - type: MA type ('sma', 'ema', 'wma') (default 'sma')
                - source: Price source column (default 'close')
        """
        default_params = {
            'period': 20,
            'type': 'sma',
            'source': 'close'
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Moving Average values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added MA column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        ma_type = self.params['type']
        source = self.params['source']
        
        if source not in data.columns:
            raise ValueError(f"Source column '{source}' not found in data")
        
        df = data.copy()
        
        # Calculate moving average
        ma_series = self.moving_average(df[source], period, ma_type)
        
        # Add to dataframe with a descriptive column name
        col_name = f"{ma_type}_{period}"
        df[col_name] = ma_series
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        ma_type = self.params['type']
        period = self.params['period']
        return [f"{ma_type}_{period}"]


class MovingAverageCrossover(BaseIndicator):
    """Moving Average Crossover indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Moving Average Crossover indicator.
        
        Args:
            params: Dictionary with parameters:
                - fast_period: Fast MA period (default 9)
                - slow_period: Slow MA period (default 21)
                - type: MA type ('sma', 'ema', 'wma') (default 'ema')
                - source: Price source column (default 'close')
        """
        default_params = {
            'fast_period': 9,
            'slow_period': 21,
            'type': 'ema',
            'source': 'close'
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Moving Average Crossover values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added MA crossover columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        ma_type = self.params['type']
        source = self.params['source']
        
        if source not in data.columns:
            raise ValueError(f"Source column '{source}' not found in data")
        
        df = data.copy()
        
        # Calculate fast and slow MAs
        fast_ma = self.moving_average(df[source], fast_period, ma_type)
        slow_ma = self.moving_average(df[source], slow_period, ma_type)
        
        # Add MAs to dataframe
        fast_col = f"{ma_type}_fast_{fast_period}"
        slow_col = f"{ma_type}_slow_{slow_period}"
        
        df[fast_col] = fast_ma
        df[slow_col] = slow_ma
        
        # Calculate crossover signal
        df['ma_crossover'] = 0  # Initialize with no signal
        
        # Generate crossover signals: 1 for bullish crossover, -1 for bearish crossover
        df['ma_crossover'] = np.where(
            (df[fast_col] > df[slow_col]) & (df[fast_col].shift(1) <= df[slow_col].shift(1)),
            1,  # Bullish crossover
            np.where(
                (df[fast_col] < df[slow_col]) & (df[fast_col].shift(1) >= df[slow_col].shift(1)),
                -1,  # Bearish crossover
                0  # No crossover
            )
        )
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        ma_type = self.params['type']
        fast_period = self.params['fast_period']
        slow_period = self.params['slow_period']
        return [
            f"{ma_type}_fast_{fast_period}",
            f"{ma_type}_slow_{slow_period}",
            'ma_crossover'
        ]


class ADX(BaseIndicator):
    """Average Directional Index indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the ADX indicator.
        
        Args:
            params: Dictionary with ADX parameters:
                - period: ADX period (default 14)
        """
        default_params = {'period': 14}
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ADX values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added ADX, +DI, and -DI columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        df = data.copy()
        
        # Calculate True Range
        df['tr0'] = abs(df['high'] - df['low'])
        df['tr1'] = abs(df['high'] - df['close'].shift(1))
        df['tr2'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr0', 'tr1', 'tr2']].max(axis=1)
        
        # Calculate Directional Movement
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        
        # Calculate +DM and -DM
        df['+dm'] = np.where(
            (df['up_move'] > df['down_move']) & (df['up_move'] > 0),
            df['up_move'],
            0
        )
        df['-dm'] = np.where(
            (df['down_move'] > df['up_move']) & (df['down_move'] > 0),
            df['down_move'],
            0
        )
        
        # Calculate smoothed versions
        df['smoothed_tr'] = df['tr'].rolling(window=period).sum()
        df['smoothed_+dm'] = df['+dm'].rolling(window=period).sum()
        df['smoothed_-dm'] = df['-dm'].rolling(window=period).sum()
        
        # Handle division by zero
        df['smoothed_tr'] = df['smoothed_tr'].replace(0, np.finfo(float).eps)
        
        # Calculate +DI and -DI
        df['+di'] = 100 * df['smoothed_+dm'] / df['smoothed_tr']
        df['-di'] = 100 * df['smoothed_-dm'] / df['smoothed_tr']
        
        # Calculate DX
        df['dx'] = 100 * abs(df['+di'] - df['-di']) / (df['+di'] + df['-di'])
        
        # Replace NaN values in DX with zeros
        df['dx'] = df['dx'].fillna(0)
        
        # Calculate ADX (smoothed DX)
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        # Drop intermediate calculations
        df = df.drop(['tr0', 'tr1', 'tr2', 'tr', 'up_move', 'down_move', 
                     '+dm', '-dm', 'smoothed_tr', 'smoothed_+dm', 'smoothed_-dm', 'dx'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['adx', '+di', '-di']


class IchimokuCloud(BaseIndicator):
    """Ichimoku Cloud indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Ichimoku Cloud indicator.
        
        Args:
            params: Dictionary with Ichimoku parameters:
                - tenkan_period: Tenkan-sen period (default 9)
                - kijun_period: Kijun-sen period (default 26)
                - senkou_b_period: Senkou Span B period (default 52)
                - displacement: Displacement period (default 26)
        """
        default_params = {
            'tenkan_period': 9,
            'kijun_period': 26,
            'senkou_b_period': 52,
            'displacement': 26
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Ichimoku Cloud values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Ichimoku Cloud columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        tenkan_period = self.params['tenkan_period']
        kijun_period = self.params['kijun_period']
        senkou_b_period = self.params['senkou_b_period']
        displacement = self.params['displacement']
        
        df = data.copy()
        
        # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for tenkan_period
        df['tenkan_sen'] = (
            df['high'].rolling(window=tenkan_period).max() + 
            df['low'].rolling(window=tenkan_period).min()
        ) / 2
        
        # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for kijun_period
        df['kijun_sen'] = (
            df['high'].rolling(window=kijun_period).max() + 
            df['low'].rolling(window=kijun_period).min()
        ) / 2
        
        # Calculate Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2 displaced forward
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(displacement)
        
        # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for senkou_b_period displaced forward
        df['senkou_span_b'] = (
            (df['high'].rolling(window=senkou_b_period).max() + 
             df['low'].rolling(window=senkou_b_period).min()) / 2
        ).shift(displacement)
        
        # Calculate Chikou Span (Lagging Span): Current closing price shifted backwards
        df['chikou_span'] = df['close'].shift(-displacement)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']


class Supertrend(BaseIndicator):
    """Supertrend indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Supertrend indicator.
        
        Args:
            params: Dictionary with Supertrend parameters:
                - period: ATR period (default 10)
                - multiplier: ATR multiplier (default 3.0)
        """
        default_params = {
            'period': 10,
            'multiplier': 3.0
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Supertrend columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        multiplier = self.params['multiplier']
        
        df = data.copy()
        
        # Calculate ATR
        df['tr'] = df.apply(
            lambda row: max(
                row['high'] - row['low'],
                abs(row['high'] - row['close'].shift(1)),
                abs(row['low'] - row['close'].shift(1))
            ),
            axis=1
        )
        df['atr'] = df['tr'].rolling(window=period).mean()
        
        # Calculate basic upper and lower bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + multiplier * df['atr']
        df['basic_lower'] = (df['high'] + df['low']) / 2 - multiplier * df['atr']
        
        # Initialize Supertrend columns
        df['supertrend'] = 0.0
        df['supertrend_direction'] = 0  # 1 for bullish, -1 for bearish
        
        # Calculate Supertrend using iterative logic
        for i in range(1, len(df)):
            # Adjust upper band
            if (df['basic_upper'].iloc[i] < df['basic_upper'].iloc[i-1]) or (df['close'].iloc[i-1] > df['basic_upper'].iloc[i-1]):
                df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
            else:
                df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i-1]
                
            # Adjust lower band
            if (df['basic_lower'].iloc[i] > df['basic_lower'].iloc[i-1]) or (df['close'].iloc[i-1] < df['basic_lower'].iloc[i-1]):
                df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
            else:
                df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i-1]
                
            # Determine Supertrend direction
            if df['close'].iloc[i-1] <= df.loc[df.index[i-1], 'supertrend']:
                df.loc[df.index[i], 'supertrend_direction'] = -1  # Bearish
            else:
                df.loc[df.index[i], 'supertrend_direction'] = 1  # Bullish
                
            # Calculate Supertrend value
            if (df['supertrend_direction'].iloc[i] == 1) and (df['close'].iloc[i] < df.loc[df.index[i], 'final_upper']):
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_upper']
            elif (df['supertrend_direction'].iloc[i] == -1) and (df['close'].iloc[i] > df.loc[df.index[i], 'final_lower']):
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lower']
            elif df['supertrend_direction'].iloc[i] == 1:
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_lower']
            else:
                df.loc[df.index[i], 'supertrend'] = df.loc[df.index[i], 'final_upper']
        
        # Drop intermediate calculations
        df = df.drop(['tr', 'basic_upper', 'basic_lower', 'final_upper', 'final_lower'], axis=1, errors='ignore')
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['atr', 'supertrend', 'supertrend_direction'] 