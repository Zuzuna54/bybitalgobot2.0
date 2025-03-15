"""
Volume-Based Indicators for the Algorithmic Trading System

This module implements volume-based indicators such as VWAP, OBV, and MFI.
"""

from typing import Dict, Any, List
from datetime import datetime

import pandas as pd
import numpy as np

from src.indicators.base_indicator import BaseIndicator


class VWAP(BaseIndicator):
    """Volume Weighted Average Price indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the VWAP indicator.
        
        Args:
            params: Dictionary with parameters:
                - reset_period: Period to reset VWAP ('daily', 'weekly', None) (default 'daily')
        """
        default_params = {
            'reset_period': 'daily'
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added VWAP column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        reset_period = self.params['reset_period']
        
        df = data.copy()
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate price * volume
        df['pv'] = df['typical_price'] * df['volume']
        
        # Calculate VWAP based on reset period
        if reset_period == 'daily':
            # Get day from index
            df['day'] = df.index.date
            
            # Group by day and calculate cumulative values
            df['cumulative_pv'] = df.groupby('day')['pv'].cumsum()
            df['cumulative_volume'] = df.groupby('day')['volume'].cumsum()
            
            # Calculate VWAP
            # Handle division by zero
            df['cumulative_volume'] = df['cumulative_volume'].replace(0, np.finfo(float).eps)
            df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
            
            # Drop intermediate columns
            df = df.drop(['day', 'typical_price', 'pv', 'cumulative_pv', 'cumulative_volume'], axis=1)
            
        elif reset_period == 'weekly':
            # Get week from index
            df['week'] = df.index.isocalendar().week
            df['year'] = df.index.year
            df['year_week'] = df['year'].astype(str) + '-' + df['week'].astype(str)
            
            # Group by week and calculate cumulative values
            df['cumulative_pv'] = df.groupby('year_week')['pv'].cumsum()
            df['cumulative_volume'] = df.groupby('year_week')['volume'].cumsum()
            
            # Calculate VWAP
            # Handle division by zero
            df['cumulative_volume'] = df['cumulative_volume'].replace(0, np.finfo(float).eps)
            df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
            
            # Drop intermediate columns
            df = df.drop(['week', 'year', 'year_week', 'typical_price', 'pv', 'cumulative_pv', 'cumulative_volume'], axis=1)
            
        else:
            # No reset, calculate cumulative values from beginning
            df['cumulative_pv'] = df['pv'].cumsum()
            df['cumulative_volume'] = df['volume'].cumsum()
            
            # Calculate VWAP
            # Handle division by zero
            df['cumulative_volume'] = df['cumulative_volume'].replace(0, np.finfo(float).eps)
            df['vwap'] = df['cumulative_pv'] / df['cumulative_volume']
            
            # Drop intermediate columns
            df = df.drop(['typical_price', 'pv', 'cumulative_pv', 'cumulative_volume'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['vwap']


class OBV(BaseIndicator):
    """On-Balance Volume indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the OBV indicator.
        
        Args:
            params: Dictionary with parameters:
                - use_close: Whether to use close price for calculation (default True)
                             If False, uses typical price (high+low+close)/3
                - ma_period: Period for OBV moving average (default 20)
                - ma_type: Moving average type ('sma', 'ema') (default 'sma')
        """
        default_params = {
            'use_close': True,
            'ma_period': 20,
            'ma_type': 'sma'
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OBV values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added OBV column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        use_close = self.params['use_close']
        ma_period = self.params['ma_period']
        ma_type = self.params['ma_type']
        
        df = data.copy()
        
        # Determine which price to use
        if use_close:
            price = df['close']
        else:
            price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate price change direction
        df['price_change'] = price.diff()
        
        # Calculate OBV
        df['obv'] = 0
        
        # First row has no price change, so OBV starts at first volume
        if len(df) > 0:
            df.loc[df.index[0], 'obv'] = df['volume'].iloc[0]
        
        # Calculate OBV for rest of rows
        for i in range(1, len(df)):
            if df['price_change'].iloc[i] > 0:
                # If price increased, add volume
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] + df['volume'].iloc[i]
            elif df['price_change'].iloc[i] < 0:
                # If price decreased, subtract volume
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1] - df['volume'].iloc[i]
            else:
                # If price unchanged, OBV remains the same
                df.loc[df.index[i], 'obv'] = df['obv'].iloc[i-1]
        
        # Calculate OBV moving average
        if ma_type.lower() == 'sma':
            df['obv_ma'] = df['obv'].rolling(window=ma_period).mean()
        elif ma_type.lower() == 'ema':
            df['obv_ma'] = df['obv'].ewm(span=ma_period, adjust=False).mean()
        
        # Drop intermediate columns
        df = df.drop(['price_change'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['obv', 'obv_ma']


class MFI(BaseIndicator):
    """Money Flow Index indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the MFI indicator.
        
        Args:
            params: Dictionary with parameters:
                - period: Look-back period (default 14)
                - overbought: Overbought level (default 80)
                - oversold: Oversold level (default 20)
        """
        default_params = {
            'period': 14,
            'overbought': 80,
            'oversold': 20
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate MFI values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added MFI column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        overbought = self.params['overbought']
        oversold = self.params['oversold']
        
        df = data.copy()
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate raw money flow
        df['money_flow'] = df['typical_price'] * df['volume']
        
        # Determine if money flow is positive or negative
        df['typical_price_shift'] = df['typical_price'].shift(1)
        df['positive_flow'] = np.where(df['typical_price'] > df['typical_price_shift'], df['money_flow'], 0)
        df['negative_flow'] = np.where(df['typical_price'] < df['typical_price_shift'], df['money_flow'], 0)
        
        # Handle first row where shift produces NaN
        if len(df) > 0:
            df.loc[df.index[0], 'positive_flow'] = 0
            df.loc[df.index[0], 'negative_flow'] = 0
        
        # Calculate positive and negative money flow sums over period
        df['positive_flow_sum'] = df['positive_flow'].rolling(window=period).sum()
        df['negative_flow_sum'] = df['negative_flow'].rolling(window=period).sum()
        
        # Calculate money flow ratio
        # Handle division by zero
        df['negative_flow_sum'] = df['negative_flow_sum'].replace(0, np.finfo(float).eps)
        df['money_flow_ratio'] = df['positive_flow_sum'] / df['negative_flow_sum']
        
        # Calculate MFI
        df['mfi'] = 100 - (100 / (1 + df['money_flow_ratio']))
        
        # Add overbought/oversold signals
        df['mfi_overbought'] = df['mfi'] > overbought
        df['mfi_oversold'] = df['mfi'] < oversold
        
        # Drop intermediate columns
        df = df.drop(['typical_price', 'money_flow', 'typical_price_shift', 
                     'positive_flow', 'negative_flow', 'positive_flow_sum', 
                     'negative_flow_sum', 'money_flow_ratio'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['mfi', 'mfi_overbought', 'mfi_oversold']


class ChaikinMoneyFlow(BaseIndicator):
    """Chaikin Money Flow indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Chaikin Money Flow indicator.
        
        Args:
            params: Dictionary with parameters:
                - period: Look-back period (default 20)
        """
        default_params = {'period': 20}
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Chaikin Money Flow values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added CMF column
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        period = self.params['period']
        
        df = data.copy()
        
        # Calculate Money Flow Multiplier
        df['mf_multiplier'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        
        # Handle division by zero
        df.loc[df['high'] == df['low'], 'mf_multiplier'] = 0
        
        # Calculate Money Flow Volume
        df['mf_volume'] = df['mf_multiplier'] * df['volume']
        
        # Calculate Chaikin Money Flow
        df['cmf'] = df['mf_volume'].rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        
        # Drop intermediate columns
        df = df.drop(['mf_multiplier', 'mf_volume'], axis=1)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        return ['cmf']


class VolumeProfile(BaseIndicator):
    """Volume Profile indicator."""
    
    def __init__(self, params: Dict[str, Any] = None):
        """
        Initialize the Volume Profile indicator.
        
        Args:
            params: Dictionary with parameters:
                - num_bins: Number of price bins (default 10)
                - lookback: Number of periods to look back (default 100)
        """
        default_params = {
            'num_bins': 10,
            'lookback': 100
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Profile values for the given data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added Volume Profile columns
        """
        if not self.validate_data(data):
            raise ValueError("Input data missing required columns")
        
        num_bins = self.params['num_bins']
        lookback = self.params['lookback']
        
        df = data.copy()
        
        # Only consider the specified lookback period
        if len(df) > lookback:
            analysis_df = df.iloc[-lookback:]
        else:
            analysis_df = df.copy()
        
        # Calculate range for bins
        price_min = analysis_df['low'].min()
        price_max = analysis_df['high'].max()
        
        # Create bins
        bin_edges = np.linspace(price_min, price_max, num_bins + 1)
        bin_width = (price_max - price_min) / num_bins
        
        # Create bin labels (middle of each bin)
        bin_centers = bin_edges[:-1] + bin_width / 2
        bin_labels = [f"bin_{i}" for i in range(len(bin_centers))]
        
        # Assign volume to bins
        volume_profile = np.zeros(num_bins)
        
        for idx, row in analysis_df.iterrows():
            # Calculate how much of the range is in each bin
            lower_bin = np.searchsorted(bin_edges, row['low'], side='right') - 1
            upper_bin = np.searchsorted(bin_edges, row['high'], side='left')
            
            # Clip to ensure bin indices are valid
            lower_bin = max(0, lower_bin)
            upper_bin = min(num_bins, upper_bin)
            
            # Simple approach: distribute volume equally among bins
            if upper_bin > lower_bin:
                volume_per_bin = row['volume'] / (upper_bin - lower_bin)
                volume_profile[lower_bin:upper_bin] += volume_per_bin
        
        # Find point of control (price level with highest volume)
        poc_bin = np.argmax(volume_profile)
        poc_price = bin_centers[poc_bin]
        
        # Assign volume profile data to dataframe
        for i, bin_label in enumerate(bin_labels):
            df[bin_label] = volume_profile[i]
        
        df['volume_poc'] = poc_price
        
        # Calculate total volume in upper and lower half (around POC)
        upper_volume = np.sum(volume_profile[poc_bin:])
        lower_volume = np.sum(volume_profile[:poc_bin])
        
        # Calculate volume balance
        total_volume = upper_volume + lower_volume
        if total_volume > 0:
            df['volume_balance'] = (upper_volume - lower_volume) / total_volume
        else:
            df['volume_balance'] = 0
        
        return df
    
    def get_column_names(self) -> List[str]:
        """Get column names added by this indicator."""
        num_bins = self.params['num_bins']
        columns = [f"bin_{i}" for i in range(num_bins)]
        columns.extend(['volume_poc', 'volume_balance'])
        return columns 