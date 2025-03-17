"""
Indicator Manager for the Algorithmic Trading System

This module provides a central manager for handling all technical indicators
and applying them consistently to market data.
"""

from typing import Dict, Any, List, Optional, Type, Union

import pandas as pd
from loguru import logger

from src.indicators.base_indicator import BaseIndicator
from src.indicators.momentum_indicators import RSI, MACD, StochasticOscillator, MomentumIndex
from src.indicators.trend_indicators import MovingAverage, MovingAverageCrossover, ADX, IchimokuCloud, Supertrend
from src.indicators.volatility_indicators import BollingerBands, ATR, KeltnerChannels, HistoricalVolatility, ChaikinVolatility
from src.indicators.volume_indicators import VWAP, OBV, MFI, ChaikinMoneyFlow, VolumeProfile


class IndicatorManager:
    """Manager class for handling all technical indicators."""
    
    def __init__(self):
        """Initialize the indicator manager."""
        self.indicators: Dict[str, BaseIndicator] = {}
        self.indicator_registry: Dict[str, Type[BaseIndicator]] = self._build_indicator_registry()
    
    def _build_indicator_registry(self) -> Dict[str, Type[BaseIndicator]]:
        """
        Build the registry of available indicators.
        
        Returns:
            Dictionary mapping indicator names to indicator classes
        """
        registry = {
            # Momentum Indicators
            'rsi': RSI,
            'macd': MACD,
            'stochastic': StochasticOscillator,
            'momentum': MomentumIndex,
            
            # Trend Indicators
            'ma': MovingAverage,
            'ma_crossover': MovingAverageCrossover,
            'adx': ADX,
            'ichimoku': IchimokuCloud,
            'supertrend': Supertrend,
            
            # Volatility Indicators
            'bollinger': BollingerBands,
            'atr': ATR,
            'keltner': KeltnerChannels,
            'historical_volatility': HistoricalVolatility,
            'chaikin_volatility': ChaikinVolatility,
            
            # Volume Indicators
            'vwap': VWAP,
            'obv': OBV,
            'mfi': MFI,
            'cmf': ChaikinMoneyFlow,
            'volume_profile': VolumeProfile
        }
        return registry
    
    def add_indicator(self, name: str, indicator_type: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an indicator to the manager.
        
        Args:
            name: Unique name for this indicator instance
            indicator_type: Type of indicator (must be in registry)
            params: Parameters for the indicator
            
        Raises:
            ValueError: If indicator_type is not in registry or name is already used
        """
        if name in self.indicators:
            raise ValueError(f"Indicator name '{name}' is already in use")
        
        if indicator_type not in self.indicator_registry:
            raise ValueError(f"Unknown indicator type: {indicator_type}")
        
        indicator_class = self.indicator_registry[indicator_type]
        indicator = indicator_class(params)
        self.indicators[name] = indicator
        
        logger.debug(f"Added indicator: {name} ({indicator_type})")
    
    def remove_indicator(self, name: str) -> None:
        """
        Remove an indicator from the manager.
        
        Args:
            name: Name of the indicator to remove
            
        Raises:
            KeyError: If the indicator doesn't exist
        """
        if name not in self.indicators:
            raise KeyError(f"Indicator '{name}' does not exist")
        
        del self.indicators[name]
        logger.debug(f"Removed indicator: {name}")
    
    def get_indicator(self, name: str) -> BaseIndicator:
        """
        Get an indicator by name.
        
        Args:
            name: Name of the indicator
            
        Returns:
            The indicator instance
            
        Raises:
            KeyError: If the indicator doesn't exist
        """
        if name not in self.indicators:
            raise KeyError(f"Indicator '{name}' does not exist")
        
        return self.indicators[name]
    
    def apply_indicators(self, data: pd.DataFrame, indicator_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply selected indicators to the data.
        
        Args:
            data: Price data DataFrame
            indicator_names: List of indicator names to apply (None for all)
            
        Returns:
            DataFrame with indicators applied
            
        Raises:
            KeyError: If an indicator name is not found
        """
        df = data.copy()
        
        names_to_apply = indicator_names or list(self.indicators.keys())
        
        for name in names_to_apply:
            if name not in self.indicators:
                raise KeyError(f"Indicator '{name}' does not exist")
            
            try:
                df = self.indicators[name].calculate(df)
                logger.debug(f"Applied indicator: {name}")
            except Exception as e:
                logger.error(f"Error applying indicator {name}: {e}")
                raise
        
        return df
    
    def get_available_indicators(self) -> List[str]:
        """
        Get a list of available indicator types.
        
        Returns:
            List of indicator type names
        """
        return list(self.indicator_registry.keys())
    
    def get_current_indicators(self) -> Dict[str, str]:
        """
        Get the current indicators and their types.
        
        Returns:
            Dictionary mapping indicator names to their types
        """
        return {name: indicator.__class__.__name__ for name, indicator in self.indicators.items()}
    
    def configure_default_indicators(self) -> None:
        """Configure a standard set of default indicators."""
        # Momentum Indicators
        self.add_indicator('default_rsi', 'rsi', {'period': 14})
        self.add_indicator('default_macd', 'macd', {'fast_period': 12, 'slow_period': 26, 'signal_period': 9})
        self.add_indicator('default_stochastic', 'stochastic', {'k_period': 14, 'k_slowing': 3, 'd_period': 3})
        
        # Trend Indicators
        self.add_indicator('sma_20', 'ma', {'period': 20, 'type': 'sma'})
        self.add_indicator('sma_50', 'ma', {'period': 50, 'type': 'sma'})
        self.add_indicator('sma_200', 'ma', {'period': 200, 'type': 'sma'})
        self.add_indicator('ema_9', 'ma', {'period': 9, 'type': 'ema'})
        self.add_indicator('ema_21', 'ma', {'period': 21, 'type': 'ema'})
        self.add_indicator('default_adx', 'adx', {'period': 14})
        
        # Volatility Indicators
        self.add_indicator('default_bollinger', 'bollinger', {'period': 20, 'std_dev': 2.0})
        self.add_indicator('default_atr', 'atr', {'period': 14})
        self.add_indicator('default_keltner', 'keltner', {'ema_period': 20, 'atr_period': 10, 'atr_multiplier': 2.0})
        
        # Volume Indicators
        self.add_indicator('default_vwap', 'vwap', {'reset_period': 'daily'})
        self.add_indicator('default_obv', 'obv', {'ma_period': 20})
        self.add_indicator('default_mfi', 'mfi', {'period': 14})
        
        logger.info("Configured default indicators")
    
    def configure_strategy_indicators(self, strategy_config: Dict[str, Any]) -> List[str]:
        """
        Configure indicators needed for a specific strategy.
        
        Args:
            strategy_config: Dictionary with strategy configuration
            
        Returns:
            List of indicator names configured for this strategy
        """
        strategy_name = strategy_config['name']
        strategy_prefix = f"{strategy_name}_"
        configured_indicators = []
        
        if strategy_name == 'ema_crossover':
            # Configure EMA Crossover indicators
            fast_period = strategy_config.get('parameters', {}).get('fast_ema', 9)
            slow_period = strategy_config.get('parameters', {}).get('slow_ema', 21)
            
            fast_name = f"{strategy_prefix}ema_fast"
            slow_name = f"{strategy_prefix}ema_slow"
            crossover_name = f"{strategy_prefix}crossover"
            volume_name = f"{strategy_prefix}volume"
            
            self.add_indicator(fast_name, 'ma', {'period': fast_period, 'type': 'ema'})
            self.add_indicator(slow_name, 'ma', {'period': slow_period, 'type': 'ema'})
            self.add_indicator(crossover_name, 'ma_crossover', {
                'fast_period': fast_period,
                'slow_period': slow_period,
                'type': 'ema'
            })
            self.add_indicator(volume_name, 'obv', {'ma_period': 20})
            
            configured_indicators = [fast_name, slow_name, crossover_name, volume_name]
            
        elif strategy_name == 'rsi_reversal':
            # Configure RSI Reversal indicators
            rsi_period = strategy_config.get('parameters', {}).get('rsi_period', 14)
            stoch_name = f"{strategy_prefix}stochastic"
            rsi_name = f"{strategy_prefix}rsi"
            
            self.add_indicator(rsi_name, 'rsi', {'period': rsi_period})
            self.add_indicator(stoch_name, 'stochastic', {'k_period': 14, 'k_slowing': 3, 'd_period': 3})
            
            configured_indicators = [rsi_name, stoch_name]
            
        elif strategy_name == 'bollinger_mean_reversion':
            # Configure Bollinger Bands Mean Reversion indicators
            bb_period = strategy_config.get('parameters', {}).get('bb_period', 20)
            bb_std_dev = strategy_config.get('parameters', {}).get('bb_std_dev', 2.0)
            rsi_period = strategy_config.get('parameters', {}).get('rsi_period', 14)
            adx_name = f"{strategy_prefix}adx"
            bb_name = f"{strategy_prefix}bollinger"
            rsi_name = f"{strategy_prefix}rsi"
            
            self.add_indicator(bb_name, 'bollinger', {'period': bb_period, 'std_dev': bb_std_dev})
            self.add_indicator(rsi_name, 'rsi', {'period': rsi_period})
            self.add_indicator(adx_name, 'adx', {'period': 14})
            
            configured_indicators = [bb_name, rsi_name, adx_name]
            
        elif strategy_name == 'macd_trend_following':
            # Configure MACD Trend Following indicators
            fast_length = strategy_config.get('parameters', {}).get('fast_length', 12)
            slow_length = strategy_config.get('parameters', {}).get('slow_length', 26)
            signal_length = strategy_config.get('parameters', {}).get('signal_length', 9)
            macd_name = f"{strategy_prefix}macd"
            
            self.add_indicator(macd_name, 'macd', {
                'fast_period': fast_length,
                'slow_period': slow_length,
                'signal_period': signal_length
            })
            
            configured_indicators = [macd_name]
            
        elif strategy_name == 'breakout_trading':
            # Configure Breakout Trading indicators
            lookback = strategy_config.get('parameters', {}).get('lookback_period', 20)
            atr_period = strategy_config.get('parameters', {}).get('atr_period', 14)
            atr_name = f"{strategy_prefix}atr"
            rsi_name = f"{strategy_prefix}rsi"
            volume_name = f"{strategy_prefix}volume"
            
            self.add_indicator(atr_name, 'atr', {'period': atr_period})
            self.add_indicator(rsi_name, 'rsi', {'period': 14})
            self.add_indicator(volume_name, 'obv', {'ma_period': 20})
            
            configured_indicators = [atr_name, rsi_name, volume_name]
            
        elif strategy_name == 'vwap_trend_trading':
            # Configure VWAP Trend Trading indicators
            vwap_period = strategy_config.get('parameters', {}).get('vwap_period', 'daily')
            vwap_name = f"{strategy_prefix}vwap"
            ema_name = f"{strategy_prefix}ema"
            
            self.add_indicator(vwap_name, 'vwap', {'reset_period': vwap_period})
            self.add_indicator(ema_name, 'ma', {'period': 20, 'type': 'ema'})
            
            configured_indicators = [vwap_name, ema_name]
            
        elif strategy_name == 'atr_volatility_scalping':
            # Configure ATR Volatility Scalping indicators
            atr_period = strategy_config.get('parameters', {}).get('atr_period', 14)
            atr_name = f"{strategy_prefix}atr"
            volatility_name = f"{strategy_prefix}volatility"
            
            self.add_indicator(atr_name, 'atr', {'period': atr_period})
            self.add_indicator(volatility_name, 'historical_volatility', {'period': 20})
            
            configured_indicators = [atr_name, volatility_name]
            
        elif strategy_name == 'adx_strength_confirmation':
            # Configure ADX Strength Confirmation indicators
            adx_period = strategy_config.get('parameters', {}).get('adx_period', 14)
            adx_name = f"{strategy_prefix}adx"
            
            self.add_indicator(adx_name, 'adx', {'period': adx_period})
            
            configured_indicators = [adx_name]
            
        elif strategy_name == 'golden_cross':
            # Configure Golden Cross & Death Cross indicators
            fast_ma = strategy_config.get('parameters', {}).get('fast_ma', 50)
            slow_ma = strategy_config.get('parameters', {}).get('slow_ma', 200)
            ma_type = strategy_config.get('parameters', {}).get('ma_type', 'SMA')
            volume_confirmation = strategy_config.get('parameters', {}).get('volume_confirmation', True)
            
            fast_name = f"{strategy_prefix}ma_fast"
            slow_name = f"{strategy_prefix}ma_slow"
            crossover_name = f"{strategy_prefix}crossover"
            
            self.add_indicator(fast_name, 'ma', {'period': fast_ma, 'type': ma_type.lower()})
            self.add_indicator(slow_name, 'ma', {'period': slow_ma, 'type': ma_type.lower()})
            self.add_indicator(crossover_name, 'ma_crossover', {
                'fast_period': fast_ma,
                'slow_period': slow_ma,
                'type': ma_type.lower()
            })
            
            configured_indicators = [fast_name, slow_name, crossover_name]
            
            if volume_confirmation:
                volume_name = f"{strategy_prefix}volume"
                self.add_indicator(volume_name, 'obv', {'ma_period': 20})
                configured_indicators.append(volume_name)
                
        elif strategy_name == 'keltner_channel_breakout':
            # Configure Keltner Channel Breakout indicators
            keltner_period = strategy_config.get('parameters', {}).get('keltner_period', 20)
            atr_period = strategy_config.get('parameters', {}).get('atr_period', 10)
            atr_multiplier = strategy_config.get('parameters', {}).get('atr_multiplier', 2.0)
            adx_period = strategy_config.get('parameters', {}).get('adx_period', 14)
            
            keltner_name = f"{strategy_prefix}keltner"
            adx_name = f"{strategy_prefix}adx"
            
            self.add_indicator(keltner_name, 'keltner', {
                'ema_period': keltner_period,
                'atr_period': atr_period,
                'atr_multiplier': atr_multiplier
            })
            self.add_indicator(adx_name, 'adx', {'period': adx_period})
            
            configured_indicators = [keltner_name, adx_name]
        
        logger.info(f"Configured indicators for {strategy_name} strategy")
        return configured_indicators 