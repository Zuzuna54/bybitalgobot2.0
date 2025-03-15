"""
Account Service for Bybit API

This module provides functionality for managing Bybit account operations,
including balance retrieval, position management, and account settings.
"""

from typing import Dict, Any, Optional, List
from loguru import logger

from src.api.bybit.core.connection import ConnectionManager
from src.api.bybit.core.error_handling import with_error_handling
from src.api.bybit.core.rate_limiting import rate_limited
from src.api.bybit.core.api_client import make_request


class AccountService:
    """
    Service for managing Bybit account operations.
    """
    
    def __init__(self, connection_manager: ConnectionManager):
        """
        Initialize the account service.
        
        Args:
            connection_manager: Connection manager instance
        """
        self.connection_manager = connection_manager
    
    @rate_limited('account')
    @with_error_handling
    def get_wallet_balance(
        self, 
        account_type: str = "UNIFIED", 
        coin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get wallet balance.
        
        Args:
            account_type: Account type ('UNIFIED', 'CONTRACT', etc.)
            coin: Currency (e.g., 'BTC', 'ETH')
            
        Returns:
            Wallet balance data
        """
        endpoint = "/v5/account/wallet-balance"
        params = {'accountType': account_type}
        
        if coin:
            params['coin'] = coin
            
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('account')
    @with_error_handling
    def get_positions(
        self, 
        category: str = "linear", 
        symbol: Optional[str] = None,
        settle_coin: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get positions.
        
        Args:
            category: Product type (linear, inverse, spot)
            symbol: Trading pair symbol
            settle_coin: Settlement currency
            
        Returns:
            Position data
        """
        endpoint = "/v5/position/list"
        params = {'category': category}
        
        if symbol:
            params['symbol'] = symbol
            
        if settle_coin:
            params['settleCoin'] = settle_coin
            
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('account')
    @with_error_handling
    def set_leverage(
        self, 
        symbol: str, 
        leverage: float, 
        leverage_mode: str = "isolated"
    ) -> Dict[str, Any]:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            leverage: Leverage multiplier
            leverage_mode: 'cross' or 'isolated'
            
        Returns:
            Response data
        """
        endpoint = "/v5/position/set-leverage"
        params = {
            'symbol': symbol,
            'buyLeverage': str(leverage),
            'sellLeverage': str(leverage)
        }
        
        response = make_request(
            connection_manager=self.connection_manager,
            method="POST",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('account')
    @with_error_handling
    def set_position_mode(
        self, 
        mode: str = "BothSide", 
        symbol: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Set position mode.
        
        Args:
            mode: Position mode ('BothSide' for hedge mode, 'MergedSingle' for one-way mode)
            symbol: Trading pair symbol (optional)
            
        Returns:
            Response data
        """
        endpoint = "/v5/position/switch-mode"
        params = {'mode': mode}
        
        if symbol:
            params['symbol'] = symbol
            
        response = make_request(
            connection_manager=self.connection_manager,
            method="POST",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('account')
    @with_error_handling
    def set_position_risk_limit(
        self, 
        symbol: str, 
        risk_id: int
    ) -> Dict[str, Any]:
        """
        Set position risk limit.
        
        Args:
            symbol: Trading pair symbol
            risk_id: Risk limit ID
            
        Returns:
            Response data
        """
        endpoint = "/v5/position/set-risk-limit"
        params = {
            'symbol': symbol,
            'riskId': risk_id
        }
        
        response = make_request(
            connection_manager=self.connection_manager,
            method="POST",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})
    
    @rate_limited('account')
    @with_error_handling
    def get_transaction_history(
        self,
        category: str = "linear",
        symbol: Optional[str] = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get transaction history.
        
        Args:
            category: Product type
            symbol: Trading pair symbol
            limit: Maximum number of records to return
            
        Returns:
            Transaction history data
        """
        endpoint = "/v5/account/transaction-log"
        params = {
            'category': category,
            'limit': limit
        }
        
        if symbol:
            params['symbol'] = symbol
            
        response = make_request(
            connection_manager=self.connection_manager,
            method="GET",
            endpoint=endpoint,
            params=params,
            auth_required=True
        )
        
        return response.get('result', {})

    @rate_limited('account')
    @with_error_handling
    def verify_credentials(self) -> bool:
        """
        Verify that the API credentials are valid.
        
        Returns:
            True if the credentials are valid, False otherwise
        """
        try:
            # Try to get wallet balance as a simple check
            self.get_wallet_balance(account_type="UNIFIED")
            return True
        except Exception as e:
            logger.warning(f"Failed to verify credentials: {e}")
            return False