import logging
import os
from dotenv import load_dotenv
from src.action_handler import register_action

logger = logging.getLogger("actions.sonic_actions")

@register_action("get-token-by-ticker")
def get_token_by_ticker(agent, **kwargs):
    """Get token address by ticker symbol"""
    try:
        ticker = kwargs.get("ticker")
        if not ticker:
            logger.error("No ticker provided")
            return None
            
        token_address = agent.connection_manager.connections["sonic"].get_token_by_ticker(ticker)
        
        if token_address:
            logger.info(f"Found token address for {ticker}: {token_address}")
        else:
            logger.info(f"No token found for ticker {ticker}")
            
        return token_address

    except Exception as e:
        logger.error(f"Failed to get token by ticker: {str(e)}")
        return None

@register_action("get-sonic-balance")
def get_sonic_balance(agent, **kwargs):
    """Get $S or token balance"""
    try:
        address = kwargs.get("address")
        token_address = kwargs.get("token_address")
        
        balance = agent.connection_manager.connections["sonic"].get_balance(
            address=address,
            token_address=token_address
        )
        
        if token_address:
            logger.info(f"Token Balance: {balance}")
        else:
            logger.info(f"$S Balance: {balance}")
            
        return balance

    except Exception as e:
        logger.error(f"Failed to get balance: {str(e)}")
        return None

@register_action("send-sonic")
def send_sonic(agent, **kwargs):
    """Send $S tokens to an address"""
    try:
        from_address = kwargs.get("from_address")
        to_address = kwargs.get("to_address")
        amount = float(kwargs.get("amount"))

        tx = agent.connection_manager.connections["sonic"].transfer(
            from_address=from_address,
            to_address=to_address,
            amount=amount
        )

        logger.info(f"Transferred {amount} $S to {to_address}")
        logger.info(f"Transaction Data: {tx}")
        return tx

    except Exception as e:
        logger.error(f"Failed to send $S: {str(e)}")
        return None

@register_action("send-sonic-token")
def send_sonic_token(agent, **kwargs):
    """Send tokens on Sonic chain"""
    try:
        from_address = kwargs.get("from_address")
        to_address = kwargs.get("to_address")
        token_address = kwargs.get("token_address")
        amount = float(kwargs.get("amount"))

        tx = agent.connection_manager.connections["sonic"].transfer(
            from_address=from_address,
            to_address=to_address,
            amount=amount,
            token_address=token_address
        )

        logger.info(f"Transferred {amount} tokens to {to_address}")
        logger.info(f"Transaction Data: {tx}")
        return tx

    except Exception as e:
        logger.error(f"Failed to send tokens: {str(e)}")
        return None

@register_action("get-swap-summary")
def get_swap_summary(agent, **kwargs):
    """Get summary of swap on Sonic chain"""
    try:
        
        token_in = kwargs.get("token_in")
        token_out = kwargs.get("token_out") 
        amount = float(kwargs.get("amount"))

        route_data = agent.connection_manager.connections["sonic"]._get_swap_route(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount
        )

        return {**route_data["routeSummary"], "routerAddress": route_data["routerAddress"]}

    except Exception as e:
        logger.error(f"Failed to get summary: {str(e)}")
        return None


@register_action("swap-sonic")
def swap_sonic(agent, **kwargs):
    """Swap tokens on Sonic chain"""
    try:
        sender  = kwargs.get("sender")
        token_in = kwargs.get("token_in")
        token_out = kwargs.get("token_out") 
        amount = float(kwargs.get("amount"))
        slippage = float(kwargs.get("slippage", 7.0))

        tx_url = agent.connection_manager.connections["sonic"].swap(
            sender=sender,
            token_in=token_in,
            token_out=token_out,
            amount=amount,
            slippage=slippage
        )

        logger.info(f"Swapping {amount} tokens")
        return tx_url

    except Exception as e:
        logger.error(f"Failed to swap tokens: {str(e)}")
        return None