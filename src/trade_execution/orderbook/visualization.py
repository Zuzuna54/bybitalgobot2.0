"""
Order Book Visualization

This module provides functions for visualizing order book data and analysis,
including heatmaps, depth charts, and visual indicators for signals.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from loguru import logger


def visualize_orderbook(orderbook: Dict[str, Any], levels: int = 20) -> plt.Figure:
    """
    Visualize the order book data.
    
    This is a wrapper function for plot_order_book_depth that provides
    a consistent naming convention throughout the codebase.
    
    Args:
        orderbook: Order book data from the exchange
        levels: Number of price levels to display
        
    Returns:
        Matplotlib figure object
    """
    return plot_order_book_depth(orderbook, levels, "Order Book Visualization")


def plot_order_book_depth(orderbook: Dict[str, Any], 
                         levels: int = 20, 
                         title: str = "Order Book Depth") -> plt.Figure:
    """
    Generate an order book depth chart showing bid and ask levels.
    
    Args:
        orderbook: Order book data from the exchange
        levels: Number of price levels to display
        title: Chart title
        
    Returns:
        Matplotlib figure object
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        logger.warning("No order book data available for visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No order book data available", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Convert to DataFrame
    bids_df = pd.DataFrame(bids, columns=["price", "size"])
    asks_df = pd.DataFrame(asks, columns=["price", "size"])
    
    bids_df["price"] = bids_df["price"].astype(float)
    bids_df["size"] = bids_df["size"].astype(float)
    asks_df["price"] = asks_df["price"].astype(float)
    asks_df["size"] = asks_df["size"].astype(float)
    
    # Limit to specified number of levels
    bids_df = bids_df.head(levels)
    asks_df = asks_df.head(levels)
    
    # Calculate cumulative sizes
    bids_df["cumulative_size"] = bids_df["size"].cumsum()
    asks_df["cumulative_size"] = asks_df["size"].cumsum()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bids (green)
    ax.fill_between(bids_df["price"], 0, bids_df["cumulative_size"], 
                   alpha=0.5, color="green", label="Bids")
    
    # Plot asks (red)
    ax.fill_between(asks_df["price"], 0, asks_df["cumulative_size"], 
                   alpha=0.5, color="red", label="Asks")
    
    # Add labels and title
    ax.set_xlabel("Price")
    ax.set_ylabel("Cumulative Size")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Calculate mid price
    mid_price = (float(bids[0][0]) + float(asks[0][0])) / 2
    
    # Add mid price line
    ax.axvline(x=mid_price, color="black", linestyle="--", 
              alpha=0.7, label=f"Mid Price: {mid_price:.2f}")
    
    # Update legend
    ax.legend()
    
    return fig


def create_order_book_heatmap(orderbook: Dict[str, Any], 
                             levels: int = 30) -> plt.Figure:
    """
    Create a heatmap visualization of the order book.
    
    Args:
        orderbook: Order book data from the exchange
        levels: Number of price levels to display
        
    Returns:
        Matplotlib figure object
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        logger.warning("No order book data available for heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No order book data available", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Convert to DataFrame
    bids_df = pd.DataFrame(bids, columns=["price", "size"])
    asks_df = pd.DataFrame(asks, columns=["price", "size"])
    
    bids_df["price"] = bids_df["price"].astype(float)
    bids_df["size"] = bids_df["size"].astype(float)
    asks_df["price"] = asks_df["price"].astype(float)
    asks_df["size"] = asks_df["size"].astype(float)
    
    # Limit to specified number of levels
    bids_df = bids_df.head(levels)
    asks_df = asks_df.head(levels)
    
    # Combine into one DataFrame for plotting
    bids_df["side"] = "bid"
    asks_df["side"] = "ask"
    combined_df = pd.concat([bids_df, asks_df])
    
    # Create normalized size for color intensity
    max_size = combined_df["size"].max()
    combined_df["normalized_size"] = combined_df["size"] / max_size
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create colormap
    norm = Normalize(vmin=0, vmax=1)
    
    # Plot bids (green) and asks (red)
    for idx, row in combined_df.iterrows():
        color = "green" if row["side"] == "bid" else "red"
        intensity = row["normalized_size"]
        y_pos = levels - bids_df.index.get_loc(idx) if row["side"] == "bid" else levels + asks_df.index.get_loc(idx)
        
        ax.barh(y_pos, row["size"], color=cm.get_cmap("Greens")(intensity) if row["side"] == "bid" 
               else cm.get_cmap("Reds")(intensity), alpha=0.7)
        
        # Add price labels
        ax.text(row["size"] * 1.02, y_pos, f"{row['price']:.2f}", 
               va="center", ha="left", fontsize=8)
    
    # Add labels and title
    ax.set_xlabel("Size")
    ax.set_title("Order Book Heatmap")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.3)
    
    # Add a divider line between bids and asks
    ax.axhline(y=levels, color="black", linestyle="-", alpha=0.3)
    
    # Add legend
    bid_patch = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.5)
    ask_patch = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=0.5)
    ax.legend([bid_patch, ask_patch], ["Bids", "Asks"], loc="upper right")
    
    return fig


def visualize_imbalance(orderbook: Dict[str, Any], depth_levels: int = 10) -> plt.Figure:
    """
    Visualize order book imbalance.
    
    Args:
        orderbook: Order book data from the exchange
        depth_levels: Number of price levels to consider
        
    Returns:
        Matplotlib figure object
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        logger.warning("No order book data available for imbalance visualization")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No order book data available", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Limit to specified depth
    bids = bids[:depth_levels] if depth_levels < len(bids) else bids
    asks = asks[:depth_levels] if depth_levels < len(asks) else asks
    
    # Calculate total volume on each side
    bid_volume = sum(float(bid[1]) for bid in bids)
    ask_volume = sum(float(ask[1]) for ask in asks)
    
    # Calculate imbalance
    total_volume = bid_volume + ask_volume
    
    if total_volume == 0:
        imbalance = 0
    else:
        imbalance = (bid_volume - ask_volume) / total_volume
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create a horizontal bar for the imbalance
    colors = ["red", "green"]
    labels = ["Ask Volume", "Bid Volume"]
    sizes = [ask_volume / total_volume * 100, bid_volume / total_volume * 100]
    
    ax.barh(0, sizes, left=[0, sizes[0]], color=colors, alpha=0.7, height=0.5)
    
    # Add labels
    for i, size in enumerate(sizes):
        if size > 5:  # Only add label if segment is large enough
            ax.text(sizes[0] if i == 1 else 0 + size / 2, 0, 
                   f"{labels[i]}\n{size:.1f}%", ha="center", va="center", color="white")
    
    # Add imbalance score
    ax.text(0.5, -0.5, f"Imbalance Score: {imbalance:.3f}", 
           ha="center", va="center", transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.7))
    
    # Format plot
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.set_title("Order Book Imbalance")
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.axvline(x=50, color="black", linestyle="--", alpha=0.5)
    
    return fig


def plot_price_levels(orderbook: Dict[str, Any], 
                     levels_dict: Dict[str, List[float]], 
                     title: str = "Significant Price Levels") -> plt.Figure:
    """
    Visualize significant price levels on the order book.
    
    Args:
        orderbook: Order book data from the exchange
        levels_dict: Dictionary with support and resistance levels
        title: Chart title
        
    Returns:
        Matplotlib figure object
    """
    # Extract support and resistance levels
    support_levels = levels_dict.get("support_levels", [])
    resistance_levels = levels_dict.get("resistance_levels", [])
    
    # Create order book depth chart as base
    fig = plot_order_book_depth(orderbook, title=title)
    ax = fig.axes[0]
    
    # Add support levels
    for level in support_levels:
        ax.axvline(x=level, color="blue", linestyle="-.", alpha=0.7)
        ax.text(level, ax.get_ylim()[1] * 0.9, f"S: {level:.2f}", 
               rotation=90, ha="right", va="top", color="blue")
    
    # Add resistance levels
    for level in resistance_levels:
        ax.axvline(x=level, color="purple", linestyle="-.", alpha=0.7)
        ax.text(level, ax.get_ylim()[1] * 0.9, f"R: {level:.2f}", 
               rotation=90, ha="right", va="top", color="purple")
    
    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    support_patch = plt.Line2D([0], [0], color="blue", linestyle="-.", alpha=0.7)
    resistance_patch = plt.Line2D([0], [0], color="purple", linestyle="-.", alpha=0.7)
    handles.extend([support_patch, resistance_patch])
    labels.extend(["Support", "Resistance"])
    ax.legend(handles, labels)
    
    return fig


def save_visualization(fig: plt.Figure, filename: str) -> str:
    """
    Save a visualization to a file.
    
    Args:
        fig: Matplotlib figure to save
        filename: Filename to save to
        
    Returns:
        Path to the saved file
    """
    try:
        fig.savefig(filename, bbox_inches='tight', dpi=300)
        logger.info(f"Visualization saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Failed to save visualization: {e}")
        return ""


def plot_liquidity_profile(orderbook: Dict[str, Any], price_range_pct: float = 2.0) -> plt.Figure:
    """
    Plot a liquidity profile showing the distribution of liquidity in the order book.
    
    Args:
        orderbook: Order book data from the exchange
        price_range_pct: Price range to display as percentage from mid price
        
    Returns:
        Matplotlib figure object
    """
    # Extract bids and asks
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    
    if not bids or not asks:
        logger.warning("No order book data available for liquidity profile")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No order book data available", 
                horizontalalignment='center', verticalalignment='center')
        return fig
    
    # Convert to DataFrame
    bids_df = pd.DataFrame(bids, columns=["price", "size"])
    asks_df = pd.DataFrame(asks, columns=["price", "size"])
    
    bids_df["price"] = bids_df["price"].astype(float)
    bids_df["size"] = bids_df["size"].astype(float)
    asks_df["price"] = asks_df["price"].astype(float)
    asks_df["size"] = asks_df["size"].astype(float)
    
    # Calculate mid price
    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid_price = (best_bid + best_ask) / 2
    
    # Calculate price range
    price_range = mid_price * price_range_pct / 100
    min_price = mid_price - price_range
    max_price = mid_price + price_range
    
    # Filter orders within price range
    bids_in_range = bids_df[bids_df["price"] >= min_price]
    asks_in_range = asks_df[asks_df["price"] <= max_price]
    
    # Group by price ranges for better visualization
    num_bins = 20
    bid_bins = np.linspace(min_price, mid_price, num_bins // 2 + 1)
    ask_bins = np.linspace(mid_price, max_price, num_bins // 2 + 1)
    
    # Group bids by price bins
    bids_in_range["bin"] = pd.cut(bids_in_range["price"], bins=bid_bins)
    bid_liquidity = bids_in_range.groupby("bin")["size"].sum().reset_index()
    bid_liquidity["mid_price"] = bid_liquidity["bin"].apply(lambda x: (x.left + x.right) / 2)
    
    # Group asks by price bins
    asks_in_range["bin"] = pd.cut(asks_in_range["price"], bins=ask_bins)
    ask_liquidity = asks_in_range.groupby("bin")["size"].sum().reset_index()
    ask_liquidity["mid_price"] = ask_liquidity["bin"].apply(lambda x: (x.left + x.right) / 2)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot bid liquidity
    ax.bar(bid_liquidity["mid_price"], bid_liquidity["size"], 
          width=(bid_bins[1] - bid_bins[0]) * 0.9, 
          color="green", alpha=0.6, label="Bid Liquidity")
    
    # Plot ask liquidity
    ax.bar(ask_liquidity["mid_price"], ask_liquidity["size"], 
          width=(ask_bins[1] - ask_bins[0]) * 0.9, 
          color="red", alpha=0.6, label="Ask Liquidity")
    
    # Add vertical line at mid price
    ax.axvline(x=mid_price, color="black", linestyle="--", 
              alpha=0.7, label=f"Mid Price: {mid_price:.2f}")
    
    # Add labels and title
    ax.set_xlabel("Price")
    ax.set_ylabel("Liquidity (Size)")
    ax.set_title("Order Book Liquidity Profile")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show prices
    plt.xticks(rotation=45)
    
    return fig 