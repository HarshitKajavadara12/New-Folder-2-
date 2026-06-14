"""
Smart Order Router (SOR)
Routes orders to optimal venues based on:
- Liquidity
- Latency
- Fees
- Historical fill quality

Designed for crypto exchanges accessible via ccxt.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Venue(Enum):
    """Supported crypto trading venues."""
    BINANCE = "BINANCE"
    COINBASE = "COINBASE"
    KRAKEN = "KRAKEN"
    OKX = "OKX"
    BYBIT = "BYBIT"
    KUCOIN = "KUCOIN"

class SlippageModel:
    """
    Phase 11: Realistic Slippage calculation.
    Slippage = f(Volatility, Liquidity, Size)
    """
    @staticmethod
    def calculate_effective_price(
        mid_price: float, 
        side: str, 
        quantity: float, 
        volatility: float, 
        spread: float,
        daily_volume: float = 1_000_000
    ) -> float:
        """
        Calculate effective execution price including impact.
        
        Formula:
        Price = Mid +/- (Spread/2 + Impact)
        Impact = Volatility * sqrt(Size / Volume)
        """
        # 1. Base Spread Cost
        half_spread = spread / 2
        
        # 2. Market Impact (Square Root Law)
        # Participation rate
        participation = quantity / (daily_volume * 0.1) # Assuming 10% of daily volume available in this window
        impact_pct = volatility * np.sqrt(participation)
        impact_cost = mid_price * impact_pct
        
        # 3. Total Slippage
        total_slippage = half_spread + impact_cost
        
        if side.upper() == "BUY":
            return mid_price + total_slippage
        else:
            return mid_price - total_slippage

@dataclass
class VenueMetrics:
    """Venue performance metrics for crypto exchanges."""
    venue: Venue
    avg_latency_ms: float  # Average REST/WS fill latency (milliseconds)
    liquidity_score: float  # 0-1, higher is better
    maker_fee_bps: float  # Maker fee in basis points
    taker_fee_bps: float  # Taker fee in basis points
    fill_rate: float  # 0-1, percentage of orders filled
    avg_slippage_bps: float  # Average slippage in basis points
    
    def calculate_score(self, order_size_usd: float, urgency: float = 0.5) -> float:
        """
        Calculate venue score for order routing.
        
        Args:
            order_size_usd: Notional order value in USD
            urgency: 0-1, higher means prioritize speed over cost
        
        Returns:
            score (higher is better)
        """
        # Normalize metrics
        latency_score = max(0, 1 - self.avg_latency_ms / 500)  # 500ms worst case
        # Use taker fee for urgent, maker fee for passive
        fee_bps = self.taker_fee_bps * urgency + self.maker_fee_bps * (1 - urgency)
        cost_score = max(0, 1 - fee_bps / 15)  # 15 bps max
        slippage_score = max(0, 1 - self.avg_slippage_bps / 10)
        
        # Large orders penalise low-liquidity venues
        size_penalty = 1.0 if order_size_usd < 50_000 else max(0.5, self.liquidity_score)
        
        score = (
            urgency * latency_score * 0.35 +
            (1 - urgency) * cost_score * 0.30 +
            self.liquidity_score * size_penalty * 0.20 +
            self.fill_rate * 0.10 +
            slippage_score * 0.05
        )
        
        return score


class SmartOrderRouter:
    """
    Intelligent order routing system.
    
    Selects optimal venue based on:
    - Order characteristics (size, urgency)
    - Venue metrics (latency, fees, liquidity)
    - Historical performance
    """
    
    def __init__(self):
        """Initialize router with real crypto exchange metrics."""
        self.venue_metrics = {
            Venue.BINANCE: VenueMetrics(
                venue=Venue.BINANCE,
                avg_latency_ms=50,
                liquidity_score=0.98,
                maker_fee_bps=1.0,
                taker_fee_bps=1.0,
                fill_rate=0.99,
                avg_slippage_bps=0.5
            ),
            Venue.COINBASE: VenueMetrics(
                venue=Venue.COINBASE,
                avg_latency_ms=80,
                liquidity_score=0.90,
                maker_fee_bps=4.0,
                taker_fee_bps=6.0,
                fill_rate=0.97,
                avg_slippage_bps=1.0
            ),
            Venue.KRAKEN: VenueMetrics(
                venue=Venue.KRAKEN,
                avg_latency_ms=100,
                liquidity_score=0.85,
                maker_fee_bps=1.6,
                taker_fee_bps=2.6,
                fill_rate=0.96,
                avg_slippage_bps=1.2
            ),
            Venue.OKX: VenueMetrics(
                venue=Venue.OKX,
                avg_latency_ms=60,
                liquidity_score=0.93,
                maker_fee_bps=0.8,
                taker_fee_bps=1.0,
                fill_rate=0.98,
                avg_slippage_bps=0.6
            ),
            Venue.BYBIT: VenueMetrics(
                venue=Venue.BYBIT,
                avg_latency_ms=55,
                liquidity_score=0.91,
                maker_fee_bps=1.0,
                taker_fee_bps=1.0,
                fill_rate=0.97,
                avg_slippage_bps=0.7
            ),
            Venue.KUCOIN: VenueMetrics(
                venue=Venue.KUCOIN,
                avg_latency_ms=90,
                liquidity_score=0.80,
                maker_fee_bps=1.0,
                taker_fee_bps=1.0,
                fill_rate=0.94,
                avg_slippage_bps=1.5
            ),
        }
        
        # Routing statistics
        self.routing_history: List[Tuple[str, Venue, float]] = []
    
    def route_order(
        self,
        symbol: str,
        quantity: float,
        urgency: float = 0.5,
        max_venues: int = 3,
        price: float = 0.0,
    ) -> List[Tuple[Venue, float, float]]:
        """
        Route order to optimal crypto venues.
        
        Args:
            symbol: Trading pair (e.g. "BTCUSDT")
            quantity: Amount of base asset to trade
            urgency: 0-1, trading urgency
            max_venues: Maximum venues to split order across
            price: Current price (used for notional calculation)
        
        Returns:
            List of (venue, quantity, expected_fee_usd) tuples
        """
        notional_usd = quantity * price if price > 0 else quantity * 50_000
        
        # Calculate scores for each venue
        venue_scores = []
        for venue, metrics in self.venue_metrics.items():
            score = metrics.calculate_score(notional_usd, urgency)
            venue_scores.append((venue, score, metrics))
        
        # Sort by score (descending)
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate quantity to top venues
        allocations = []
        remaining = quantity
        
        for i, (venue, score, metrics) in enumerate(venue_scores[:max_venues]):
            if i == max_venues - 1:
                alloc_qty = remaining
            else:
                total_score = sum(s for _, s, _ in venue_scores[:max_venues])
                alloc_qty = quantity * (score / total_score) if total_score > 0 else remaining
                alloc_qty = min(alloc_qty, remaining)
            
            if alloc_qty > 0:
                fee_bps = metrics.taker_fee_bps if urgency > 0.5 else metrics.maker_fee_bps
                expected_fee = alloc_qty * price * fee_bps / 10_000 if price > 0 else 0
                allocations.append((venue, alloc_qty, expected_fee))
                remaining -= alloc_qty
                
                self.routing_history.append((symbol, venue, alloc_qty))
        
        return allocations
    
    def get_best_venue(self, symbol: str, quantity: float, urgency: float = 0.5) -> Venue:
        """Get single best venue for order."""
        allocations = self.route_order(symbol, quantity, urgency, max_venues=1)
        return allocations[0][0] if allocations else Venue.BINANCE
    
    def update_venue_metrics(self, venue: Venue, **kwargs):
        """Update venue metrics based on execution results."""
        if venue in self.venue_metrics:
            metrics = self.venue_metrics[venue]
            for key, value in kwargs.items():
                if hasattr(metrics, key):
                    # Exponential moving average
                    current = getattr(metrics, key)
                    updated = 0.9 * current + 0.1 * value
                    setattr(metrics, key, updated)
    
    def get_routing_stats(self) -> Dict:
        """Get routing statistics."""
        if not self.routing_history:
            return {}
        
        venue_counts = {}
        venue_volume = {}
        
        for symbol, venue, qty in self.routing_history:
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
            venue_volume[venue] = venue_volume.get(venue, 0) + qty
        
        total_orders = len(self.routing_history)
        total_volume = sum(venue_volume.values())
        
        stats = {
            'total_orders_routed': total_orders,
            'total_volume': total_volume,
            'venue_distribution': {
                v.value: {
                    'orders': venue_counts.get(v, 0),
                    'volume': venue_volume.get(v, 0),
                    'order_pct': venue_counts.get(v, 0) / total_orders * 100,
                    'volume_pct': venue_volume.get(v, 0) / total_volume * 100
                }
                for v in Venue
            }
        }
        
        return stats


class SlicingEngine:
    """
    PHASE 10: Order Slicing
    Different logic for different regimes.
    """
    @staticmethod
    def slice_order(total_qty: float, regime: str) -> List[float]:
        if total_qty < 0.01:
            return [total_qty]
            
        # If SEED or GROWTH, just execute (simulated for simplicity here)
        if "SEED" in regime or "GROWTH" in regime:
            return [total_qty]
            
        # If SCALE +, slice into chunks (TWAP style implied)
        # Returns list of quantities to execute
        chunks = []
        remaining = total_qty
        
        while remaining > 0:
            # Max chunk size logic (e.g. 0.5 BTC max)
            chunk = min(0.5, remaining)
            chunks.append(chunk)
            remaining -= chunk
            
        return chunks

# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("SMART ORDER ROUTER DEMO — CRYPTO VENUES")
    print("=" * 70)
    
    router = SmartOrderRouter()
    
    # Test different order scenarios
    scenarios = [
        ("BTCUSDT", 0.5, 65000, 0.9, "High urgency BTC order"),
        ("BTCUSDT", 10.0, 65000, 0.3, "Large BTC order, low urgency (TWAP)"),
        ("ETHUSDT", 5.0, 3500, 0.7, "Medium urgency ETH order"),
        ("SOLUSDT", 100.0, 150, 1.0, "Ultra urgent SOL order"),
    ]
    
    for symbol, qty, price, urgency, description in scenarios:
        print(f"\n{description}")
        print(f"Symbol: {symbol}, Quantity: {qty}, Price: ${price:,.0f}, Urgency: {urgency:.1f}")
        print("-" * 70)
        
        allocations = router.route_order(symbol, qty, urgency, max_venues=3, price=price)
        
        total_fee = 0
        for venue, alloc_qty, fee in allocations:
            pct = alloc_qty / qty * 100
            print(f"  {venue.value:15s}: {alloc_qty:10.4f} ({pct:5.1f}%) - Fee: ${fee:.4f}")
            total_fee += fee
        
        print(f"  {'Total Fee':15s}: ${total_fee:.4f}")
    
    # Display routing statistics
    print("\n" + "=" * 70)
    print("ROUTING STATISTICS")
    print("=" * 70)
    
    stats = router.get_routing_stats()
    print(f"\nTotal Orders Routed: {stats['total_orders_routed']}")
    print(f"Total Volume: {stats['total_volume']:.4f}")
    
    print("\nVenue Distribution:")
    for venue_name, data in stats['venue_distribution'].items():
        if data['orders'] > 0:
            print(f"  {venue_name:15s}: {data['orders']:3d} orders ({data['order_pct']:5.1f}%), "
                  f"{data['volume']:.4f} ({data['volume_pct']:5.1f}%)")
