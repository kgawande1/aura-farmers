from datamodel import Order, OrderDepth, TradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import numpy as np
import json
import jsonpickle  # type: ignore
import statistics

class Trader:
    def __init__(self):
        pass

    def get_fair_value_resin(self, price_cache, product) -> float:
        default_value: float = 0.0
        diff: int = 1
        max_ticks: int = 5
        
        if product not in price_cache or len(price_cache[product]) <= 2 * max_ticks:
            return price_cache[product][-1] if product in price_cache else default_value
        
        prices = np.array(price_cache[product])
        diffed_prices = np.diff(prices, n=diff)

        n = len(diffed_prices)
        X, y = [], []

        for t in range(max_ticks, n):
            X.append(diffed_prices[t-max_ticks:t][::-1])
            y.append(diffed_prices[t])

        X = np.array(X)
        y = np.array(y)
        phi, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        last_p = diffed_prices[-max_ticks:][::-1]
        forecast_diff = np.dot(phi, last_p)
        forecast_price = prices[-1] + forecast_diff

        return forecast_price

    def trade_resin(self, product, order_depth, state, price_cache) -> List[Order]:
        orders = []
        fair_price = self.get_fair_value_resin(price_cache, product)
        position = state.position.get(product, 0)

        # Buy if ask < fair
        for ask_price in sorted(order_depth.sell_orders):
            if ask_price < fair_price:
                volume = order_depth.sell_orders[ask_price]
                buy_qty = min(-volume, 50 - position)
                if buy_qty > 0:
                    orders.append(Order(product, ask_price, buy_qty))
                    position += buy_qty

        # Sell if bid > fair
        for bid_price in sorted(order_depth.buy_orders, reverse=True):
            if bid_price > fair_price:
                volume = order_depth.buy_orders[bid_price]
                sell_qty = min(volume, position + 50)
                if sell_qty > 0:
                    orders.append(Order(product, bid_price, -sell_qty))
                    position -= sell_qty

        return orders

    def trade_kelp(self, product, order_depth, state, price_cache, window_size=10) -> List[Order]:

        orders = []
        position = state.position.get(product, 0)

        if product not in price_cache or len(price_cache[product]) < window_size:
            return orders

        recent_prices = price_cache[product][-window_size:]
        avg_price = sum(recent_prices) / len(recent_prices)
        std_dev = statistics.stdev(recent_prices)

        upper_band = avg_price + std_dev
        lower_band = avg_price - std_dev

        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0
        else:
            return orders

        # Buy if under lower band
        if mid_price < lower_band:
            for ask_price in sorted(order_depth.sell_orders):
                if ask_price < avg_price:
                    volume = -order_depth.sell_orders[ask_price]
                    buy_qty = min(volume, 50 - position)
                    if buy_qty > 0:
                        orders.append(Order(product, ask_price, buy_qty))
                        position += buy_qty

        # Sell if over upper band
        elif mid_price > upper_band:
            for bid_price in sorted(order_depth.buy_orders, reverse=True):
                if bid_price > avg_price:
                    volume = order_depth.buy_orders[bid_price]
                    sell_qty = min(volume, position + 50)
                    if sell_qty > 0:
                        orders.append(Order(product, bid_price, -sell_qty))
                        position -= sell_qty

        return orders



    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            traderData = jsonpickle.decode(state.traderData)
        except Exception:
            traderData = {}

        price_cache: Dict[str, List[float]] = traderData.get("price_cache", {})

        for product in state.order_depths:
            order_depth = state.order_depths[product]

            # Update price cache with mid-price
            if product not in price_cache:
                price_cache[product] = []

            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2.0
                price_cache[product].append(mid_price)

            # Dispatch to product-specific strategy
            if product == "RAINFOREST_RESIN":
                result[product] = self.trade_resin(product, order_depth, state, price_cache)
            elif product == "KELP":
                result[product] = self.trade_kelp(product, order_depth, state, price_cache, window_size=5)

        traderData = jsonpickle.encode({"price_cache": price_cache})
        return result, conversions, traderData
