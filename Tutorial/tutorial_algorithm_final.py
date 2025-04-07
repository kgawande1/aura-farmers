# Our final tutorial algorithm


from datamodel import Order, OrderDepth, TradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import numpy as np
import json
import jsonpickle # type: ignore

class Trader:

    def get_fair_value(self, product, price_cache) -> float:

        default_value: float = 0.0
        diff: int = 1
        max_ticks: int = 5
        
        if product not in price_cache or len(price_cache[product]) <= 2 * max_ticks:
            if product in price_cache:
                return price_cache[product][-1]
            else:
                return default_value
            
        prices = np.array(price_cache[product])
        diffed_prices = np.diff(prices, n=diff)

        n = len(diffed_prices)
        X = []
        y = []

        for t in range(max_ticks, n):
            X.append(diffed_prices[t-max_ticks:t][::-1])
            y.append(diffed_prices[t])

        X = np.array(X)
        y = np.array(y)

        phi, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Forecast the next difference using the last p differences.
        last_p = diffed_prices[-max_ticks:][::-1]
        forecast_diff = np.dot(phi, last_p)
        # The forecasted price is the last observed price plus the forecasted difference.
        forecast_price = prices[-1] + forecast_diff
        
        return forecast_price

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        try:
            traderData = jsonpickle.decode(state.traderData)
        except Exception as _:
            traderData = {}

        price_cache: Dict[str: List[float]] = traderData["price_cache"] if "price_cache" in traderData else {}

        for product in state.order_depths:

            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            fair_price = self.get_fair_value(product, price_cache)

            if fair_price == 0:

                if product not in price_cache:
                    price_cache[product] = []

                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2.0

                price_cache[product].append(mid_price)

                result[product] = orders

                continue

            # Buy if ask < fair
            for ask_price in sorted(order_depth.sell_orders):
                if ask_price < fair_price:
                    volume = order_depth.sell_orders[ask_price]
                    buy_qty = min(-volume, 50 - state.position.get(product, 0))
                    if buy_qty > 0:
                        orders.append(Order(product, ask_price, buy_qty))

            # Sell if bid > fair
            for bid_price in sorted(order_depth.buy_orders, reverse=True):
                if bid_price > fair_price:
                    volume = order_depth.buy_orders[bid_price]
                    sell_qty = min(volume, state.position.get(product, 0) + 50)
                    if sell_qty > 0:
                        orders.append(Order(product, bid_price, -sell_qty))

            if product not in price_cache:
                price_cache[product] = []

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            price_cache[product].append(mid_price)

            result[product] = orders


        traderData = jsonpickle.encode({
            "price_cache": price_cache
        })
        
        return result, conversions, traderData