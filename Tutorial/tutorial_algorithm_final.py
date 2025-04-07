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
        max_ticks: int = 10
        
        if product not in price_cache or len(price_cache[product]) < 10:
            if product in price_cache:
                return price_cache[product][-1]
            else:
                return default_value # fix later
            
        prices = np.array(price_cache[product])
        diffed_prices = np.diff(prices, n=diff)

        n = len(prices)
        X = [], y = []

        for t in range(max_ticks, n):
            X.append(diffed_prices[t-max_ticks:t][::-1])
            y.append(diffed_prices[t])

        X = np.array(X)
        y = np.array(y)

        phi, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

        # Forecast the next difference using the last p differences.
        last_p = diffed_prices[-p:][::-1]
        forecast_diff = np.dot(phi, last_p)
        # The forecasted price is the last observed price plus the forecasted difference.
        forecast_price = prices[-1] + forecast_diff
        
        return forecast_price

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = jsonpickle.decode(state.traderData)

        price_cache = traderData["price_cache"]

        for product in state.order_depths:

            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            fair_price = self.get_fair_value(product, price_cache)

            new_price = 0

            # Buy if ask < fair
            for ask_price in sorted(order_depth.sell_orders):
                if ask_price < fair_price:
                    volume = order_depth.sell_orders[ask_price]
                    buy_qty = min(-volume, 50 - state.position.get(product, 0))
                    if buy_qty > 0:
                        orders.append(Order(product, ask_price, buy_qty))

                new_price += ask_price

            # Sell if bid > fair
            for bid_price in sorted(order_depth.buy_orders, reverse=True):
                if bid_price > fair_price:
                    volume = order_depth.buy_orders[bid_price]
                    sell_qty = min(volume, state.position.get(product, 0) + 50)
                    if sell_qty > 0:
                        orders.append(Order(product, bid_price, -sell_qty))

            price_cache[product].add(new_price / len(order_depth.sell_orders))

            result[product] = orders


        traderData = jsonpickle.encode({
            "price_cache": price_cache
        })
        
        return result, conversions, traderData