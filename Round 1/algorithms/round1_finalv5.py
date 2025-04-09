from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import numpy as np
import types
import jsonpickle
from collections import deque

# First Submission
class Trader:

    def __init__(self):
        
        self.limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }




    def market_making_strategy(self, product, price_cache, position, order_depth, fair_value, traderData, rho=1, k=200):
            
        orders = []

        if product not in price_cache:
            price_cache[product] = deque(maxlen=k)

        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders, price_cache

        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else 0
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else 0



        mid_price = (best_bid + best_ask) / 2.0

        price_cache[product].append(mid_price)

        
        proportion = 0
        for i in range(1, k):
            if price_cache[product][i-1] < price_cache[product][i]:
                proportion += 1

        Su = proportion / k

        Su_transform = 1 if Su > 0.7 else (.8 - Su)


        best_bid_amount = order_depth.buy_orders[best_bid]
        best_ask_amount = order_depth.sell_orders[best_ask]

        buy_price = best_bid + 1
        sell_price = best_ask - 1
        buy_qty = min(self.limits[product] - position, best_bid_amount * Su_transform)
        sell_qty = min(self.limits[product] + position, best_ask_amount * Su_transform)


        if mid_price < fair_value - rho:
            orders.append(Order(product, buy_price, buy_qty))
        elif (mid_price > fair_value + rho):
            orders.append(Order(product, sell_price, -sell_qty))

        traderData = jsonpickle.encode({
            "price_cache": price_cache
        })


        return orders, traderData


        


    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        k = 200 # window size for merton estimation

        try:
            traderData = jsonpickle.decode(state.traderData)
        except Exception as _:
            traderData = {}

        if traderData is None:
            traderData = {}

        price_cache = traderData["price_cache"] if "price_cache" in traderData else {}


        ink_value = self.get_fair_value_merton(
            T=100,
            mu=0.00032542,
            lamb=0.000223,
            sigma=0.0003324,
            v=0,
            delta=0.00006245,
            prev_prices= list(price_cache["SQUID_INK"]) if "SQUID_INK" in price_cache else [2000],   # FIX: Pass a float instead of the entire price_cache
            mu_w=0.5,
            sigma_w=0.5,
        )
        # Initial simple fair values for demo purposes
        fair_prices = {
            "RAINFOREST_RESIN": 10000,
            "KELP": 10000,
            "SQUID_INK": ink_value
        }

        for product in state.order_depths:
            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            fair_price = fair_prices[product]

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

            result[product] = orders


            if product not in price_cache:
                price_cache[product] = deque()

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            if len(price_cache[product]) < k:
                price_cache[product].append(mid_price)
            else:
                price_cache[product].popleft()
                price_cache[product].append(mid_price)

            result[product] = orders


        traderData = jsonpickle.encode({
            "price_cache": price_cache
        })

        return result, conversions, traderData
