from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import numpy as np
import types
import jsonpickle
from collections import deque

# First Submission
class Trader:
    def __init__(self):
        self.limit = 50

    def exponential_moving_average(self, prices, alpha=0.3):
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def get_trend(self, prices):
        if len(prices) < 2:
            return 0
        x = np.arange(len(prices))
        y = np.array(prices)

        slope, _ = np.polyfit(x, y, 1)
        return slope

    def get_fair_value_merton(
            self,
            T: float,
            mu: float, 
            lamb: float, 
            sigma: float, 
            v: float,
            delta: float, 
            prev_prices: List[float],
            n: int = 5,
            m: int = 100,
            beta: float = 0.47):

        kappa = np.exp(v + 0.5 * delta**2) - 1
        dt = T / m
        total_price = 0

        ema = self.exponential_moving_average(prices=prev_prices)
        trend = self.get_trend(prices=prev_prices)
        S_0 = ema + trend * 3

        for _ in range(n):
            S_t = S_0
            for _ in range(m):
                W = np.random.normal(0, np.sqrt(dt))
                N = np.random.poisson(lamb * dt)
                jump_sum = np.sum(np.random.normal(v, delta, size=N)) if N > 0 else 0.0

                drift = (mu - lamb * kappa - 0.5 * sigma**2) * dt
                diffusion = sigma * W
                S_t = np.exp(np.log(S_t) + beta * (drift + diffusion + jump_sum))

            total_price += S_t

        return total_price / n

    def get_kelp_fair_value(self, product, order_depths):

        buy_orders = sorted(order_depths.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depths.sell_orders.items())

        popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
        popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]

        return round((popular_buy_price + popular_sell_price) / 2)


    def market_making_strategy(self, product, orders, orders_depth, price_cache, position, window_size=10):

        window = deque(maxlen=10)

        if product == "KELP":
            fair_value = self.get_kelp_fair_value(product, orders_depth)
        elif product == "SQUID_INK":
            fair_value = self.get_fair_value_merton(
                T=100,
                mu=0.0003,
                lamb=0.0107,
                sigma=5.0000,
                v=-0.0001,
                delta=0.0028,
                prev_prices= list(price_cache["SQUID_INK"]) if "SQUID_INK" in price_cache else [2000.0],   # FIX: Pass a float instead of the entire price_cache
                )

        else:
            fair_value = 10000


        buy_orders = sorted(orders_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(orders_depth.sell_orders.items())

        buy_amount = self.limit - position
        sell_amount = self.limit + position

        closeToLimit = (abs(position) == self.limit)

        window.append(closeToLimit)

        if len(window) > window_size:
            window.popleft()

        hard = len(window) == window_size and all(window)
        soft = len(window) == window_size and sum(window) >= window_size / 2 and window[-1]

        if position > self.limit * 0.5:
            max_buy = fair_value - 1
        else:
            max_buy = fair_value

        if position > self.limit * -0.5:
            min_sell = fair_value + 1
        else:
            min_sell = fair_value

        max_buy = int(max_buy)
        min_sell = int(min_sell)

        # BUY
        for price, volume in sell_orders:
            if buy_amount > 0 and price <= max_buy:
                quantity = min(buy_amount, -volume)

                orders.append(Order(product, price, quantity))
                buy_amount -= quantity


        if buy_amount > 0 and hard:
            quantity = buy_amount // 2

            orders.append(Order(product, int(fair_value) - 2, quantity))
            buy_amount -= quantity

        if buy_amount > 0 and soft:
            quantity = buy_amount // 2

            orders.append(Order(product, price, quantity))
            buy_amount -= quantity

        if buy_amount > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy, popular_buy_price + 1)
            orders.append(Order(product, price, buy_amount))


        # SELL

        for price, volume in buy_orders:
            if sell_amount > 0 and price >= min_sell:
                quantity = min(sell_amount, volume)
                orders.append(Order(product, price, -quantity))
                sell_amount -= quantity

        if sell_amount > 0 and hard:
            quantity = sell_amount // 2

            orders.append(Order(product, fair_value + 2, -quantity))
            sell_amount -= quantity

        if sell_amount > 0 and soft:
            quantity = sell_amount // 2
            orders.append(Order(product, price, -quantity))
            sell_amount -= quantity

        if sell_amount > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell, popular_sell_price - 1)
            orders.append(Order(product, price, -sell_amount))


        return orders

        
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


        for product in state.order_depths:

            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            orders = self.market_making_strategy(product, orders, order_depth, price_cache, position, 10)

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
