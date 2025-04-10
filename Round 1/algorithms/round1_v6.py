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
            mu_w: float,
            sigma_w: float,
            n: int = 100,
            beta: float = 0.47):
        
        # n -> number of simulations
        # m -> number of future walk

        kappa = np.exp(v + 0.5 * pow(delta, 2)) - 1
        avg = 0

        # get average price
        # weights = np.random.normal(loc=mu_w, scale=sigma_w, size=len(prev_prices))
        # weights = np.ones(shape=len(prev_prices))
        # normalized_weights = weights / np.sum(weights)

        # prev_price = np.dot(normalized_weights, prev_prices)

        ema = self.exponential_moving_average(prices=prev_prices)
        trend = self.get_trend(prices=prev_prices)

        prev_price = ema + trend * 3

        for i in range(n):

            W_T = np.random.normal(0, np.sqrt(T)) # brownian motion

            # Number of jumps (Poisson)
            N_T = np.random.poisson(lamb * T)

            # Sum of jump magnitudes (log-normal in log-space)
            jump_sum = np.sum(np.random.normal(v, delta, size=N_T)) if N_T > 0 else 0.0

            # Combine terms
            drift = (mu - lamb * kappa - 0.5 * sigma**2) * T
            diffusion = sigma * W_T

            log_S = np.log(prev_price) + (drift + diffusion + jump_sum) * beta
            S_T = np.exp(log_S)
            avg += S_T

        return 2000 - (avg / n - 2000)
        # return avg / n

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
                T=5,
                mu=0.0336,#does
                lamb=0.0000000001,
                sigma=0.000009,#does
                v=1.0000000,
                delta=5.000000,
                prev_prices= list(price_cache["SQUID_INK"]) if "SQUID_INK" in price_cache else [2000.0],   # FIX: Pass a float instead of the entire price_cache
                mu_w=0.5,
                sigma_w=0.5)

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

        # BUY
        for price, volume in sell_orders:
            if buy_amount > 0 and price <= max_buy:
                quantity = min(buy_amount, -volume)

                orders.append(Order(product, price, quantity))
                buy_amount -= quantity


        if buy_amount > 0 and hard:
            quantity = buy_amount // 2

            orders.append(Order(product, fair_value - 2, quantity))
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
            position = state.position.get(product, 0)

            # if product == "KELP":
            #     fair_price = self.get_kelp_fair_value(self, state.order_depths[product])

            # # Buy if ask < fair
            # for ask_price in sorted(order_depth.sell_orders):
            #     if ask_price < fair_price:
            #         volume = order_depth.sell_orders[ask_price]
            #         buy_qty = min(-volume, 50 - state.position.get(product, 0))
            #         if buy_qty > 0:
            #             orders.append(Order(product, ask_price, buy_qty))

            # # Sell if bid > fair
            # for bid_price in sorted(order_depth.buy_orders, reverse=True):
            #     if bid_price > fair_price:
            #         volume = order_depth.buy_orders[bid_price]
            #         sell_qty = min(volume, state.position.get(product, 0) + 50)
            #         if sell_qty > 0:
            #             orders.append(Order(product, bid_price, -sell_qty))

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
