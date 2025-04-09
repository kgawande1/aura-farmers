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
            beta: float = 2):
        
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




    def market_making_strategy(self, product, price_cache, position, order_depth, fair_value, traderData, rho=1, k=10):
            
        orders = []

        if product not in price_cache:
            price_cache[product] = deque(maxlen=k)

        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders, price_cache
        


        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())


        mid_price = (best_bid + best_ask) / 2.0

        price_cache[product].append(mid_price)

        proportion = sum( 1 for i in range(1, len(price_cache[product])) if price_cache[product][i] > price_cache[product][i - 1])

            

        Su = proportion / k


        Su_transform = min(max(Su, 0.1), 1.0)


        best_bid_amount = order_depth.buy_orders[best_bid]
        best_ask_amount = order_depth.sell_orders[best_ask]

        buy_price = best_bid + 1
        sell_price = best_ask - 1
        buy_qty = int(min(Su_transform * (self.limits[product] - position), best_bid_amount * Su_transform))
        sell_qty = int(min(Su_transform * (self.limits[product]) + position, best_ask_amount * Su_transform))


        if buy_price < fair_value - rho:
            orders.append(Order(product, buy_price, buy_qty))
        elif (sell_price > fair_value + rho):
            orders.append(Order(product, sell_price, -sell_qty))

        traderData = jsonpickle.encode({
            "price_cache": price_cache
        })


        return orders, traderData


        


    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        k = 200  # window size for merton estimation

        try:
            traderData = jsonpickle.decode(state.traderData)
        except Exception as _:
            traderData = {}

        if traderData is None:
            traderData = {}

        price_cache = traderData["price_cache"] if "price_cache" in traderData else {}

        # Estimate fair value for SQUID_INK using Merton model
        ink_value = self.get_fair_value_merton(
            T=100,
            mu=0.00032542,
            lamb=0.000223,
            sigma=0.0003324,
            v=0,
            delta=0.00006245,
            prev_prices=list(price_cache["SQUID_INK"]) if "SQUID_INK" in price_cache else [2000],
            mu_w=0.5,
            sigma_w=0.5,
        )

        fair_prices = {
            "RAINFOREST_RESIN": 10000,
            "KELP": 10000,
            "SQUID_INK": ink_value
        }

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)
            fair_value = fair_prices[product]

            orders, traderData = self.market_making_strategy(
                product=product,
                price_cache=price_cache,
                position=position,
                order_depth=order_depth,
                fair_value=fair_value,
                traderData=traderData,
                rho=0.0,
                k=k
            )

            result[product] = orders

        
        return result, conversions, traderData
