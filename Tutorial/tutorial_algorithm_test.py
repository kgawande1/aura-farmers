from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import numpy as np
import types
import jsonpickle


# First Submission
class Trader:

    def get_fair_value_merton(
            self,
            T: float,
            mu: float, 
            lamb: float, 
            sigma: float, 
            v: float,
            delta: float, 
            prev_price: float,
            mu_w: float, 
            sigma_w: float,
            prev_w: float,
            n: int = 100,
            m: int = 5,
            beta: float = 0.01):
        
        # n -> number of simulations
        # m -> number of future walk

        kappa = np.exp(v + 0.5 * pow(delta, 2)) - 1
        avg = 0

        for _ in range(n):

            W_T = np.random.normal(0, np.sqrt(T)) # brownian motion

            # Number of jumps (Poisson)
            N_T = np.random.poisson(lamb * T)

            # Sum of jump magnitudes (log-normal in log-space)
            jump_sum = np.sum(np.random.normal(v, delta, size=N_T)) if N_T > 0 else 0.0

            # Combine terms
            drift = (mu - lamb * kappa - 0.5 * sigma**2) * T
            diffusion = sigma * W_T

            log_S = np.log(prev_price) + (drift + diffusion + jump_sum) / beta
            S_T = np.exp(log_S)
            avg += S_T

        return 2000 - (avg / n - 2000)
        # return avg / n

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        try:
            traderData = jsonpickle.decode(state.traderData)
        except Exception as _:
            traderData = {}

        if traderData is None:
            traderData = {}

        price_cache = traderData["price_cache"] if "price_cache" in traderData else {}


        ink_value = self.get_fair_value_merton(
            T=100,
            mu=50.5149,
            lamb=10.5,
            sigma=20.0288,
            v=0,
            delta=10.5,
            prev_price= price_cache["SQUID_INK"] if "SQUID_INK" in price_cache else 2000,   # FIX: Pass a float instead of the entire price_cache
            mu_w=0.5,
            sigma_w=0.5,
            prev_w=0
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
                price_cache[product] = []

            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0

            price_cache[product] = mid_price

            result[product] = orders


        traderData = jsonpickle.encode({
            "price_cache": price_cache
        })

        return result, conversions, traderData
