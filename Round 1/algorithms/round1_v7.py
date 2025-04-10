from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict
import numpy as np
import math
import jsonpickle
from collections import deque

# Product parameter mapping:
#   RAINFOREST_RESIN uses the former AMETHYSTS parameters (resin strategy)
#   KELP uses the former STARFRUIT parameters (kelp strategy)
#   SQUID_INK remains unchanged in its fair value calculation,
#       but now we also build orders for it using defaults.
PARAMS = {
    "RAINFOREST_RESIN": {  # resin strategy (was AMETHYSTS)
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,  # disregard orders near fair value
        "join_edge": 2,       # join orders inside this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    "KELP": {  # kelp strategy (was STARFRUIT)
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    }
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        # For resin and kelp strategies, use specific position limits:
        self.LIMIT = {"RAINFOREST_RESIN": 20, "KELP": 20}
        # For products not explicitly in LIMIT (like SQUID_INK) use this default limit:
        self.limit = 50  
        self.windows = {}

    # --- Utility Functions ---
    def exponential_moving_average(self, prices, alpha=0.3):
        if not prices:
            # Return a default value (e.g., 2000.0, or any other default price you deem appropriate)
            return 2000.0
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

    def get_fair_value_merton(self, T: float, mu: float, lamb: float, sigma: float, v: float,
                              delta: float, prev_prices: List[float], mu_w: float, sigma_w: float,
                              n: int = 100, beta: float = 0.47):
        kappa = np.exp(v + 0.5 * delta**2) - 1
        avg = 0
        ema = self.exponential_moving_average(prev_prices)
        trend = self.get_trend(prev_prices)
        prev_price = ema + trend * 3

        for i in range(n):
            W_T = np.random.normal(0, np.sqrt(T))
            N_T = np.random.poisson(lamb * T)
            jump_sum = np.sum(np.random.normal(v, delta, size=N_T)) if N_T > 0 else 0.0
            drift = (mu - lamb * kappa - 0.5 * sigma**2) * T
            diffusion = sigma * W_T
            log_S = np.log(prev_price) + (drift + diffusion + jump_sum) * beta
            S_T = np.exp(log_S)
            avg += S_T

        return round(2000 - (avg / n - 2000))

    # --- Fair Value Functions for individual products ---
    # For SQUID_INK, we use the Merton function (unchanged) and then later apply order logic.
    # For KELP, we use the former STARFRUIT logic.
    def get_kelp_fair_value(self, order_depth: OrderDepth) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price]) >= self.params["KELP"]["adverse_volume"]
            ]
            filtered_bid = [
                price for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price]) >= self.params["KELP"]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mmmid_price = (best_ask + best_bid) / 2
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if hasattr(self, "kelp_last_price"):
                last_price = self.kelp_last_price
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = last_returns * self.params["KELP"]["reversion_beta"]
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            self.kelp_last_price = mmmid_price
            return fair
        return None

    # --- Order Building Functions ---
    # In functions below, if the product is not in self.LIMIT, we default to self.limit.
    def take_best_orders(self, product: str, fair_value: float, take_width: float, orders: List[Order],
                         order_depth: OrderDepth, position: int,
                         buy_order_volume: int, sell_order_volume: int,
                         prevent_adverse: bool = False, adverse_volume: int = 0) -> (int, int):
        position_limit = self.LIMIT[product] if product in self.LIMIT else self.limit
        # Process sell orders for taking (buying)
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(best_ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        # Process buy orders for taking (selling)
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(best_bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def market_make(self, product: str, orders: List[Order], bid: int, ask: int, position: int,
                    buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_limit = self.LIMIT[product] if product in self.LIMIT else self.limit
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))
        return buy_order_volume, sell_order_volume

    def clear_position_order(self, product: str, fair_value: float, width: int, orders: List[Order],
                             order_depth: OrderDepth, position: int,
                             buy_order_volume: int, sell_order_volume: int) -> (int, int):
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)
        position_limit = self.LIMIT[product] if product in self.LIMIT else self.limit
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)
        if position_after_take > 0:
            clear_quantity = sum(volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask)
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)
        if position_after_take < 0:
            clear_quantity = sum(abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid)
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        return buy_order_volume, sell_order_volume

    def take_orders(self, product: str, order_depth: OrderDepth, fair_value: float, take_width: float,
                    position: int, prevent_adverse: bool = False, adverse_volume: int = 0) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
        buy_order_volume, sell_order_volume = self.take_best_orders(
            product, fair_value, take_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume, prevent_adverse, adverse_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(self, product: str, order_depth: OrderDepth, fair_value: float, clear_width: int,
                     position: int, buy_order_volume: int, sell_order_volume: int) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product, fair_value, clear_width, orders, order_depth, position,
            buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    def make_orders(self, product: str, order_depth: OrderDepth, fair_value: float, position: int,
                    buy_order_volume: int, sell_order_volume: int,
                    disregard_edge: float, join_edge: float, default_edge: float,
                    manage_position: bool = False, soft_position_limit: int = 0) -> (List[Order], int, int):
        orders: List[Order] = []
        asks_above_fair = [price for price in order_depth.sell_orders.keys() if price > fair_value + disregard_edge]
        bids_below_fair = [price for price in order_depth.buy_orders.keys() if price < fair_value - disregard_edge]
        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None
        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1
        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1
        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1
        buy_order_volume, sell_order_volume = self.market_make(
            product, orders, bid, ask, position, buy_order_volume, sell_order_volume
        )
        return orders, buy_order_volume, sell_order_volume

    # --- Run Method ---
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""
        k = 200  # window size for price cache

        try:
            traderData = jsonpickle.decode(state.traderData)
        except Exception:
            traderData = {}
        if traderData is None:
            traderData = {}
        price_cache = traderData["price_cache"] if "price_cache" in traderData else {}

        # Process each product in the order depths
        for product in state.order_depths:
            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "SQUID_INK":
                fair_value = self.get_fair_value_merton(
                    T=5,
                    mu=0.0336,
                    lamb=0.0000000001,
                    sigma=0.000009,
                    v=1.0,
                    delta=5.0,
                    prev_prices=list(price_cache["SQUID_INK"]) if "SQUID_INK" in price_cache else [2000.0],
                    mu_w=0.5,
                    sigma_w=0.5
                )
                # Set SQUID_INK default parameters for order construction:
                squid_take_width = 1
                squid_clear_width = 0
                squid_default_edge = 2
                squid_disregard_edge = 1
                squid_join_edge = 1

                squid_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    "SQUID_INK", order_depth, fair_value, squid_take_width, position
                )
                squid_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    "SQUID_INK", order_depth, fair_value, squid_clear_width, position,
                    buy_order_volume, sell_order_volume
                )
                squid_make_orders, _, _ = self.make_orders(
                    "SQUID_INK", order_depth, fair_value, position,
                    buy_order_volume, sell_order_volume,
                    squid_disregard_edge, squid_join_edge, squid_default_edge,
                    manage_position=False
                )
                orders = squid_take_orders + squid_clear_orders + squid_make_orders

            elif product == "KELP":
                fair_value = self.get_kelp_fair_value(order_depth)
                kelp_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    "KELP", order_depth, fair_value, self.params["KELP"]["take_width"],
                    position, self.params["KELP"]["prevent_adverse"], self.params["KELP"]["adverse_volume"]
                )
                kelp_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    "KELP", order_depth, fair_value, self.params["KELP"]["clear_width"],
                    position, buy_order_volume, sell_order_volume
                )
                kelp_make_orders, _, _ = self.make_orders(
                    "KELP", order_depth, fair_value, position,
                    buy_order_volume, sell_order_volume,
                    self.params["KELP"]["disregard_edge"], self.params["KELP"]["join_edge"],
                    self.params["KELP"]["default_edge"]
                )
                orders = kelp_take_orders + kelp_clear_orders + kelp_make_orders

            elif product == "RAINFOREST_RESIN":
                fair_value = self.params["RAINFOREST_RESIN"]["fair_value"]
                resin_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                    "RAINFOREST_RESIN", order_depth, fair_value, self.params["RAINFOREST_RESIN"]["take_width"],
                    position
                )
                resin_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                    "RAINFOREST_RESIN", order_depth, fair_value, self.params["RAINFOREST_RESIN"]["clear_width"],
                    position, buy_order_volume, sell_order_volume
                )
                resin_make_orders, _, _ = self.make_orders(
                    "RAINFOREST_RESIN", order_depth, fair_value, position,
                    buy_order_volume, sell_order_volume,
                    self.params["RAINFOREST_RESIN"]["disregard_edge"],
                    self.params["RAINFOREST_RESIN"]["join_edge"],
                    self.params["RAINFOREST_RESIN"]["default_edge"],
                    True,
                    self.params["RAINFOREST_RESIN"]["soft_position_limit"]
                )
                orders = resin_take_orders + resin_clear_orders + resin_make_orders

            else:
                # Default fallback: compute mid-price and use that as fair value
                buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
                sell_orders = sorted(order_depth.sell_orders.items())
                popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0] if buy_orders else 0
                popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0] if sell_orders else 0
                fair_value = round((popular_buy_price + popular_sell_price) / 2) if popular_buy_price and popular_sell_price else 10000
                orders = []

            result[product] = orders

            # Update price cache (used for Merton estimation on SQUID_INK)
            if product not in price_cache:
                price_cache[product] = deque()
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2.0
                k_window = k
                if len(price_cache[product]) < k_window:
                    price_cache[product].append(mid_price)
                else:
                    price_cache[product].popleft()
                    price_cache[product].append(mid_price)

        traderData = jsonpickle.encode({"price_cache": price_cache})
        return result, conversions, traderData
