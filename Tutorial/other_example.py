from datamodel import OrderDepth, Order
from typing import List
import math
import jsonpickle

class Trader:
    def __init__(self):
        self.resin_prices = []
        self.resin_vwap = []

    def kelp_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("KELP", best_ask, quantity))
                    buy_order_volume += quantity

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("KELP", best_bid, -1 * quantity))
                    sell_order_volume += quantity

        baaf = min([p for p in order_depth.sell_orders if p > fair_value + 1], default=fair_value + 2)
        bbbf = max([p for p in order_depth.buy_orders if p < fair_value - 1], default=fair_value - 2)

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("KELP", bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("KELP", baaf - 1, -sell_quantity))

        return orders

    def rainforest_resin_orders(self, order_depth: OrderDepth, timespan: int, width: float, take_width: float, position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if len(order_depth.sell_orders) == 0 or len(order_depth.buy_orders) == 0:
            return orders

        best_ask = min(order_depth.sell_orders.keys())
        best_bid = max(order_depth.buy_orders.keys())

        filtered_ask = [p for p in order_depth.sell_orders if abs(order_depth.sell_orders[p]) >= 15]
        filtered_bid = [p for p in order_depth.buy_orders if abs(order_depth.buy_orders[p]) >= 15]
        mm_ask = min(filtered_ask) if filtered_ask else best_ask
        mm_bid = max(filtered_bid) if filtered_bid else best_bid

        mid_price = (mm_ask + mm_bid) / 2
        self.resin_prices.append(mid_price)

        volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
        vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] + best_ask * order_depth.buy_orders[best_bid]) / volume
        self.resin_vwap.append({"vol": volume, "vwap": vwap})

        if len(self.resin_vwap) > timespan:
            self.resin_vwap.pop(0)
        if len(self.resin_prices) > timespan:
            self.resin_prices.pop(0)

        fair_value = sum([x["vwap"] * x['vol'] for x in self.resin_vwap]) / sum([x['vol'] for x in self.resin_vwap])
        fair_value = mid_price  # Optionally override VWAP with mid_price

        if best_ask <= fair_value - take_width and -1 * order_depth.sell_orders[best_ask] <= 20:
            quantity = min(-1 * order_depth.sell_orders[best_ask], position_limit - position)
            if quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                buy_order_volume += quantity

        if best_bid >= fair_value + take_width and order_depth.buy_orders[best_bid] <= 20:
            quantity = min(order_depth.buy_orders[best_bid], position_limit + position)
            if quantity > 0:
                orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                sell_order_volume += quantity

        aaf = [p for p in order_depth.sell_orders if p > fair_value + 1]
        bbf = [p for p in order_depth.buy_orders if p < fair_value - 1]
        baaf = min(aaf) if aaf else fair_value + 2
        bbbf = max(bbf) if bbf else fair_value - 2

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", bbbf + 1, buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", baaf - 1, -sell_quantity))

        return orders

    def run(self, state):
        result = {}
        resin_timespan = 10
        resin_width = 3.5
        resin_take_width = 1
        resin_position_limit = 20
        kelp_fair_value = 10000
        kelp_width = 2
        kelp_position_limit = 20

        kelp_pos = state.position.get("KELP", 0)
        if "KELP" in state.order_depths:
            result["KELP"] = self.kelp_orders(state.order_depths["KELP"], kelp_fair_value, kelp_width, kelp_pos, kelp_position_limit)

        resin_pos = state.position.get("RAINFOREST_RESIN", 0)
        if "RAINFOREST_RESIN" in state.order_depths:
            result["RAINFOREST_RESIN"] = self.rainforest_resin_orders(
                state.order_depths["RAINFOREST_RESIN"], resin_timespan, resin_width,
                resin_take_width, resin_pos, resin_position_limit
            )

        trader_data = jsonpickle.encode({"resin_prices": self.resin_prices, "resin_vwap": self.resin_vwap})
        conversions = 1 

        return result, conversions, trader_data
