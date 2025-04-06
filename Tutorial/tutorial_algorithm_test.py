from datamodel import Order, OrderDepth, TradingState
from typing import List, Dict

# First Submission
class Trader:

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        # Initial simple fair values for demo purposes
        fair_prices = {
            "RAINFOREST_RESIN": 10000,
            "KELP": 10000
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


        return result, conversions, traderData
