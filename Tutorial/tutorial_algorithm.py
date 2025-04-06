# Create the algorithm


from datamodel import Order, OrderDepth, TradingState
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Dict, Any
import json


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()

class Trader:
    def __init__(self):
        self.resin_prices: List[float] = []

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        conversions = 0
        traderData = ""

        fair_price_kelp = 10000  # fixed fair value for KELP

        for product in state.order_depths:
            orders: List[Order] = []
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            # === Mean Reversion for RAINFOREST_RESIN ===
            if product == "RAINFOREST_RESIN":
                best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None
                best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None

                if best_ask is not None and best_bid is not None:
                    mid_price = (best_ask + best_bid) / 2
                    self.resin_prices.append(mid_price)

                    # Keep only the last 10 prices
                    if len(self.resin_prices) > 10:
                        self.resin_prices.pop(0)

                    avg_price = sum(self.resin_prices) / len(self.resin_prices)

                    traderData += f"{product}: Mid={mid_price:.1f}, Avg={avg_price:.1f}, Pos={position}\n"

                    # Buy logic
                    if mid_price < avg_price - 1:
                        for ask_price in sorted(order_depth.sell_orders):
                            if ask_price < avg_price:
                                volume = -order_depth.sell_orders[ask_price]
                                buy_qty = min(volume, 50 - position)
                                if buy_qty > 0:
                                    orders.append(Order(product, ask_price, buy_qty))
                                    position += buy_qty

                    # Sell logic
                    elif mid_price > avg_price + 1:
                        for bid_price in sorted(order_depth.buy_orders, reverse=True):
                            if bid_price > avg_price:
                                volume = order_depth.buy_orders[bid_price]
                                sell_qty = min(volume, position + 50)
                                if sell_qty > 0:
                                    orders.append(Order(product, bid_price, -sell_qty))
                                    position -= sell_qty

            # === Threshold Fair Price Strategy for KELP ===
            elif product == "KELP":
                traderData += f"{product}: Pos={position}, Fair={fair_price_kelp}\n"

                for ask_price in sorted(order_depth.sell_orders):
                    if ask_price < fair_price_kelp:
                        volume = -order_depth.sell_orders[ask_price]
                        buy_qty = min(volume, 50 - position)
                        if buy_qty > 0:
                            orders.append(Order(product, ask_price, buy_qty))
                            position += buy_qty

                for bid_price in sorted(order_depth.buy_orders, reverse=True):
                    if bid_price > fair_price_kelp:
                        volume = order_depth.buy_orders[bid_price]
                        sell_qty = min(volume, position + 50)
                        if sell_qty > 0:
                            orders.append(Order(product, bid_price, -sell_qty))
                            position -= sell_qty

            result[product] = orders
        
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData

