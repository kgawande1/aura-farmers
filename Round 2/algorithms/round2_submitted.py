# Our algorithm for round1 final

import json
from abc import abstractmethod
from collections import deque
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, TypeAlias
import numpy as np
import jsonpickle

JSON: TypeAlias = dict[str, "JSON"] | list["JSON"] | str | int | float | bool | None

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

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
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

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

        return value[:max_length - 3] + "..."

logger = Logger()

class Strategy:
    def __init__(self, symbol: str, limit: int) -> None:
        self.symbol = symbol
        self.limit = limit

    @abstractmethod
    def act(self, state: TradingState) -> None:
        raise NotImplementedError()

    def run(self, state: TradingState) -> list[Order]:
        self.orders = []
        self.act(state)
        return self.orders

    def buy(self, price, quantity) -> None:
        price, quantity = int(price), int(quantity)
        self.orders.append(Order(self.symbol, price, quantity))

    def sell(self, price, quantity) -> None:
        price, quantity = int(price), int(quantity)
        self.orders.append(Order(self.symbol, price, -quantity))

    def save(self) -> JSON:
        return None

    def load(self, data: JSON) -> None:
        pass

class MarketMakingStrategy(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Track if we're stuck at a limit
        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size // 2
        hard_liquidate = all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < -self.limit * 0.5 else true_value

        # Hit best sells
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                qty = min(to_buy, -volume)
                self.buy(price, qty)
                to_buy -= qty

        # Add buy liquidity if not fully filled
        if to_buy > 0:
            passive_buy_price = true_value - 2 if soft_liquidate else true_value - 1
            self.buy(passive_buy_price, to_buy)

        # Hit best buys
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                qty = min(to_sell, volume)
                self.sell(price, qty)
                to_sell -= qty

        # Add sell liquidity if not fully filled
        if to_sell > 0:
            passive_sell_price = true_value + 2 if soft_liquidate else true_value + 1
            self.sell(passive_sell_price, to_sell)

    def save(self) -> JSON:
        return list(self.window)

    def load(self, data: JSON) -> None:
        self.window = deque(data, maxlen=self.window_size)

class MarketMakingStrategyResin(Strategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)

        self.window = deque()
        self.window_size = 10

    @abstractmethod
    def get_true_value(state: TradingState) -> int:
        raise NotImplementedError()

    def act(self, state: TradingState) -> None:
        true_value = self.get_true_value(state)

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        self.window.append(abs(position) == self.limit)
        if len(self.window) > self.window_size:
            self.window.popleft()

        soft_liquidate = len(self.window) == self.window_size and sum(self.window) >= self.window_size / 2 and self.window[-1]
        hard_liquidate = len(self.window) == self.window_size and all(self.window)

        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        min_sell_price = true_value + 1 if position < self.limit * -0.5 else true_value

        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                self.buy(price, quantity)
                to_buy -= quantity

        if to_buy > 0 and hard_liquidate:
            quantity = to_buy // 2
            self.buy(true_value, quantity)
            to_buy -= quantity

        if to_buy > 0 and soft_liquidate:
            quantity = to_buy // 2
            self.buy(true_value - 2, quantity)
            to_buy -= quantity

        if to_buy > 0:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            self.buy(price, to_buy)

        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                self.sell(price, quantity)
                to_sell -= quantity

        if to_sell > 0 and hard_liquidate:
            quantity = to_sell // 2
            self.sell(true_value, quantity)
            to_sell -= quantity

        if to_sell > 0 and soft_liquidate:
            quantity = to_sell // 2
            self.sell(true_value + 2, quantity)
            to_sell -= quantity

        if to_sell > 0:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            self.sell(price, to_sell)


class RainforestResinStrategy(MarketMakingStrategyResin):
    def get_true_value(self, state: TradingState) -> int:
        return 10000


class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        if not buy_orders or not sell_orders:
            return 0

        # Top 3 levels weighted by volume
        top_buys = buy_orders[:3]
        top_sells = sell_orders[:3]

        buy_weighted = sum(price * volume for price, volume in top_buys)
        buy_total = sum(volume for _, volume in top_buys)

        sell_weighted = sum(price * (-volume) for price, volume in top_sells)
        sell_total = sum(-volume for _, volume in top_sells)

        buy_avg = buy_weighted / buy_total if buy_total else 0
        sell_avg = sell_weighted / sell_total if sell_total else 0

        return round((buy_avg + sell_avg) / 2)


class SquidInkStrategy(MarketMakingStrategy):
    def __init__(self, symbol: Symbol, limit: int) -> None:
        super().__init__(symbol, limit)
        self.price_history = deque(maxlen=100)

    def get_true_value(self, state: TradingState) -> int:
        order_depth: OrderDepth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        # Cache price
        self.price_history.append(mid_price)

        if len(self.price_history) < 20:
            return round(mid_price)

        mean = np.mean(self.price_history)
        std = np.std(self.price_history)
        z = (mid_price - mean) / std if std != 0 else 0

        # Mean reversion signal
        true_value = mid_price - z * std
        return round(true_value)

    def act(self, state: TradingState) -> None:
        order_depth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        fair_value = self.get_true_value(state)
        position = state.position.get(self.symbol, 0)
        available_to_buy = self.limit - position
        available_to_sell = self.limit + position

        signal_strength = abs(fair_value - mid_price)
        size = max(1, min(int(signal_strength ** 1.2), self.limit // 2))

        if fair_value > mid_price and available_to_buy > 0:
            self.buy(int(mid_price), min(size, available_to_buy))

        if fair_value < mid_price and available_to_sell > 0:
            self.sell(int(mid_price), min(size, available_to_sell))

    def save(self) -> JSON:
        return list(self.price_history)

    def load(self, data: JSON) -> None:
        self.price_history = deque(data, maxlen=100)


class BinomialStrategy(MarketMakingStrategy):

    def __init__(self, symbol, limit):
        super().__init__(symbol, limit)

        np.random.seed(42)

        self.price_history = deque(maxlen=100)

        self.expected_return = {
            "DJEMBES": 0.0001,
            "CROISSANTS": 0.0001,
            "JAMS": 0.0001,
            "PICNIC_BASKET1": 0.0002,
            "PICNIC_BASKET2": 0.0002,
        }
        # values set by the functions run in act()
        self.theta = 0
        self.fair_price = 0

        self.tolerance = {
            "DJEMBES": 10,
            "CROISSANTS": 10,
            "JAMS": 10,
            "PICNIC_BASKET1": 14,
            "PICNIC_BASKET2": 9,
        }

        self.maxlen = {
            "DJEMBES": 100,
            "CROISSANTS": 100,
            "JAMS": 100,
            "PICNIC_BASKET1": 300,
            "PICNIC_BASKET2": 300,
        }

        self.scaling_factor = {
            "DJEMBES": 100,
            "CROISSANTS": 100,
            "JAMS": 100,
            "PICNIC_BASKET1": 100,
            "PICNIC_BASKET2": 100,
        }

    def get_stats(self, data):
        if len(data) < 2:
            # Not enough data to compute a comparison, set defaults
            return 0.5, 0.0
    
        # Calculate indicator for each data point (starting from the second)
        indicators = []
        for i in range(1, len(data)):
            # data[i-1] > data[i] gives 0, else 1
            if data[i-1] > data[i]:
                indicators.append(0)
            else:
                indicators.append(1)
        
        n = len(indicators)  # This is len(data) - 1
        theta_bar = np.mean(indicators)
        
        # Use np.var with ddof=1 to compute sample variance
        sample_variance = np.var(indicators, ddof=1) if n > 1 else 0.0
        
        return theta_bar, sample_variance
    
    def preprocess_params(self, theta_bar, sample_variance, epsilon=1e-6, max_nu = 300):

        sample_variance = max(sample_variance, epsilon)
        nu = (theta_bar * (1 - theta_bar)) / sample_variance - 1
        # Optionally clip nu to an upper bound
        nu = min(nu, max_nu)
        alpha = theta_bar * nu
        beta = (1 - theta_bar) * nu
        return alpha, beta

    def find_trend(self):

        data = list(self.price_history)
        n = len(data)

        if n < 2:
            self.theta = 0.5
            return

        theta_bar, sample_variance = self.get_stats(data=data)

        # alpha, beta = self.preprocess_params(theta_bar=theta_bar, sample_variance=sample_variance)

        nu = (theta_bar * (1 - theta_bar)) / (sample_variance) - 1
        alpha = theta_bar * nu
        beta = (1 - theta_bar) * nu

        self.theta = np.random.beta(abs(alpha) * self.scaling_factor[self.symbol], abs(beta) * self.scaling_factor[self.symbol])
 
    def get_true_value(self, state):

        if len(self.price_history) < 1:
            price = {
                "DJEMBES": 13000,
                "CROISSANTS": 4304,
                "JAMS": 6670,
                "PICNIC_BASKET1": 59289,
                "PICNIC_BASKET2": 30609,
            }
            return price[self.symbol]

        curr_price = self.price_history[-1]
        self.fair_price = self.theta * (curr_price * (1 + self.expected_return[self.symbol])) + (1 - self.theta) * (curr_price * (1 - self.expected_return[self.symbol]))

    def act(self, state: TradingState) -> None:

        # self.price_history = self.parse_trading_history(state=state)
        self.find_trend()
        self.get_true_value(state=TradingState)

        position = state.position.get(self.symbol, 0)
        available_to_buy = self.limit - position
        available_to_sell = self.limit + position

        order_depth = state.order_depths[self.symbol]
        buy_orders = sorted(order_depth.buy_orders.items(), reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items())

        best_bid = buy_orders[0][0]
        best_ask = sell_orders[0][0]

        # make sure that these are sorted in the intended order!

        if self.theta < 0.5:
            
            proportion = 1 - self.theta

            # perform some clipping
            # if proportion >= 0.7:
            #     proportion = 1

            to_buy = available_to_buy * proportion
            to_sell = available_to_sell * proportion

            max_buy_price = self.fair_price - self.tolerance[self.symbol]
            min_sell_price = self.fair_price + self.tolerance[self.symbol]

            # hit the best sells possible (here is where we buy)
            for price, volume in sell_orders:
                if to_buy > 0 and price <= max_buy_price:
                    qty = min(to_buy, -volume)
                    self.buy(int(price), qty)
                    to_buy -= qty

            # passing buying
            if to_buy > 0:
                self.buy(int(max_buy_price - 1), to_buy)
            
            # hit the best buys we can (here is where we sell)
            for price, volume in buy_orders:
                if to_sell > 0 and price >= min_sell_price:
                    qty = min(to_sell, volume)
                    self.sell(int(price), qty)
                    to_sell -= qty

            # passive selling
            if to_sell > 0:
                self.sell(int(min_sell_price + 1), to_sell)

        else:

            proportion = self.theta

            # perform some clipping
            # if proportion >= 0.7:
            #     proportion = 1

            to_buy = available_to_buy * proportion
            to_sell = available_to_sell * proportion

            max_buy_price = self.fair_price - self.tolerance[self.symbol]
            min_sell_price = self.fair_price + self.tolerance[self.symbol] 
            # hit the best sells possible (here is where we buy)
            for price, volume in sell_orders:
                if to_buy > 0 and price <= max_buy_price:
                    qty = min(to_buy, -volume)
                    self.buy(price, qty)

            # passing buying
            if to_buy > 0:
                self.buy(max_buy_price - 1, to_buy)
            
            # hit the best buys we can (here is where we sell)
            for price, volume in buy_orders:
                if to_sell > 0 and price >= min_sell_price:
                    qty = min(to_sell, volume)
                    self.sell(price, qty)
                    to_sell -= qty

            # passive selling
            if to_sell > 0:
                self.sell(min_sell_price + 1, to_sell)

        # roll over the mid price
        mid_price = (best_bid + best_ask) / 2

        # cache price:
        self.price_history.append(mid_price)

    def save(self) -> JSON:
        return list(self.price_history)

    def load(self, data: JSON) -> None:
        self.price_history = deque(data, maxlen=self.maxlen[self.symbol])

class BasketArbitrageStrategy(MarketMakingStrategy):

    def __init__(self, symbol, limit):

        super().__init__(symbol, limit)

        self.croissant_data = deque(maxlen=100)
        self.jam_data = deque(maxlen=100)
        self.djembe_data = deque(maxlen=100)

        self.pb1_spread_history = deque(maxlen=40)
        self.pb2_spread_history = deque(maxlen=40)

    def mid(self, od, symbol):
        order = od.get(symbol)
        if not order or not order.buy_orders or not order.sell_orders:
            return None
        return (max(order.buy_orders) + min(order.sell_orders)) / 2

    def act(self, state: TradingState):
        od = state.order_depths
        pos = state.position
        orders = []

        croissant = self.mid(od, "CROISSANTS")
        jam = self.mid(od, "JAMS")
        djembe = self.mid(od, "DJEMBES")
        pb1 = self.mid(od, "PICNIC_BASKET1")
        pb2 = self.mid(od, "PICNIC_BASKET2")

        if None in [croissant, jam, djembe, pb1, pb2]:
            return orders  # wait for more data

        pb1_fair = 6 * croissant + 3 * jam + 1 * djembe
        pb2_fair = 4 * croissant + 2 * jam

        spread1 = pb1 - pb1_fair
        spread2 = pb2 - pb2_fair

        self.pb1_spread_history.append(spread1)
        self.pb2_spread_history.append(spread2)

        def z_score(history, current):
            if len(history) < 20:
                return 0
            mean = np.mean(history)
            std = np.std(history)
            return (current - mean) / std if std != 0 else 0

        z1 = z_score(self.pb1_spread_history, spread1)
        z2 = z_score(self.pb2_spread_history, spread2)

        threshold, offset = 1, 0.2

        c1_volume, j1_volume, d1_volume = 6, 3, 1
        c2_volume, j2_volume = 4, 2

        if self.symbol == "CROISSANTS":

            if z1 > threshold - offset:
                self.buy(croissant, c1_volume)
            elif z1 < -threshold + offset:
                self.sell(croissant, c1_volume)

            if z2 > threshold - offset:
                self.buy(croissant, c2_volume)
            elif z2 < -threshold + offset:
                self.sell(croissant, c2_volume)

        elif self.symbol == "JAMS":

            if z1 > threshold - offset:
                self.buy(jam, j1_volume)
            elif z1 < -threshold + offset:
                self.sell(jam, j1_volume)

            if z2 > threshold - offset:
                self.buy(jam, j2_volume)
            elif z2 < -threshold + offset:
                self.sell(jam, j2_volume)

        elif self.symbol == "DJEMBES":

            if z1 > threshold - offset:
                self.buy(djembe, d1_volume)
            elif z1 < -threshold + offset:
                self.sell(djembe, d1_volume)

        self.croissant_data.append(croissant)
        self.djembe_data.append(djembe)
        self.jam_data.append(jam)

    def load(self, croissant_data, jam_data, djembe_data):
        self.croissant_data = deque(croissant_data, maxlen=100)
        self.jam_data = deque(jam_data, maxlen=100)
        self.djembe_data = deque(djembe_data, maxlen=100)

    def save(self) -> JSON:
        if self.symbol == "CROISSANTS":
            return list(self.croissant_data)
        elif self.symbol == "JAMS":
            return list(self.jam_data)
        elif self.symbol == "DJEMBES":
            return list(self.djembe_data)


class Trader:
    def __init__(self) -> None:
        limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK" : 50,
            "DJEMBES": 60,
            "CROISSANTS": 250,
            "JAMS": 350,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
        }

        self.strategies = {symbol: clazz(symbol, limits[symbol]) for symbol, clazz in {
            "RAINFOREST_RESIN": RainforestResinStrategy,
            "KELP": KelpStrategy,
            "SQUID_INK" : SquidInkStrategy,
            "DJEMBES": BasketArbitrageStrategy,
            "JAMS": BasketArbitrageStrategy,
            "PICNIC_BASKET1": BinomialStrategy,
            "PICNIC_BASKET2": BinomialStrategy,
            "CROISSANTS": BasketArbitrageStrategy
        }.items()}

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        logger.print(state.position)

        conversions = 0

        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders = {}
        for symbol, strategy in self.strategies.items():

            if symbol in old_trader_data:

                if symbol in ["DJEMBES", "CROISSANTS", "JAMS"]:
                    strategy.load(old_trader_data.get("CROISSANTS", None), old_trader_data.get("JAMS", None), old_trader_data.get("DJEMBES", None))
                else:
                    strategy.load(old_trader_data.get(symbol, None))

            if symbol in state.order_depths:
                orders[symbol] = strategy.run(state)

            new_trader_data[symbol] = strategy.save()

        trader_data = json.dumps(new_trader_data, separators=(",", ":"))

        logger.flush(state, orders, conversions, trader_data)
        return orders, conversions, trader_data