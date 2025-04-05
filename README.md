# imc-prosperity-3-aura-farmers

## Textbooks

Stochastic Calculus for Finance[https://cms.dm.uba.ar/academico/materias/2docuat2016/analisis_cuantitativo_en_finanzas/Steve_Shreve_Stochastic_Calculus_for_Finance_I.pdf]



## Trading Algorithms

### 1. Market Making

You continuously place **buy (bid)** and **sell (ask)** limit orders near the current market price, aiming to earn the **spread** (the difference between the two prices).

#### How it works:
- Place a **buy** order just below fair value
- Place a **sell** order just above fair value
- Wait for other traders to hit your orders
- Manage inventory (avoid too much long or short exposure)

#### Pros:
- Steady income if volume is high
- Provides market liquidity
- Easy to implement basic version

#### Cons:
- Risky in volatile markets
- You can get stuck with inventory
- Requires good inventory management

#### Use When:
- **Stable or range-bound markets**
- Assets with **tight spreads and high volume**
- You want consistent profits with low directional risk

#### Best Data/Patterns:
- Order book data (bid/ask levels, depth)
- Mid-price stability
- Historical volatility for spread setting

---

### 2. Market Taking / Aggressive Execution

#### What it is:
Place orders that **immediately execute** at the best available price — buying the ask or selling the bid — based on a strong conviction of price movement.

#### How it works:
- Determine if current ask is **cheap** → buy it  
- Determine if current bid is **expensive** → sell into it  
- Use a **fair value model** to justify decisions  

#### Example:
If fair value is $105 and ask is $102 → buy immediately

#### Pros:
- Instant execution  
- Can take advantage of mispricings  

#### Cons:
- You always **pay the spread**  
- Higher trading cost and slippage  

#### Use When:
- Strong mispricing signals exist  
- You need fast entry/exit  
- Speed matters more than price (e.g., volatile markets)  

#### Best Data/Patterns:
- Real-time order book and best bid/ask  
- Fair value estimates (VWAP, EMA)  
- Short-term signal triggers (price jumps)  

---

### 3. Mean Reversion

#### What it is:
Betting that a price that has moved far from its **average (mean)** will revert back.

#### How it works:
- Track moving average or VWAP  
- If price goes too high → sell  
- If price goes too low → buy  

#### Example:
- 10-period moving average is $100  
- Price spikes to $106 → sell expecting a drop  

#### Pros:
- Works well in range-bound markets  
- High win rate if properly tuned  

#### Cons:
- Can lead to losses in trending markets  
- Requires good calibration of thresholds  

#### Use When:
- Assets are **mean-reverting** (e.g., ETFs, currency pairs)  
- Low volatility environments  
- Price tends to oscillate around a stable average  

#### Best Data/Patterns:
- Moving averages (SMA, EMA)  
- Bollinger Bands  
- Historical mean spread levels  

---

### 4. Momentum Trading

#### What it is:
Buy when prices are rising and sell when prices are falling — riding the wave.

#### How it works:
- Detect trend via moving average crossover or price delta  
- Go with the direction: Buy high, sell higher  

#### Example:
- Price increases 3 ticks in a row → signal to buy  

#### Pros:
- Big gains if trends continue  
- Easy to implement  

#### Cons:
- False signals in sideways markets  
- Slippage if too aggressive  

#### Use When:
- Trending markets (up or down)  
- Price is making higher highs or lower lows  
- You want to capture breakouts  

#### Best Data/Patterns:
- Price deltas over time  
- Moving average crossovers  
- RSI or MACD indicators  

---

### 5. Statistical Arbitrage (Stat Arb)

#### What it is:
Trading based on statistical relationships between multiple assets — typically **pairs trading**.

#### How it works:
- Identify two correlated assets (e.g., PEARLS and BANANAS)  
- When their prices diverge, bet they’ll return to their average ratio  
- Long one, short the other  

#### Example:
- PEARLS usually = 2× BANANAS  
- Now PEARLS is 210, BANANAS is 90 → expect convergence  

#### Pros:
- Market-neutral  
- Can be profitable in all market conditions  

#### Cons:
- Relationship may break  
- Requires statistical validation  

#### Use When:
- Two highly correlated instruments are available  
- Market conditions are neutral or sideways  
- You want low directional exposure  

#### Best Data/Patterns:
- Price ratio spreads  
- Z-score or mean reversion metrics  
- Correlation matrices  

---

### 6. VWAP / TWAP Execution

#### What it is:
Smart execution strategies that aim to minimize **market impact** when trading large sizes.

#### VWAP = Volume Weighted Average Price
- Buy/sell in proportion to market volume  

#### TWAP = Time Weighted Average Price
- Spread trades evenly over time  

#### How it works:
- Break large order into smaller ones  
- Execute gradually to avoid moving the market  

#### Pros:
- Minimized price slippage  
- Avoids signaling large trades  

#### Cons:
- Execution may lag in fast markets  
- Not ideal for short-term trades  

#### Use When:
- Executing large trades  
- Illiquid markets or when concealment is key  
- Benchmarked execution is required (e.g., institutional)  

#### Best Data/Patterns:
- Historical volume data  
- Time intervals and execution windows  

---

### 7. Inventory Skewing (Risk-Adjusted Market Making)

#### What it is:
Adjust your bid/ask prices based on your **current position** to reduce risk.

#### How it works:
- Long position → lower ask price  
- Short position → raise bid price  
- Adjust fair value: `adjusted_fv = fv - risk_factor * position`  

#### Pros:
- Prevents overexposure  
- Keeps inventory manageable  

#### Cons:
- Can reduce profitability if too conservative  
- Needs tuning for each market  

#### Use When:
- Market making strategies  
- Inventory approaches position limits  
- You need to actively reduce exposure  

#### Best Data/Patterns:
- Current position/inventory  
- Fair value estimates  
- Market volatility (to adjust skew sensitivity)  

---

### 8. Order Book Imbalance Strategy

#### What it is:
Looks at **volume imbalance** between bids and asks to predict short-term price movements.

#### How it works:
- Large buy volume vs. sell volume → bullish  
- Large sell volume vs. buy volume → bearish  

#### Example:
```python
if total_bids >> total_asks:
    # Buy signal
