"""Entry point for running the Bitcoin reinforcement-learning trader."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import alpaca_trade_api as tradeapi
import gym
import numpy as np
import yfinance as yf
from gym.spaces import Box, Dict as DictSpace
from stable_baselines3 import DDPG

LOGGER = logging.getLogger("bitcoin_trader")


def _float_env(name: str) -> float:
    value = os.getenv(name)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - defensive path
        raise RuntimeError(f"Environment variable {name} must be a numeric value") from exc


def _str_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass
class MarketWindow:
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    adj_closes: np.ndarray
    volumes: np.ndarray

    def to_dict(self) -> Dict[str, np.ndarray]:
        return {
            "Open": self.opens,
            "High": self.highs,
            "Low": self.lows,
            "Close": self.closes,
            "Adj Close": self.adj_closes,
            "Volume": self.volumes,
        }


class BitcoinTradingEnv(gym.Env):
    """Minimal Alpaca-powered trading environment for inference."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        endpoint_url: str = "https://paper-api.alpaca.markets",
        symbol: str = "BTC-USD",
        window: int = 30,
    ) -> None:
        super().__init__()

        self._alpaca = tradeapi.REST(api_key, api_secret, endpoint_url, api_version="v2")
        self._symbol = symbol
        self._window = window
        self._position = 0.0

        self.action_space = Box(low=-0.25, high=0.25, shape=(1,), dtype=np.float32)
        market_space = Box(low=0, high=np.inf, shape=(window,), dtype=np.float32)
        self.observation_space = DictSpace(
            {
                "Balance": Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                "Bitcoins": Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
                **{key: market_space for key in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]},
            }
        )

    def reset(self) -> Dict[str, np.ndarray]:
        self._position = 0.0
        return self._fetch_state()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, float]]:
        action_value = float(np.clip(action[0], self.action_space.low[0], self.action_space.high[0]))
        account = self._alpaca.get_account()
        last_price = self._latest_price()

        if action_value > 0:
            usd_to_spend = float(account.cash) * action_value
            quantity = usd_to_spend / last_price if last_price else 0.0
            if quantity > 0:
                self._alpaca.submit_order(symbol="BTCUSD", qty=quantity, side="buy", type="market", time_in_force="gtc")
                self._position += quantity
                LOGGER.info("Placed BUY order for %.6f BTC", quantity)
        elif action_value < 0 and self._position > 0:
            quantity = min(self._position, -action_value * self._position)
            self._alpaca.submit_order(symbol="BTCUSD", qty=quantity, side="sell", type="market", time_in_force="gtc")
            self._position -= quantity
            LOGGER.info("Placed SELL order for %.6f BTC", quantity)

        state = self._fetch_state()
        net_worth = float(self._alpaca.get_account().equity)
        reward = net_worth

        return state, reward, False, {"net_worth": net_worth}

    def render(self, mode: str = "human") -> None:
        account = self._alpaca.get_account()
        LOGGER.info("Balance USD: %s | Equity: %s | BTC held: %.6f", account.cash, account.equity, self._position)

    def _latest_price(self) -> float:
        barset = self._alpaca.get_crypto_bars("BTCUSD", "1Min", limit=1)
        try:
            return float(barset[-1].c)
        except (IndexError, AttributeError):
            return 0.0

    def _fetch_state(self) -> Dict[str, np.ndarray]:
        window = self._download_window()
        account = self._alpaca.get_account()

        state = {
            "Balance": np.array([float(account.cash)], dtype=np.float32),
            "Bitcoins": np.array([self._position], dtype=np.float32),
        }
        state.update({key: value.astype(np.float32) for key, value in window.to_dict().items()})

        return state

    def _download_window(self) -> MarketWindow:
        data = yf.download(tickers=self._symbol, period="{}m".format(self._window + 1), interval="1m")
        if data.empty:
            raise RuntimeError("No market data returned from yfinance")

        tail = data.tail(self._window)
        return MarketWindow(
            opens=tail["Open"].to_numpy(),
            highs=tail["High"].to_numpy(),
            lows=tail["Low"].to_numpy(),
            closes=tail["Close"].to_numpy(),
            adj_closes=tail["Adj Close"].to_numpy(),
            volumes=tail["Volume"].to_numpy(),
        )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DDPG Bitcoin trading agent in paper-trading mode.")
    parser.add_argument("--model-path", default="models/ddpg_bitcoin_hist.zip", help="Path to the trained DDPG model.")
    parser.add_argument("--poll-interval", type=int, default=300, help="Seconds to wait between predictions.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    api_key = _str_env("APCA_API_KEY_ID")
    api_secret = _str_env("APCA_API_SECRET_KEY")
    endpoint_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    env = BitcoinTradingEnv(api_key=api_key, api_secret=api_secret, endpoint_url=endpoint_url)
    LOGGER.info("Initialising trading environment against %s", endpoint_url)

    model = DDPG.load(args.model_path)
    LOGGER.info("Loaded model from %s", args.model_path)

    observation = env.reset()
    while True:
        action, _state = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        LOGGER.info("Net worth %.2f | reward %.2f | info %s", info.get("net_worth", 0.0), reward, info)
        if done:
            observation = env.reset()
        env.render()
        time.sleep(args.poll_interval)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
