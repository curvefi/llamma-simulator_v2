from abc import ABC, abstractmethod


class BasePriceOracle(ABC):
    @abstractmethod
    def calculate_oracle_prices(self, price_data: list): ...


class EmaPriceOracle(BasePriceOracle):
    def __init__(self, t_exp: int, candle_seconds: int = 60):
        if t_exp <= 0 or candle_seconds <= 0:
            raise ValueError("EMA horizon and candle duration must be positive")
        self.t_exp = t_exp  # in seconds
        self.candle_seconds = candle_seconds

    def calculate_oracle_prices(self, price_data: list) -> list:
        """Oracle available at each candle open; the first candle is warm-up.

        Timestamps are seconds. A close becomes available at its bar end,
        including across gaps; it must never affect its own opening oracle.
        """
        data = []
        ema = ema_t = previous = None
        for t, _, _, _, close, _ in price_data:
            if previous is not None:
                available_at, previous_close = previous
                if available_at > t:
                    raise ValueError("Candles must be ordered and non-overlapping")
                if ema is None:
                    ema = previous_close
                else:
                    ema_mul = 2 ** (-(available_at - ema_t) / self.t_exp)
                    ema = ema * ema_mul + previous_close * (1 - ema_mul)
                ema_t = available_at
            data.append(ema)
            previous = (t + self.candle_seconds, close)

        return data
