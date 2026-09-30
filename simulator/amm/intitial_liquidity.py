from abc import ABC, abstractmethod
from typing import Any


class BaseRangeInitialLiquidity(ABC):

    def __init__(self, p0: float, dn: int):
        self.p0 = p0
        self.dn = dn

    @abstractmethod
    def deposit(self, amm: Any, initial_liquidity: float): ...


class ConstantInitialLiquidity(BaseRangeInitialLiquidity):

    def deposit(self, amm: Any, initial_liquidity: float):
        # Preserve the simulator's bands 1..N without letting restored oracle
        # memory move the position down an extra band via deposit_nrange().
        q = (amm.A - 1) / amm.A
        amm.deposit_range(initial_liquidity, self.p0 * q ** (self.dn - 1), self.p0)
