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
        # Phil's oracle-anchored grid places an unshifted position in bands 1..N.
        # Select that range directly, independently of the restored oracle memory.
        q = (amm.A - 1) / amm.A
        amm.deposit_range(initial_liquidity, self.p0 * q ** (self.dn - 1), self.p0)
