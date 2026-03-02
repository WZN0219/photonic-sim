"""
光学组件抽象基类与光信号数据结构

所有光学组件继承 OpticalComponent，实现 forward(signal) -> signal 接口，
使得任意组件可像积木一样串联为光链路。
"""
from abc import ABC, abstractmethod
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpticalSignal:
    """
    多波长光信号

    表示一组 WDM 通道上的光信号状态。
    每个通道包含波长、功率和可选的承载数据。

    Attributes:
        wavelengths: 各通道波长 [nm], shape=(N,)
        powers:      各通道功率 [mW], shape=(N,)
        data:        各通道承载的数据 (可选), shape=(N,) 或 (N, M)
    """
    wavelengths: np.ndarray
    powers: np.ndarray
    data: Optional[np.ndarray] = None

    @property
    def num_channels(self) -> int:
        return len(self.wavelengths)

    def copy(self) -> "OpticalSignal":
        """深拷贝"""
        return OpticalSignal(
            wavelengths=self.wavelengths.copy(),
            powers=self.powers.copy(),
            data=self.data.copy() if self.data is not None else None,
        )

    def __repr__(self) -> str:
        return (f"OpticalSignal(channels={self.num_channels}, "
                f"λ=[{self.wavelengths[0]:.2f}..{self.wavelengths[-1]:.2f}]nm, "
                f"P_avg={np.mean(self.powers):.3f}mW)")


class OpticalComponent(ABC):
    """
    光学组件抽象基类

    所有光学组件（MRR、调制器、探测器、光纤等）继承此类，
    实现 forward() 方法处理光信号。
    """

    @abstractmethod
    def forward(self, signal: OpticalSignal) -> OpticalSignal:
        """
        处理输入光信号

        Args:
            signal: 输入光信号

        Returns:
            处理后的光信号
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """组件名称"""
        ...

    def __repr__(self) -> str:
        return f"{self.name}"
