"""
光链路编排器

将多个 OpticalComponent 串联为端到端的光信号处理链路。
"""
from typing import List
from ..components.base import OpticalComponent, OpticalSignal


class OpticalLink:
    """
    光链路

    将多个 OpticalComponent 按顺序串联，信号依次通过每个组件。

    Example:
        >>> link = OpticalLink([
        ...     EOMModulator(),
        ...     MRRBankFilter(bank),
        ...     Photodetector(),
        ... ])
        >>> output = link.forward(comb_signal)
    """

    def __init__(self, components: List[OpticalComponent] = None):
        self.components: List[OpticalComponent] = components or []

    def add(self, component: OpticalComponent) -> "OpticalLink":
        """添加组件到链路末尾，支持链式调用"""
        self.components.append(component)
        return self

    def forward(self, signal: OpticalSignal) -> OpticalSignal:
        """
        端到端信号传输

        信号依次通过每个组件的 forward() 方法。
        """
        for comp in self.components:
            signal = comp.forward(signal)
        return signal

    def __repr__(self) -> str:
        chain = " → ".join(c.name for c in self.components)
        return f"OpticalLink[{chain}]"
