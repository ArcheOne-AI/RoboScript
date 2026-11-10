"""`wicl+rsi`: wicl with P4 attached (`rsi.method.attach`), registered in
`roboscript.methods` so that a config names it (an Agent host's `method`, which only loads a
registered name or a module attribute, never a class made at run time). Its Agent host runs
wicl's `p3` with wicl's prompts; an RSI document read beside the Task Memory
(`agent.overlay`) is the only difference from `wicl` there."""

from typing import Final

from roboscript.learning.method import Declaration
from roboscript_methods.rsi.method import attach
from roboscript_methods.wicl.method import Wicl

WiclRsi: Final[type[Declaration]] = attach(Wicl)
