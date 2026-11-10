"""The RSI method attached to the core's test method (`roboscript.testing.phases.TestMethod`), as
a P4 RSI arm's Agent host loads it (`p4_fakes:TestMethodRsi`) in the pipeline tests."""

from roboscript.testing.phases import TestMethod
from roboscript_methods.rsi.method import attach

TestMethodRsi = attach(TestMethod)
