"""The built-in learning methods of RoboScript, one module each (docs/how-to/add-a-method.md).

- `wicl`: one demonstration -> S1 imitate -> S2 practice -> frozen Task Memory -> evaluation.
- `skill_builder`: P1 builds a skillset: checks the core's entries on a benchmark, expresses each
  demonstration in skills, and has a coding agent write the missing ones.
- `rsi`: P4 after any method's evaluation: reflect on its records into an RSI document, evaluate
  again reading it, round by round against a paired baseline (`rsi.rounds`).

A method depends on the core (`roboscript`) only, never on a benchmark or robot package.
"""

__version__ = "0.1.0.dev0"
