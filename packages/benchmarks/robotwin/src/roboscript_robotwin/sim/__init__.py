"""The RoboTwin env server (simulator side): import only in RoboTwin's Python environment.

`host` serves the env planes and runs everything simulated on one thread through `simulator`;
`official` reproduces RoboTwin's own episode setup; `shim` is the only code reading the task
object for the agent plane; `referee` holds the ground truth of the control plane.
"""
