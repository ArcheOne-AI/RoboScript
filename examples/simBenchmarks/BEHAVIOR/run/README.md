# Recipe bindings

Method bindings for BEHAVIOR: the generic recipe lives in `recipes/<method>/`, and
`pipeline.example.yaml` here adds the BEHAVIOR-specific part.

`wicl` requires `reruns=True`, which BEHAVIOR does not offer. A method for BEHAVIOR declares
`reruns=False` in its `requires` and judges by the terminal q-score.
