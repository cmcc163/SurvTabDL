# Architecture attribution

SurvTabDL packages survival adapters around selected components from the manuscript's TabSurvey-derived code. The original model authors retain ownership of their architecture contributions.

- TabSurvey: https://github.com/kathrinse/TabSurvey (original authors: Kathrin Sessler, Tobias Leemann and Vadim Borisov). License retained in `licenses/TabSurvey-MIT.txt`.
- SAINT: https://github.com/somepago/saint. Bundled feature/row attention, tokenizer and prediction-head components under Apache 2.0, with the full upstream license in `licenses/SAINT-Apache-2.0.txt`. The bundled source is trimmed for survival use; adaptation notices are included in its files.
- NODE: https://github.com/Qwicen/node. Bundled differentiable tree, sparse activation and dense-block components under MIT; full upstream license in `licenses/NODE.txt`. Utilities are reduced to the tensor conversion used by initialization, and the sparse activation wrapper is named for serialization.
- TabNet: https://github.com/dreamquark-ai/tabnet. Installed as an optional dependency; its native network feeds a survival objective.
- XGBoost: https://github.com/dmlc/xgboost. Optional dependency using `survival:cox`.
- TabPFN and extensions: https://github.com/PriorLabs/TabPFN and https://github.com/PriorLabs/tabpfn-extensions. Optional dependencies; upstream code and weights are not bundled.

The package adds a shared censored-survival estimator, preprocessing, Breslow probabilities, CLI and tests. The FT-Transformer label refers to the study's column-attention variant, not the original authors' reference implementation. The TabPFN adapter follows the study's embedding-plus-MLP approach.
