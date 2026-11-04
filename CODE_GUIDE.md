# SSDF code map

This page maps the main SSDF components to their implementations.

| Component | Code | What it does |
| --- | --- | --- |
| Context encoding | [`FrozenQwenContext.encode`](ssdf/context.py) | Passes ordered observations and a reference face through frozen Qwen3-VL-Embedding-2B, then L2-normalizes the last valid token. |
| Facial encoding | [`SSDF.forward`](ssdf/model.py) | Reads CLIP patch and pooled tokens with learned spatial queries and a temporal encoder. |
| Decision fusion | [`SSDF.forward`](ssdf/model.py), [`SourceSupervisedHead`](ssdf/model.py) | Combines the visual representation and adapted contextual vector in a joint Gaussian head. |
| Source supervision | [`SourceSupervisedHead`](ssdf/model.py), [method notes](METHOD.md) | Returns visual, contextual, and joint classifier scores and Gaussian regularization terms. |
| Prepared input | [`PreparedVideoDataset`](ssdf/data.py), [`collate_videos`](ssdf/data.py) | Loads selected face frames and the paired cached context vector into a batch. |

The Qwen encoder and original CLIP weights are frozen. The facial reader, contextual adapter, and fusion heads are trainable. The visual reader does not receive the contextual vector before the decision head. The joint classifier supplies the final expression scores.

The paper contains the experimental protocol and results. This repository contains the model and prepared-input components.
