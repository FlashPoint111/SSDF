# SSDF method notes

## Inputs and independent encoders

The facial input is a sequence of RGB target-face frames. A frozen CLIP-ViT-B/16 visual backbone supplies patch tokens and one pooled token per frame. LoRA adapters update attention and feed-forward projections while the original CLIP weights remain frozen. Their configuration is supplied when the model is constructed.

Four learned queries read the patch tokens through four attention heads of width 64. The readout is added to the pooled frame token. A one-layer temporal Transformer with a learned sequence token aggregates the 16 frame representations into a 768-dimensional visual vector $v$.

The contextual input is a fixed, normalized 2048-dimensional embedding $g$ of ordered full-scene observations and a reference face. The [`FrozenQwenContext`](ssdf/context.py) code constructs the multimodal request with an externally supplied instruction and task text, passes the prepared observations and their source timestamps to Qwen3-VL-Embedding-2B, and reads the last valid token from a frozen forward pass. The vector is L2-normalized and computed before SSDF training. A trainable adapter maps it to a 256-dimensional contextual vector $c$. The facial forward path takes no contextual input before decision fusion.

## Source-supervised decision fusion

Separate diagonal-Gaussian encoders map $v$ and $c$ to 256-dimensional latent vectors $z_v$ and $z_c$. Their concatenation enters a third Gaussian encoder, producing $z_j$. Each Gaussian encoder predicts a mean and a positive scale; training samples with reparameterization, while inference uses posterior means. The visual, contextual, and joint latents each have an expression classifier. The joint classifier supplies the final decision.

For the three heads $s\in\{v,c,j\}$, the objective combines direct expression supervision with Gaussian regularization:

$$
\mathcal L=\sum_s\lambda_s\operatorname{CE}_s+\sum_s\gamma_s\operatorname{KL}(q_s\|\mathcal N(0,I)).
$$

The source classifiers provide direct supervision as the joint classifier learns their combination. Each head returns classifier scores and a KL term; the objective coefficients are supplied by the training procedure.

At inference, each Gaussian encoder uses its posterior mean, and the **joint classifier** supplies the expression scores.

The [`ssdf/data.py`](ssdf/data.py) loader pairs a selected face sequence with its cached $g$ vector. The model definition in [`ssdf/model.py`](ssdf/model.py) contains the fusion structure, and the Qwen module shows how $g$ is produced from prepared observations.
