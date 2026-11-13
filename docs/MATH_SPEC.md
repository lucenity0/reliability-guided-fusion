# Mathematical specification: Reliability-Guided Dynamic Expert Fusion

This is the mathematical specification. Items tagged **[A-n]** are derivations, collected in `docs/derivations/`. Every default value listed here is a starting point for the validation-only sweep, not a final choice.

## Notation

| Symbol | Meaning |
|---|---|
| $(I, T, y)$ | image, text, label $y\in\{0,\ldots,C-1\}$; binary shorthand uses $C=2$ |
| $d$ | CLIP joint embedding dim (768 for ViT-L/14) |
| $h$ | fusion hidden dim (default 512) |
| $K$ | number of experts (3: img, txt, int) |
| $\sigma$ | logistic sigmoid; $\mathrm{BCE}(z,y) = -y\log\sigma(z)-(1-y)\log(1-\sigma(z))$ |
| $\mathrm{sg}(\cdot)$ | stop-gradient |
| $\Delta^{K-1}$ | probability simplex |

## 1. Frozen features (computed once, cached)

Backbone: `openai/clip-vit-large-patch14-336`, frozen, eval mode, fp16 storage.

- **Image tokens:** take the last vision hidden state, then apply `post_layernorm` and `visual_projection` to **every** token. That gives the CLS token plus a 24×24 patch grid. Apply 2×2 average pooling to the grid to get 144 patch tokens.
  $V=[\bar v; v_1;\dots;v_{144}]\in\mathbb{R}^{145\times d}$. Here $\bar v$ is the CLS token, which is exactly CLIP's image embedding.
- **Text tokens:** take the last text hidden state, then apply `final_layer_norm` and `text_projection` to every token.
  $U\in\mathbb{R}^{L\times d}$ with mask $m\in\{0,1\}^L$, $L\le 77$. The pooled text vector $\bar u$ is the EOS token, which is CLIP's text embedding.
- **Similarity:** $s=\cos(\bar v,\bar u)$ on the raw, unnormalised-by-temperature embeddings.
- **Storage:** about 145·768·2 B ≈ 223 KB per image. That's about 2.7 GB for FHM and roughly 11 GB across all four datasets. Full-resolution 577-token caches are an ablation only (run on Colab, #30).

**[A-1]** Projecting non-CLS tokens through `visual_projection` is an approximation, because the projection was trained only on CLS and EOS. Justify it briefly in the paper as a design choice with prior use (e.g. MaskCLIP-style dense features, *verify citation*). Alternatively, measure the effect against using unprojected 1024-d tokens with a learned projection.

## 2. Experts

For C-class tasks ($C\ge2$) let $z_k=W_k h_k+b_k\in\mathbb R^C$ and $p_k=\operatorname{softmax}(z_k)$. The C-row heads below leave hidden features unchanged. Binary tasks may retain their single-logit BCE representation. Define $\operatorname{CE}(z,y)=\operatorname{LSE}_c(z_c)-z_y$. See A-7, Claims A7.1–15 for numbered derivations.

**D-017 approved shapes (complexity integration approved in D-019).** Expert MLPs: img/txt $2h\to h\to h$; interaction $(2h+n^2)\to h\to h$. Use GELU with dropout between affine layers. Router pooled maps are $d\to h$ each and router MLP is $(2h+K+1)\to h\to K$. The complete Variant F logit map is $h\to h\to q$, where $q=C$ for categorical output or $q=1$ for binary shorthand; this includes its final classifier, so do not append another output head.

**Input projections:** $\phi_v(x)=\mathrm{Drop}(\mathrm{LN}(W_v x))$ and $\phi_t(x)=\mathrm{Drop}(\mathrm{LN}(W_t x))$, with $W_\cdot\in\mathbb{R}^{h\times d}$.

**Masked attention pooling** with a learned query $q$:
$$\mathrm{AP}_q(X,m)=\sum_j \operatorname{softmax}_j\!\Big(\tfrac{q^\top W_k x_j}{\sqrt h}+\log m_j\Big)\,W_o x_j .$$

**D-017 pooling execution.** Reassociate the same pool by computing $u=W_k^\top q$, scores $u^\top x_j/\sqrt h$, then $W_o(\sum_j a_jx_j)$. This preserves the equation and costs $4h^2+4Th$ core FLOPs per T-token pool (literal tokenwise maps cost $4Th^2+4Th$).

**$E_{img}$:**
$$h_{img}=\mathrm{MLP}_{img}\big([\mathrm{AP}_{q_v}(\phi_v(V));\ \phi_v(\bar v)]\big),\qquad z_{img}=W_{img}h_{img}+b_{img}.$$

**$E_{txt}$:** the same as $E_{img}$, using $U$, $m$ and $\bar u$.

**$E_{int}$:**

1. **Cross-attention.** Use $N_x=2$ pre-LN layers with simultaneous updates. Each direction attends to the *previous-layer* states of the other modality. Start from $X_v^{(0)}=\phi_v(V)$ and $X_t^{(0)}=\phi_t(U)$.
$$\tilde X_v=X_v+\mathrm{MHA}(\mathrm{LN}X_v,\mathrm{LN}X_t,\mathrm{LN}X_t;\ m),\quad \tilde X_t=X_t+\mathrm{MHA}(\mathrm{LN}X_t,\mathrm{LN}X_v,\mathrm{LN}X_v)$$
$$X_v\leftarrow\tilde X_v+\mathrm{FFN}_v(\mathrm{LN}\tilde X_v),\qquad X_t\leftarrow\tilde X_t+\mathrm{FFN}_t(\mathrm{LN}\tilde X_t)$$
Defaults: 8 heads, FFN width $2h$, dropout 0.1–0.3 (sweep).
2. **Cross-fusion** (Hate-CLIPper-style). Project the pooled vectors with $p_v=P_v\bar v$ and $p_t=P_t\bar u\in\mathbb{R}^n$. Form $F=p_vp_t^\top$ and flatten it to $c_{cf}=\mathrm{vec}(F)\in\mathbb{R}^{n^2}$. Default $n=32$ *(check the Hate-CLIPper paper's choice)*.
3. **Combine:**
$$h_{int}=\mathrm{MLP}_{int}\big([\mathrm{AP}(X_v^{(N_x)});\ \mathrm{AP}(X_t^{(N_x)},m);\ c_{cf}]\big),\qquad z_{int}=W_{int}h_{int}+b_{int}.$$

## 3. Router

**Expert uncertainties.** Let $p_k=\operatorname{softmax}(z_k)$. The normalized categorical entropy is
$$\hat H_k=-\sum_{c=0}^{C-1}p_{kc}\log p_{kc}/\log C\in[0,1].$$
The binary shorthand is its exact C=2 reduction.

**Router input:**
$$r=\big[\phi^r_v(\bar v);\ \phi^r_t(\bar u);\ \mathrm{sg}(\hat H_{img}),\mathrm{sg}(\hat H_{txt}),\mathrm{sg}(\hat H_{int});\ s\big].$$
The default is stop-gradient on $\hat H_k$, so the router can't learn to manipulate the experts' confidence. Removing the stop-gradient is an ablation.

**Router output:**
$$g=\mathrm{MLP}_r(r)\in\mathbb{R}^K,\qquad \pi=\operatorname{softmax}(g)\in\Delta^{K-1}.$$

## 4. Fusion: decision D1 (choose on val, report all three)

| Variant | Prediction | Notes |
|---|---|---|
| **L** (logit mixture) | $z=\sum_k\pi_k z_k\in\mathbb R^C$, $\hat p=\operatorname{softmax}(z)$ | Normalized geometric probability pool; §7. |
| **P** (probability mixture) | $\hat p_c=\sum_k\pi_k\operatorname{softmax}(z_k)_c$ | Exact categorical mixture; τ=1 target equals responsibility only under a uniform expert prior (§5). |
| **F** (feature mixture) | $\hat p=\operatorname{softmax}\big(W_f\mathrm{MLP}_f(\sum_k\pi_k h_k)+b_f\big)$ | Feature mixing unchanged; C-row output head. π interpolates representations, not decision shares. |

**Selection rule:** choose the variant with the highest mean val AUROC over 3 seeds on FHM, then use it for every dataset.
**[A-4]** State which interpretability claims hold under each variant.

## 5. Reliability-guided routing target

Per-sample expert losses are detached: $\ell_k=\mathrm{sg}\big(\operatorname{CE}(z_k,y)\big)$; use BCE in the binary single-logit representation.

$$t_k=\frac{\exp(-\ell_k/\tau)}{\sum_j\exp(-\ell_j/\tau)},\qquad \mathcal{L}_{rel}=\mathrm{KL}(t\,\|\,\pi)=\sum_k t_k\log\frac{t_k}{\pi_k}.$$

For any class label, $\exp(-\ell_k)=p_{ky}=P_k(y\mid x)$. For C=2 this equals the Bernoulli likelihood. So
$$t_k\propto P_k(y\mid x)^{1/\tau}.$$

**[A-2] Prove these properties and put them in the paper:**

- **P1 (limits; the "unified" claim).**
  - As $\tau\to\infty$, $t\to$ uniform, so when $\mathcal{L}_{rel}$ dominates, π tends to $1/K$: **static equal-weight fusion**.
  - As $\tau\to0$, $t$ is uniform over the loss minimisers, one-hot only if the minimiser is unique (D-011).
  - These are target endpoints. For free reliability-only population routing, $\pi^*(x)=E[t\mid x]$: hard routing requires the same unique winner almost surely given x. Static routing additionally requires reliability-only optimization or reliability dominance with fixed experts and an attainable optimum; finite-weight joint training has no endpoint guarantee (A7.6–8).
- **P2 (population minimiser).** For fixed $x$,
$$\arg\min_{\pi\in\Delta}\ \mathbb{E}_{y\mid x}\,\mathrm{KL}(t(x,y)\|\pi)=\mathbb{E}_{y\mid x}[t(x,y)].$$
Hint: KL = cross-entropy − H(t), and H(t) doesn't depend on π.
- **P3 (EM link).**
  - At $\tau=1$ under Variant P with a uniform prior over experts, $t_k=P(k\mid x,y)$. That is the E-step responsibility of a mixture of experts (Jacobs et al. 1991; Jordan & Jacobs 1994).
  - With the router's own prior, the true responsibility is $\propto\pi_kP_k(y\mid x)$. Discuss uniform vs router prior, and why the uniform prior avoids rich-get-richer collapse.

**[A-3] Novelty positioning.** Write a precise comparison against:

- MoE EM training;
- learning-to-defer and selective routing;
- **QMF** (Zhang et al., ICML 2023), which weights modalities by energy-based uncertainty with a generalisation bound (PMLR 202:41753–41769);
- **PDF** (Cao et al., ICML 2024), which predicts fusion weights from co-belief and relative calibration (PMLR 235:5608–5628). Section-level comparison: `docs/derivations/A-3.md`.

The claim we can defend is the **interaction expert inside the routing set**, plus the τ-family that unifies static and hard routing, evaluated on adversarial memes. Spell out what is and isn't new.

## 6. Training objective

$$\mathcal{L}=-\log\hat p_y+\lambda_{aux}\frac1K\sum_k\operatorname{CE}(z_k,y)+\lambda_{rel}\,\mathrm{KL}(t\|\pi)$$
Use the exact binary BCE reduction when C=2.

- For categorical P use $-\operatorname{LSE}_k[\operatorname{logsoftmax}(g)_k+\operatorname{logsoftmax}(z_k)_y]$. Do not clamp categorical components independently. For the binary single-logit path retain $-\log[y\hat p+(1-y)(1-\hat p)]$ after clamping $\hat p$ to $[10^{-6},1-10^{-6}]$; derivatives apply in its unclamped interior (A7.13).
- Use AdamW with decoupled weight decay (default 0.01–0.05), a cosine schedule with 5–10% warmup, and early stopping on val accuracy for MVSA-Single/Food-101, retaining val AUROC for binary datasets, with patience 5. The shared D1 variant is still selected by the existing FHM val rule in §4.
- Defaults: λ_aux=0.5, λ_rel=0.1, τ=1.
- Sweep ranges: λ_aux ∈ [0.1, 1], λ_rel ∈ [0, 1], τ ∈ {0.25, 0.5, 1, 2, 4}, lr ∈ [1e-5, 5e-4] (log), dropout ∈ [0.1, 0.5], h ∈ {256, 512}.
- **Deep supervision (λ_aux > 0) is required.** Without it the experts are not individually trained predictors, so using their losses as reliability supervision is not justified. Auxiliary CE makes those losses meaningful training targets, but does not guarantee calibration (A7.5).
- **No** MixUp, R-Drop, SupCon, curriculum, label smoothing or α-entropy/diversity terms by default. Any of them may be added only with an ablation row showing a val gain.
- **Optional regularisers** (the sweep decides): modality dropout, i.e. zero out $V$ or $U$ with p ≈ 0.1, and token dropout.

## 7. Gradient analysis: why the gate gets signal (**[A-5]**, for the paper's motivation)

**Categorical router gradients (A7.10–12).** Hold expert outputs fixed, set
$z=\sum_k\pi_kz_k$, $p_L=\operatorname{softmax}(z)$, $p_k=\operatorname{softmax}(z_k)$,
$S=\sum_k\pi_kp_{ky}$ and $r_k=\pi_kp_{ky}/S$. Then
$$\partial_{g_j}L_L=\pi_j(p_L-e_y)^\top(z_j-z),\qquad
\partial_{g_j}L_P=\pi_j-r_j,\qquad \partial_{g_j}L_{rel}=\pi_j-t_j.$$
The total direct router gradient is the main gradient plus $\lambda_{rel}(\pi-t)$.
For finite logits, the whole L gradient vanishes iff every expert difference is
orthogonal to $p_L-e_y$ (constant class-logit shifts are irrelevant); the whole P gradient vanishes iff all observed-class
probabilities $p_{ky}$ agree. Neither requires full-distribution agreement for
C>2. C=2 reduces exactly to A-5's scalar formulas using $z_{k1}-z_{k0}$.
The auxiliary term has no direct router-logit derivative; shared parameters
require the full chain rule. See A7.12 for F's Jacobian expression.

**V3 comparison (D-010).** A-5's constant-gate scale absorption and binary
modality-gate entropy/variance analysis remain unchanged for any label count.
They are mechanisms consistent with collapse, not a causal proof for a run.

## 8. Complexity (**[A-6]**, answers R3's request for a cost analysis)

Under D-019, use A-6's two-FLOPs-per-MAC convention; report dense contraction FLOPs separately from elementwise overhead. At P=145, executed L≤77, hidden width h, H heads dividing h, FFN width f and n_layers:

- Each bidirectional layer: projections $8(P+L)h^2$, scores **and** values $8PLh$, FFNs $4(P+L)hf$. Multiply the sum by n_layers.
- Shared input projections: $2(P+L)dh$. Cross-fusion: $4dn+n^2$.
- Four attention pools, approved reassociation: $16h^2+8(P+L)h$. Literal reference: $8(P+L)h^2+8(P+L)h$.
- Expert/router MLPs: sum $2ab$ over every a→b affine layer using D-017. Router input width is $2h+K+1$, plus two d→h pooled projections.
- Expert output heads: $2Khq$ (q=1 for binary shorthand; q=C for categorical). L/P mixing adds $2Kq$. F instead adds $2Kh+2h^2+2hq$ with its complete h→h→q head; retain the auxiliary expert heads.
- Frozen CLIP image/text forward and dense-token cache overhead are reported separately. The frozen vision encoder uses 577 tokens before pooling to 145; it must not be costed at P=145.
- A-6 itemizes biases, normalizations, activations, softmax, masking and residuals outside these core counts. Profiler agreement within 10% remains pending #31.

Then measure: parameters, latency in ms per sample at batch sizes 1 and 64 on the available GPU, peak memory, and training wall-clock per seed. Compare against a 7B LMM's per-sample latency.

## 9. Calibration (optional, reported only if done)

Report ECE with 15 equal-width bins, before and after temperature scaling fitted on val. **This is not fairness.** The V3 paper's fairness claims are dropped unless they're implemented and evaluated.

## 10. Statistical testing

Let $S$ be the number of seeds: 5 for every method, **10 for the headline dynamic-vs-static pair**. That pair shares seeds and initialisation of the expert weights where possible.

1. **Primary test:** a paired t-test across seeds on test accuracy for MVSA-Single/Food-101 and test AUROC for the binary datasets, dynamic vs static. Report the mean Δ, the 95% CI of Δ and p.
   - Note: a two-sided Wilcoxon signed-rank test with $n=5$ has minimum p = 0.0625, so it **cannot** reach 0.05. That's why the primary test is a t-test with $S=10$. Report Wilcoxon only as a secondary test, at S=10.
2. **Per-seed tests on test predictions**, reported as the median p and the fraction of seeds with p < 0.05:
   - **DeLong (binary datasets only)** test for two correlated AUROCs (DeLong et al. 1988), using the fast algorithm (Sun & Xu 2014);
   - **paired bootstrap** with B = 10,000: resample test indices, apply the same indices to both models, take a percentile 95% CI of Δaccuracy for MVSA-Single/Food-101 (ΔAUROC for binary datasets), and compute $p=\min\{1,2\min(\Pr(\Delta^*\le0),\Pr(\Delta^*\ge0))\}$;
   - **McNemar's test** on per-sample correctness: categorical predictions are argmax with smallest-index tie resolution and no threshold; binary models use their own val-chosen thresholds. Use the exact binomial version if there are fewer than 25 discordant pairs, otherwise χ² with continuity correction. Zero discordance gives p=1.
3. **Multiple comparisons:** apply Holm–Bonferroni over the family of datasets (FHM, MAMI, Harm-C, Harm-P, CrisisMMD, MVSA-Single, Food-101) for the headline comparison. Do not pool samples and seeds as independent observations.
4. **Sanity tests (unit tests):**
   - identical predictors give p ≈ 1;
   - a clearly better predictor on synthetic data gives p < 0.01;
   - On binary datasets, DeLong agrees with the AUROC bootstrap's sign and rough magnitude.
   - On multiclass synthetic predictions, test paired accuracy bootstrap and McNemar correctness counts. No DeLong call is made.

## 11. Deliverables from the math owner

- `docs/derivations/A-1.md` … `A-6.md`, written in LaTeX-in-Markdown.
- A finalised version of this file, with defaults updated after the sweep and every *(verify)* resolved.
- Method-section equations ready to paste into the paper (IEEE format, numbered).
