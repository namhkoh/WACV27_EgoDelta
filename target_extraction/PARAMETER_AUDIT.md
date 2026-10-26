# Stage-3 parameter audit

Frozen paper revision: `a15262713a00e48b2e7aca2c631f1dd8cac832c0`.

## Counting policy

`Tuned` is the headline re-tuning liability: a value was selected on this
benchmark and could plausibly change on a new crop/testbed. `Derived` is
computed from each scene at run time. `Domain` is fixed by the evaluation or
intervention mechanics rather than fit to these data. `Structural` is an
algorithmic convention, categorical choice, guard, or numerical truncation.
Structural values are disclosed but excluded from the headline count.

The compact counts used in the paper are **3 tuned values for deleafing** and
**6 tuned values for harvesting**. Those counts treat `delta > 0.5` as the
shared Stage-2 mask convention and list the unchanged DBSCAN proposal settings
separately as inherited. A literal Stage-3 call-graph count is therefore larger:
deleafing uses three method-specific fitted values plus the shared threshold and
two inherited DBSCAN settings; harvesting uses six method-specific/shared
fitted values plus the two inherited DBSCAN settings. This distinction is made
explicit rather than hiding prediction-affecting inherited values.

## Final mechanics-conditioned pipelines

| Pipeline | Constant or rule | Value | Class | Sensitivity measured? | Evidence / role |
|---|---|---:|---|---|---|
| Both | change/grouping threshold `tau_c` | 0.5 | Tuned (shared Stage-2 convention in compact count) | Yes | Swept over 0.2/0.3/0.4/0.5; coupled and decomposed diagnostics. |
| Both | DBSCAN radius | 0.02 m | Tuned, inherited | Partial | Radius/connectivity diagnostics included 0.02 m; final all-scene DBSCAN radius was not independently re-selected. |
| Both | DBSCAN `min_samples` | 20 | Tuned, inherited | Partial | Compared with connected-component post-filters and true DBSCAN; no final all-scene sweep. |
| Both | scale multiplier | 2.0 | Tuned | Yes | `sigma = 2 median_NN`; multiplier sweep 1/1.5/2/3/5 and diagnostic decomposition. |
| Both | scene scale statistic | median non-self nearest-neighbor distance | Derived | N/A | Converts reconstruction sampling scale to `sigma`. |
| Retained | normalized removed-mass gate | 0.0037746118219953335 | Tuned | Yes | Full all-30 PR sweeps for absolute, scene-delta, rank, and overlap-normalized mass; selected in-sample. |
| Retained | global peak NMS radius | 0.03 m | Domain | No | Same spatial resolution as the 3 cm one-to-one evaluation/tool tolerance. |
| Retained | contact-field neighbor cutoff | `3 sigma` | Structural | No | Numerical truncation of the Gaussian kernel. |
| Retained | Gaussian denominator | `2 sigma^2` | Structural | No | Standard Gaussian definition. |
| Retained | normal compatibility | `abs(n_i dot n_j)` | Structural | Yes | No-normal ablation measured; not a scalar threshold. |
| Retained | local peak/refinement radius | `sigma` | Derived | Yes through sigma sweep | Same scene-derived scale as the contact kernel. |
| Retained | boundary-score proximity | `exp[-0.5(d/sigma)^2]` | Structural form selected in-sample | Yes | Ablated as one-sidedness only, distance only, and product. The form itself was selected after benchmark inspection. |
| Retained | boundary one-sidedness | delta-weighted unit-vector resultant | Structural form selected in-sample | Yes | Same selector ablation; no additional numeric constant. |
| Removed | component mean-delta gate | 0.9008104936197888 | Tuned | Yes | All-30 component-level sweep; selected in-sample. |
| Removed | component primitive-count gate | 205 | Tuned | Yes | All-30 component-level sweep; selected in-sample. |
| Removed | dense-mode `min_cluster_size` | 120 | Tuned | Yes | Sweep 20/40/80/120/160/200; useful band 80--160. |
| Removed | density upper-fence multiplier | 1.5 | Domain convention; use selected in-sample | No direct multiplier sweep | Tukey `Q3 + 1.5 IQR`; the scene fence is derived, but adoption of this rule followed benchmark inspection. |
| Removed | scene density quartiles | `Q1`, `Q3` at radius `sigma` | Derived | N/A | Recomputed per scene; defines the Tukey fence. |
| Removed | HDBSCAN `min_samples` | 20 (shared DBSCAN floor) | Tuned, inherited | No independent sweep | Reuses the inherited density floor rather than adding another value. |
| Removed | HDBSCAN selection | `leaf`, `allow_single_cluster=True` | Structural | No | Dense-mode enumeration convention. |
| Removed | partition completion | nearest dense-core primitive | Structural | No | Assigns all parent-component primitives to one mode. |
| Removed | gravity direction | world `+Z` | Domain | Compared, not swept | Physical prior for hanging payloads; retained-facing alternative reduced performance. |
| Removed | central surface radius | `sigma` in horizontal plane | Derived | Yes through sigma sweep | Selects the gravity-facing surface peak near each mode center. |
| Removed | refinement support | horizontal radius `sigma`, down to `z_peak-sigma` | Derived/Structural | Yes through sigma sweep | Local delta-weighted surface refinement. |
| Evaluation | one-to-one matching radius | 0.03 m | Domain | Threshold table exists | Benchmark/tool resolution; not selected by Stage 3. |

Thus the final method has no color threshold, PCA endpoint, crop label, hand
distance, shape descriptor, or consensus-vote constant. The endpoint-side
router is a categorical mechanics convention, not a fitted numeric value.

## Legacy extractor: archived Table-6/headline set

The repository does not contain the manuscript source or literal Table 6. It
does preserve the release statement “approximately 20 Table-6 parameters,” the
exact evaluation configuration, and the full call graph. The following 20-row
headline set is therefore a **reconstruction from the archived release config**,
not a claim that the unavailable manuscript table used exactly these row names.
Every value is classified `Tuned`; no systematic sensitivity sweep is preserved
for the final legacy call graph.

| Legacy path | Headline parameter | Value | Class | Sensitivity measured? |
|---|---|---:|---|---|
| Shared | removed-score threshold | 0.5 | Tuned | No |
| Shared | change-cluster DBSCAN radius / min samples | 0.02 m / 20 | Tuned | No |
| Shared | proposal-component radius / min points | 0.03 m / 8 | Tuned | No |
| Shared | support dilation / minimum support | 0.03 m / 8 | Tuned | No |
| Shared | proposal deduplication radius | 0.025 m | Tuned | No |
| Leaf | primary score threshold | 1.26 | Tuned | No |
| Leaf | primary support threshold | 80 | Tuned | No |
| Leaf | primary axis-length threshold | 0.12 m | Tuned | No |
| Leaf | primary contact-count threshold | 24 | Tuned | No |
| Leaf | primary green-contact threshold | 0.18 | Tuned | No |
| Leaf | endpoint fraction | 0.15 | Tuned | No |
| Leaf | maximum stem distance | 0.08 m | Tuned | No |
| Leaf | contact radius / points | 0.03 m / 8 | Tuned | No |
| Leaf | clustered-support ratio | 0.8 | Tuned | No |
| Tomato | fruit color gates `(G-R, G/sumRGB)` | 0.0352147 / 0.336774 | Tuned | No |
| Tomato | peak score / support threshold | 0.9 / 80 | Tuned | No |
| Tomato | peak NMS / support radius | 0.08 m / 0.02 m | Tuned | No |
| Tomato | peak snap / max-snap radius | 0.075 m / 0.065 m | Tuned | No |
| Tomato | snap pair / support radius | 0.045 m / 0.025 m | Tuned | No |
| Tomato | primary maximum count | 2 | Tuned | No |

## Legacy constants absent from the reconstructed headline set

The actual active `proposal.mode: contact` call graph uses substantially more
prediction-affecting constants than the approximately-20 headline count. Values
below come from `old/code/configs/objectness_full.yaml` or, where marked
“hardcoded,” directly from `old/code/egodelta_diff/contact_proposal.py`.
Disabled `baseline_nms` values, rendering settings, evaluators, serialization
constants, array indices, and numerical epsilons are excluded.

| Legacy subpath | Additional prediction-affecting constants | Class | Sweep? |
|---|---|---|---|
| Shared proposal | maximum proposals 64; static-green gates 0.30 and -0.05 | Tuned | No |
| Leaf component handling | require clustered component; support-overlap dedup 0.80; cluster-overlap dedup 0.80; maximum support ratio 0.75 | Tuned | No |
| Leaf proposal validity | minimum axis 0.015 m; minimum green strength 0.02; core-snap distance 0.015 m; static-joint distance 0.015 m; static-joint mean score 0.80 | Tuned | No |
| Leaf endpoint ranking | configured weights 0.020, 0.45, 2.0 | Tuned | No |
| Leaf proposal score (hardcoded) | contact weights 0.04 and 0.35; stem-distance weight 0.5 | Tuned | No |
| Leaf green score (hardcoded) | `(G-R)` contribution multiplier 0.5 | Tuned | No |
| Static-green fallback (hardcoded) | require `max(10, min(50, N_static/20))` green candidates | Structural/Tuned fallback | No |
| Tomato support validity | green gates 0.35 and 0.02; lobe minimum 8; contact radius 0.06 m; maximum static-stem distance 0.10 m; require clustered support | Tuned | No |
| Tomato peak enumeration | seed threshold 0.5; max peaks/support 4 | Tuned | No |
| Tomato snap color | strong-green gates 0.40 and 0.08 | Tuned | No |
| Tomato secondary recovery | enabled; seed 0.82; support 45; separation 0.055 m; snap-density 0.10 | Tuned | No |
| Tomato proposal rank (hardcoded) | support cap 500; density bonus 0.08; secondary penalty 0.08 | Tuned | No |
| Tomato snap density (hardcoded) | green/fruit count caps 25/60 | Tuned | No |
| Tomato snap candidate score (hardcoded) | weights 1.20, 0.25, 0.35, 0.35, 0.15, 0.05, -0.18 | Tuned | No |
| Tomato snap fallback (hardcoded) | fruit-distance 0.20; green bonus -0.015; removal-score bonus -0.010; midpoint 0.5 | Tuned/Structural | No |
| Primary selection | primary tomato score 0; peak-only required; primary leaf clustered support required | Tuned/Structural | No |

The code audit therefore supports “approximately 20 headline parameters,” but
also shows that the literal legacy re-tuning liability is larger. Reporting
only the YAML’s most visible values would undercount hardcoded score weights and
fallback gates.
