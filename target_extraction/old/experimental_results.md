# Contact-mass target extraction — complete experimental record

This document preserves the Phase-1 checkpoint, the frozen full-scale results,
and every cache-only diagnostic used to reach the final method decision.  The
second frozen estimator applies
`w_rem = delta`, `w_ret = omega - delta`, deletes the component-mass
percentile, replaces per-component centroids with locally refined contact
peaks, and exposes the interface pair `(g_ret, g_rem)` for downstream
tool-side selection.  The final all-30 deleafing gate/selector analysis and the
closed negative-result ledger appear immediately before the artifact index.

Stage 3 consumes no appearance features; appearance enters only through the
upstream change score.

## Second frozen estimator

The estimator groups the thresholded change field with the baseline DBSCAN
settings (`eps = 0.02 m`, `min_samples = 20`, `tau_c = 0.5`).  For each group,
the same pairwise contact contributions produce fields on retained and removed
primitives.  Local maxima within `sigma = 2 * median_NN` identify modes; each
mode is refined by a local rho-weighted centroid within `sigma`; and 3 cm NMS
is applied independently on each side.  The 3 cm radius is the evaluator/tool
resolution, not a crop-specific fitted value.  Pairing a surviving endpoint
to the nearest opposite-side mode in its DBSCAN component yields
`(g_ret, g_rem)`.  The estimator contains no task, crop, or appearance branch.

The component-mass percentile is deleted rather than reclassified.  The
confidence partition has the direct interpretation
`omega_i = w_rem_i + w_ret_i`.

## Frozen Phase-1 record

The initially specified connected-component estimator emitted 240,233 targets
over 60 scenes and obtained overall F1 0.0008.  This is preserved as a negative
result.  The smoke-scene F1 0.0020107238605898124 is also preserved exactly.

Two specification/interface problems were identified after freezing:

1. The cached `delta` already equals `omega * raw_change`.  Applying `omega`
   again double-weighted removal and retention.  The corrected partition is
   `w_rem = delta`, `w_ret = omega - delta`, so that
   `omega = w_rem + w_ret`.
2. Radius-connected components at `sigma` fragment the thresholded change
   field, while unconditionally emitting every fragment omits the rejection
   role of the legacy selector.

The weight correction changes the smoke-grid cell-median predictions by only
0.27 mm, but it remains semantically necessary.  It does not change grouping.

## DBSCAN grouping equivalence

On `egodelta_040bc88b3760_leaf`, true
`sklearn.cluster.DBSCAN(eps=0.02, min_samples=20)` at `tau_c=0.5` forms 28 raw
clusters.  The Stage-2 cache contains 27 nonnegative legacy cluster labels.
Membership comparison shows:

- 27 of 28 raw clusters map one-to-one to all 27 stored legacy clusters.
- All 27 pairs have exact primitive membership, not merely centroid overlap.
- Raw cluster 19 is the only unmatched cluster; it is removed by the legacy
  volume/axis filtering surrounding DBSCAN.
- Exact mapped primitive overlap is 11,188.
- The four selected legacy leaf proposals have representative primitives in
  raw clusters 0, 1, 2, and 3 respectively.  Every legacy support-cluster ID
  used by those proposals maps to a raw cluster formed in this diagnostic.

Thus, for this deleafing scene, instance formation is standard DBSCAN and is
effectively identical to the baseline.  The differentiating stage is the
selector: legacy reduces the 27 retained clusters to four targets using its
crop-specific endpoint and consensus machinery, whereas the first
contact-mass implementation emitted every component.

The cross-task qualification matters.  On the paired harvesting scene,
DBSCAN forms 23 raw clusters and the cache stores 20 after filtering; all 20
stored clusters again have exact raw membership.  However, both fruit GTs map
to the same raw cluster 0, and both legacy tomato proposals also use stored
cluster 0.  The tomato selector therefore performs within-component multi-peak
extraction as well as rejection.  A one-component/one-target estimator cannot
recover both fruits on this scene, regardless of its ranking score.

## DBSCAN-granularity score audit

True attachment clusters are labeled independently of every score: for each GT,
take the closest DBSCAN component in removed-primitive space and require its
distance to be within the estimator's structural `3 * sigma` neighborhood.
`d_gt` in the cluster table separately measures the contact-mass centroid to
the nearest GT.  Clusters with zero contact mass remain in the full audit but
abstain before ranking because they have no target to emit.

S1–S4 and S6 rank high-to-low.  S5 ranks low-to-high, following its physical
claim that a real removed module contacts retained structure over only a small
fraction of its extent.

### Deleafing, `tau_c=0.5`, 28 raw clusters

All four GTs have adjacent, distinct clusters: 0, 1, 2, and 3.

| Score | GT-cluster ranks | Precision at component recall 1.0 | Clean margin |
|---|---|---:|---|
| S1 total contact mass | 3, 6, 2, 5 | 0.6667 | no |
| S2 peak rho | 3, 7, 2, 6 | 0.5714 | no |
| S3 concentration | 13, 6, 12, 5 | 0.3077 | no |
| S4 contact mass / ext80 | 1, 4, 8, 3 | 0.5000 | no |
| S5 contact fraction, ascending | 5, 3, 7, 4 | 0.5714 | no |
| S6 component size | 2, 3, 1, 4 | 1.0000 | yes |

S3's poor ranks are evidence of the representation failure rather than evidence
against contact geometry.  S3 measures mass concentration around one component
centroid; with separated contact modes, that centroid lies between modes and
the measured concentration is low by construction.

S6 has a strict margin on this scene: the smallest true component contains
1,427 primitives and the largest false component contains 353.  It is retained
as the requested trivial baseline, not selected as the method: it has no
contact-geometry justification and the harvesting scene does not preserve the
one-cluster/one-instance assumption.

Although all four true components are present, the contact centroid of cluster
2 is 19.93 cm from the nearest GT.  Consequently, even the oracle set of four
true DBSCAN clusters cannot obtain target recall 1.0 with one centroid per
cluster; the unthresholded result is TP/FP/FN = 3/12/1.  This is the same
large-component merging/localization failure exposed by D1.

At `tau_c=0.3`, the scene has 35 raw clusters.  The four true clusters remain
0, 1, 2, and 3.  S1 reaches component recall 1.0 at rank 6; S5 also reaches it
at rank 6; S6 remains the only clean separator with ranks 2, 3, 1, 4.

### Harvesting control scene

At `tau_c=0.5`, DBSCAN forms 23 raw clusters, of which 19 have positive contact
mass.  Both GT fruits map to cluster 0.  Its single contact centroid is 4.07 cm
from the nearest GT, so the one-centroid estimator produces zero true positives
even when every cluster is emitted.  Because the two GTs do not map to distinct
components, precision at component recall 1.0 is undefined for every S1–S6
ranking.  The same merge remains at `tau_c=0.3`.

## Score choice

No S1–S6 score is selected from this audit.  No single score has a clean
true-versus-false margin on both task scenes, and the harvesting failure occurs
before rejection because two targets occupy one component.  Scores are not
combined into a vote.

The governing principle remains: select one physically justified geometric
signal only if it cleanly separates true attachment candidates after the
candidate representation can express every GT instance.

## Peak reformulation diagnostic

The next cache-only diagnostic replaced each component centroid with local
maxima of its retained-side `rho` field.  A retained primitive is a local peak
when its `rho` is maximal within `sigma`.  Each peak is scored by the `rho` mass
within `sigma`, and candidates are globally suppressed at 3 cm.  The NMS radius
reuses the evaluation matching tolerance: predictions closer than that are not
distinguishable by the evaluator, so it is treated as an existing domain
constant rather than a fitted peak radius.

### Deleafing

- Local peaks before / after NMS: 109 / 57.
- All four GTs have a surviving peak within 3 cm, at score ranks 1, 8, 3, and 6.
- Full recall first occurs at `k=8`: precision 0.50, F1 0.6667.
- Median matched error is 1.34 cm, versus 1.81 cm for legacy on this scene.
- The old centroid has median matched error 0.56 cm only over its three matches;
  it misses the fourth GT, so that lower conditional median is not a superior
  result.

For problematic cluster 2, the old centroid is 19.93 cm from GT.  Its `rho`
field has nine local peaks and seven surviving global NMS.  A peak at
`[-0.418155, -0.295180, -0.214244]` is 1.842 cm from GT and has local peak mass
80.376 (global rank 3).  The strongest mode has mass 97.466 but is 25.82 cm from
GT.  These two dominant modes are 24.11 cm apart; the old centroid lies only
0.80 cm from their connecting line, between them.  This confirms that the
19.93 cm centroid failure is averaging across multiple contact regions.

### Harvesting

- Local peaks before / after NMS: 211 / 98.
- Component 0 has four local modes and two survive global NMS.
- The two surviving positions are
  `[-0.177234, -0.500042, -0.444190]` and
  `[-0.192501, -0.501988, -0.471733]`.
- Their local peak masses are 1.6855 and 1.2550, at global ranks 32 and 35.
- Their distances to the associated fruit-apex GT annotations are 4.352 cm and
  3.028 cm.
- The modes are 3.155 cm apart, while the two fruit-apex annotations are 5.978
  cm apart.
- The old centroid lies 0.876 cm from the midpoint of the two surviving modes.

Thus the harvesting retained-side `rho` field is topologically bimodal and the
centroid is indeed averaging those modes.  Neither retained-side mode lies
within the 3 cm gate around the fruit-apex annotation.  This was initially
treated as localization failure; the annotation-side correction below shows
that the retained field was answering a different interface-side question.

## GT-centered side diagnostic — corrected interpretation

The follow-up diagnostic originally tested whether weak harvesting peaks were
caused by absent retained geometry, absent removed mass, or normal-term
suppression.  The numerical measurements remain valid, but the original A1
interpretation is retracted: harvesting GT is annotated at the tomato apex on
the removed side of the peduncle interface, not at a retained stem-side socket.
For each GT, `m_gt_rho` and `m_gt_rem` use the associated DBSCAN component;
retained mass and counts use all retained primitives in the scene.

| task | GT | component | retained count (1 sigma / 3 sigma) | retained mass (1 sigma / 3 sigma) | removed mass (1 sigma / 3 sigma) | rho mass in 3 sigma | rho mass, no normal term |
|---|---:|---:|---:|---:|---:|---:|---:|
| deleafing | 0 | 0 | 60 / 253 | 47.305 / 214.142 | 1.501 / 24.102 | 275.316 | 384.218 |
| deleafing | 1 | 1 | 24 / 107 | 17.292 / 88.108 | 3.268 / 20.901 | 63.952 | 89.984 |
| deleafing | 2 | 2 | 53 / 224 | 41.545 / 190.631 | 0 / 13.147 | 100.146 | 160.856 |
| deleafing | 3 | 3 | 32 / 144 | 21.178 / 117.455 | 4.648 / 26.332 | 86.294 | 136.859 |
| harvesting | 0 | 0 | **0 / 4** | **0 / 2.681** | 24.191 / 219.582 | **0.570** | **1.585** |
| harvesting | 1 | 0 | **0 / 3** | **0 / 2.055** | 4.611 / 130.077 | **1.255** | **2.780** |

The two harvesting GTs have no retained primitive within `sigma`, while their
removed mass is 24.19 and 4.61 and their nearest removed component primitives
are 0.20 cm and 0.88 cm away.  This is expected for removed-side fruit-apex
annotations and is not evidence of missing reconstruction.  The retained-side
peaks lie across the peduncle interface from those annotations.

Dropping the normal term raises harvesting GT-local rho by factors 2.78 and
2.21, but the counterfactual masses remain only 1.58 and 2.78, versus
89.98--384.22 for deleafing.  The normal term suppresses some already weak
evidence.  This 2.2--2.8x effect is secondary to annotation side, but is not
negligible and remains a measured E3-1 ablation finding.

### Harvesting peak displacement directions

Directions below use the component-0 `w_rem`-weighted removed centroid.  Each
retained-side peak is paired with its nearest fruit-apex annotation.

| peak | GT | GT-to-peak unit vector | dot toward removed centroid | dot toward other apex annotation |
|---|---:|---|---:|---:|
| A | 0 | [-0.655556, 0.110761, 0.746979] | -0.304584 | 0.320318 |
| B | 1 | [0.369938, -0.020230, 0.928836] | 0.078105 | 0.701317 |

Peak A is not strongly aligned with either proposed direction.  Peak B is
substantially aligned toward the other annotation, so adjacency contributes to
its direction.  Neither GT-to-peak displacement points toward the fruit center;
this is consistent with the peaks lying on the retained/stem side, opposite the
removed fruit body, rather than with a missing-signal diagnosis.

## Local peak refinement

Each NMS-surviving peak was refined to the rho-weighted centroid of retained
primitives within `sigma` of that peak.  Refinement occurs after peak scoring
and NMS, so it adds no parameter and changes neither candidate identity nor
rank.

On deleafing, at the same `k=8` full-recall operating point, refinement keeps
TP/FP/FN at 4/4/0 and improves median matched error from **1.340 cm to
0.867 cm**.  The same improvement is obtained when all 57 NMS survivors are
emitted because the four matched modes are unchanged.  This is more complete
and more precise than legacy on this scene (four matches, median 1.812 cm).
The old component centroid's 0.560 cm median remains conditional on only three
matches and therefore is not a like-for-like full-recall result.

The retained-side harvesting field remains at zero recall against removed-side
GT after refinement.  Peak A changes from 4.3525 cm to 4.3500 cm from the fruit
apex; peak B changes from 3.0277 cm to 3.1111 cm.  This comparison measures the
opposite interface sides, not retained-side localization quality.

## Post-retained-side diagnostic

The post-side hypothesis was tested on the same two scenes by retaining removal
evidence on the pre DBSCAN components and moving the outer contact field to
post primitives:

`rho_j = omega_post_j * sum_i delta_i * K_sigma(i,j) * abs(dot(n_i,n_j))`.

The Phase-1 cache did not preserve per-primitive post geometry or confidence.
The diagnostic therefore reloaded the original post PLY, applied the identical
finite and reverse shared-frustum filters, and reconstructed only
`omega_post`.  It did not rerun Stage-2 change scoring or write a new cache.
The trace-only confidence computation is exactly equivalent because each
visible camera contributes `trace(I - vv^T) / d^2 = 2 / d^2`.  Validation
matched the original post populations exactly (150,955 leaf; 138,654 tomato).
Confidence-summary differences were at most 1.4e-6.

### GT coverage

| task | GT | primitive count within 1 sigma, pre-retained / post | count within 3 sigma, pre-retained / post | confidence mass within 1 sigma, pre-retained / post | mass within 3 sigma, pre-retained / post |
|---|---:|---:|---:|---:|---:|
| deleafing | 0 | 60 / 28 | 253 / 227 | 47.305 / 26.782 | 214.142 / 217.047 |
| deleafing | 1 | 24 / **0** | 107 / 108 | 17.292 / **0** | 88.108 / 102.834 |
| deleafing | 2 | 53 / 24 | 224 / 211 | 41.545 / 22.729 | 190.631 / 199.948 |
| deleafing | 3 | 32 / **0** | 144 / 120 | 21.178 / **0** | 117.455 / 113.438 |
| harvesting | 0 | **0 / 0** | **4 / 0** | **0 / 0** | **2.681 / 0** |
| harvesting | 1 | **0 / 0** | **3 / 12** | **0 / 0** | **2.055 / 11.148** |

This is a null result for moving the outer field to post, but it is not evidence
of a reconstruction limit at harvesting GT.  The GTs are on removed fruit
apices, so post retained primitives are not expected at those coordinates.
The post-side experiment still shows that simply changing reconstruction does
not make a retained-side field match removed-side annotations.

Deleafing post coverage is healthy at `3 * sigma` and yields full recall, but
it is not uniformly at least as good locally: GTs 1 and 3 have no post
primitive within `sigma`.  Thus the post side does not provide a uniformly
better task-agnostic retained representation on the two-scene audit.

### Post-side peaks and metrics

| task | peaks before / after NMS | nearest GT peak ranks | precision at recall 1 | median error, raw / refined | pre refined | legacy |
|---|---:|---|---:|---:|---:|---:|
| deleafing | 17 / 15 | 2, 6, 5, 4 | 0.6667 at k=6 | 1.474 / 1.667 cm | 0.867 cm at k=8 | 1.812 cm |
| harvesting | 96 / 46 | 4, 4 | undefined | no matches / no matches | no matches | 0.530 cm over one match |

The post-side deleafing field improves candidate precision at full recall from
0.50 to 0.667, but worsens localization: local refinement changes the median
from 1.474 cm to 1.667 cm.  The pre-side refinement remains the clean
localization result at 0.867 cm.

For harvesting, both GTs select the same nearest post peak.  Its raw distances
are 4.430 cm and 3.758 cm; refinement changes them to 4.561 cm and 3.829 cm.
No post-side threshold reaches nonzero recall.  Moving the retained field to
post is therefore not adopted; the correct change is to represent both sides
of the interface.

## All-scene GT-side audit — retained-support interpretation retracted

The GT-centered support diagnostic was then run over all 60 existing Stage-2
caches: 54 harvesting GTs and 106 deleafing GTs.  This run loaded no pre/post
models and performed no Stage-1/2 recomputation.  For every GT, the diagnostic
records `sigma`, retained counts and confidence mass within `sigma` and
`3 * sigma`, associated-component rho mass within `3 * sigma`, and distance to
the nearest retained primitive.

| task | GTs | n_ret within sigma = 0 | fraction | n_ret within sigma < 10 | fraction < 10 | n_ret median [p25, p75] | nearest retained distance median [p25, p75] |
|---|---:|---:|---:|---:|---:|---:|---:|
| deleafing | 106 | 6 | 5.66% | 19 | 17.92% | 29 [14.25, 52.5] | 0.424 [0.252, 0.654] cm |
| harvesting | 54 | 51 | 94.44% | 54 | 100% | 0 [0, 0] | 4.152 [3.200, 4.903] cm |

The earlier claim that these values measure a 94.34% versus 5.56%
"reconstructed retained-support fraction" is retracted.  The tasks annotate
opposite sides of their interfaces.  Harvesting GTs lie on removed tomato
apices, so the lack of retained primitives at GT and the 4.152 cm median
distance measure peduncle-side separation, not failed reconstruction.
Deleafing GTs are predominantly retained-side, so retained proximity is
expected there.  These numbers are annotation-side measurements and must not be
reported as a dataset observability property.

## E3-3 mirror and interface-pair audit

The fair removed-side mirror was run over all 60 caches with the same DBSCAN
components, peaks, local rho refinement, and 3 cm NMS:

`rho_rem[j] = w_rem[j] * sum_{i in R} w_ret[i] * K_sigma(i,j) * abs(dot(n_i,n_j))`,
for removed primitive `j` in component `c` and all retained primitives `R`.

The same pair contributions also define retained-side rho, so the estimator can
represent an interface as `(g_ret, g_rem)` without task, crop, color, or shape
branching.  Endpoint selection is a downstream tool convention: a stem-side
cutter may use `g_ret`, while a fruit gripper may use `g_rem`.

| task | GTs | mirror peaks before / after NMS | GTs reached at 3 cm | precision at recall 1 | best attainable recall | median matched error | fair legacy F1 / median |
|---|---:|---:|---:|---:|---:|---:|---:|
| harvesting | 54 | 2,247 / 992 | 34 | undefined | 0.6296 | 1.713 cm | 0.9074 / 1.584 cm |
| deleafing | 106 | 2,122 / 902 | 99 | undefined | 0.9340 | 1.312 cm | 0.8780 / 1.182 cm |

The mirror confirms the annotation-side correction but does not yet solve
selection/localization across the dataset.  No score threshold reaches recall
1.0.  At maximum harvesting recall, it emits 945 predictions for 34 TP, giving
precision 0.0360 and F1 0.0681.  Deleafing reaches 99/106 GTs; its best
precision at that recall is 0.3028.  All per-GT within-scene peak ranks are
preserved in the diagnostic CSV.

For each audited GT, its nearest refined mirror peak was paired with the nearest
refined retained local mode from the same DBSCAN component.  Among instances
whose mirror endpoint lies within 3 cm of GT, the inferred endpoint separation
is:

| task | paired instances | median [p25, p75] |
|---|---:|---:|
| harvesting | 34 | **3.144 [1.456, 4.095] cm** |
| deleafing | 99 | **0.609 [0.428, 0.957] cm** |

This supports the interface-pair interpretation: fruit-side and stem-side
peduncle endpoints are separated by centimeters, while the petiole interface
endpoints are much closer.

## Harvesting truss-status and mock-testbed contrast audit

The tomato calyx is part of the removed assembly; the retained anatomical
structure is the thin green truss.  A cache-only diagnostic therefore audited
what geometry actually occupies the corridor above each of the 54 harvesting
GTs.  Color is used **only** for this diagnostic classification and never
enters the estimator.

The operational green gate reuses the legacy diagnostic thresholds:
`G/(R+G+B) >= 0.35` or `G-R >= 0.02`, with blue-dominant points excluded.  The
truss-corridor proxy additionally requires a position 1--8 cm from GT within a
60-degree cone around testbed world +Z.  Black support rods are position-
anchored by the vertical axes through the paired deleafing GTs, using the
scene-derived `sigma` as the XY proximity radius.  Green points outside the
truss and rod proxies are labeled ivy; other points remain unresolved.  These
are auditable position/color proxies, not manually annotated semantic masks.

### A. Truss status

| GT-local corridor status | GTs | fraction |
|---|---:|---:|
| sparse (`n < 10`) | 3 | 5.56% |
| removed-dominant (`n >= 10`, majority `delta > 0.5`) | 17 | 31.48% |
| present, retained or mixed | 34 | 62.96% |

No scene is uniformly sparse.  At scene level, 6/30 are uniformly removed-
dominant, 15/30 are uniformly retained/mixed, and 9/30 contain targets with
different statuses.  The corridor contains a median 40 primitives per GT
[IQR 27.25, 73], with median per-GT `omega` 0.948 [IQR 0.939, 0.960].

The change classification varies sharply with distance:

| distance from GT | proxy primitives | fraction `delta > 0.5` |
|---|---:|---:|
| 1--2 cm | 58 | 82.76% |
| 2--3 cm | 66 | 77.27% |
| 3--4 cm | 155 | 33.55% |
| 4--5 cm | 307 | 29.64% |
| 5--6 cm | 692 | 33.96% |
| 6--7 cm | 920 | 33.70% |
| 7--8 cm | 860 | 21.86% |

Thus case (i), globally missing thin-truss reconstruction, is not supported:
only 3/54 target corridors are sparse and their confidence is otherwise high.
The 1--3 cm interface corridor is predominantly classified as removed, while
the distal 3--8 cm corridor is mostly retained/mixed.  This supports case (ii)
locally, but with an important semantic limit: the position/color proxy cannot
separate the removed calyx from the immediately adjacent green truss.  The
near-GT removed fraction is therefore an upper bound on removed *truss* rather
than proof that every such primitive belongs to the truss.

### B. Geometry nearest the harvesting GT

The 20 nearest retained primitives at every harvesting GT give 1,080 audited
neighbors:

| diagnostic class | primitives | fraction | distance median [IQR] |
|---|---:|---:|---:|
| green truss corridor | 639 | 59.17% | 5.437 [4.656, 6.363] cm |
| paired black-rod axis | 0 | 0% | -- |
| ivy backdrop | 381 | 35.28% | 8.056 [4.741, 9.111] cm |
| unresolved | 60 | 5.56% | 3.667 [2.572, 5.297] cm |

Under this positional audit, the retained neighbors used by the mirror are not
the black background rods.  Ivy nevertheless contributes more than one third
of the nearest-neighbor pool, so background vegetation remains a material
transfer risk.

### C. Is mirror recall caused by the black rod?

For all 34 harvesting GTs reached by the mirror within 3 cm, the nearest
retained primitive to the matched peak is classified as truss for 26 (76.47%),
ivy for 6 (17.65%), unresolved for 2 (5.88%), and black rod for 0.  The paired
black-rod axis is a median 19.41 cm from GT, whereas the nearest retained
primitive is a median 2.88 cm from the mirror peak.

The dot product between the GT-to-peak displacement and the direction from GT
to the nearest paired black-rod axis has median -0.235 [IQR -0.487, 0.297].
Only 12/34 are positive and 4/34 (11.76%) exceed 0.7.  The matched peaks are
therefore not consistently oriented toward the black support rod.  The mirror's
0.630 harvesting recall is not explained by accidental black-rod alignment
under this audit.  The 6 ivy-supported matches must still be treated as
potential testbed/background successes rather than assumed anatomical truss
contacts.

### D. Deleafing rod versus harvesting truss density

For a directly controlled testbed measurement, both structures use world Z.
Harvesting counts green primitives in the one-sided 3 cm segment above the
fruit apex; deleafing counts retained primitives in the two-sided 6 cm segment
through the rod-side GT.  Counts are divided by the available axial length and
all GTs are included; no fitted axis or density threshold is used.

| cone half-angle | harvest primitives/cm | deleaf primitives/cm | deleaf / harvest | paired-scene ratio median [IQR] |
|---:|---:|---:|---:|---:|
| 45 deg | 0.667 | 3.027 | 4.540x | 3.448 [1.263, 9.406] |
| 60 deg | 0.938 | 7.480 | **7.972x** | 8.294 [4.356, 18.729] |
| 75 deg | 1.494 | 14.557 | 9.745x | 10.469 [7.250, 20.812] |

At the primary 60-degree gate, the mock deleafing rod is reconstructed at
about eight times the primitive density of the harvesting truss.  Four of the
30 paired scenes have zero harvesting density in this 3 cm window.  The
contrast persists from 45 to 75 degrees, so it is not a single cone-angle
artifact.

This is a direct limitation of the mock benchmark.  A real deleafing cut
retains a thin green stem, not a thick metal rod.  The measured density contrast
shows how much easier the benchmark makes the retained-side deleafing signal;
it does not establish a causal eight-fold performance gain, but it bounds the
testbed advantage that must be challenged.  This is the strongest current
argument for the planned potted-plant validation, where both retained
structures are thin plant tissue.

## Truss-status cross-tab, applicability curve, and ivy risk

These follow-ups loaded the same 60 caches and the existing all-scene mirror
outputs.  They did not rerun Stage 1/2, modify the estimator, or start a full-
scale experiment.

### Truss status does not explain the mirror recall ceiling

The 34 retained/mixed GTs and 34 mirror-reached GTs are not the same set:

| GT-local truss status | reached | not reached | total |
|---|---:|---:|---:|
| retained/mixed | 23 | 11 | 34 |
| removed-dominant | 8 | 9 | 17 |
| sparse | 3 | 0 | 3 |

Overall harvesting recall is **34/54 = 0.630**.  Excluding only the 17
removed-dominant GTs gives **26/37 = 0.703**, not 34/37.  Retained/mixed GTs
alone are reached at 23/34 = 0.676, while removed-dominant GTs are still
reached at 8/17 = 0.471 and all three sparse proxies are reached.  Therefore
the A status neither defines the observed ceiling nor isolates a valid
removed/retained scope.

This exclusion is also not presently computable without GT: A is centered on
the GT and uses diagnostic color.  It cannot be reported as a deployable pre-
extraction scope condition until an equivalent candidate-centered,
geometry-only rule is defined and validated.

### Unified retained-density curve

The D descriptive densities deliberately measure different physical structures;
they characterize the mock testbed but are not the applicability criterion.
For this pooled curve, the density definition is instead identical for both
tasks: retained primitives satisfying `delta <= 0.5` and `omega-delta > 0`,
within 3 cm of GT and a two-sided 60-degree cone around world Z, divided by the
6 cm available axis length.

| retained density (primitives/cm) | n | match rate [95% Wilson CI] | median matched error | median `m_peak` |
|---|---:|---:|---:|---:|
| 0 | 47 | 0.553 [0.412, 0.686] | 1.690 cm | 0.724 |
| (0, 0.5] | 16 | 0.812 [0.570, 0.934] | 1.359 cm | 13.151 |
| (0.5, 1] | 4 | 1.000 [0.510, 1.000] | 1.187 cm | 51.718 |
| (1, 2] | 14 | 1.000 [0.785, 1.000] | 1.078 cm | 62.938 |
| (2, 5] | 25 | 0.960 [0.805, 0.993] | 1.095 cm | 60.657 |
| (5, 10] | 30 | 1.000 [0.886, 1.000] | 1.472 cm | 88.706 |
| (10, 20] | 15 | 0.933 [0.702, 0.988] | 1.517 cm | 151.395 |
| >20 | 9 | 0.889 [0.565, 0.980] | 1.590 cm | 158.354 |

A monotone isotonic fit crosses match rate 0.5 at **0 primitives/cm**
[scene-bootstrap 95% interval 0, 0.167] and 0.9 at **0.667
primitives/cm** [0.167, 1.0].  The zero 0.5 crossing means this density does
not provide a useful failure floor for the removed-side mirror: many harvesting
GTs are reached even though retained geometry is outside the GT-centered 3 cm
window.

Nearest-retained distance is more discriminative:

| nearest retained distance | n | match rate [95% Wilson CI] | median matched error | median `m_peak` |
|---|---:|---:|---:|---:|
| 0--0.25 cm | 25 | 1.000 [0.867, 1.000] | 1.009 cm | 104.240 |
| 0.25--0.5 cm | 34 | 0.971 [0.851, 0.995] | 1.289 cm | 97.320 |
| 0.5--1 cm | 43 | 0.930 [0.814, 0.976] | 1.488 cm | 58.428 |
| 1--2 cm | 7 | 0.714 [0.359, 0.918] | 1.802 cm | 50.228 |
| 2--3 cm | 7 | 1.000 [0.646, 1.000] | 1.277 cm | 0.106 |
| 3--4 cm | 15 | 0.600 [0.357, 0.802] | 2.503 cm | 0.727 |
| 4--5 cm | 16 | 0.688 [0.444, 0.858] | 1.429 cm | 0.351 |
| >=5 cm | 13 | 0.231 [0.082, 0.503] | 2.477 cm | 0.724 |

The decreasing isotonic fit remains at or above 0.9 through **0.769 cm**
[scene-bootstrap interval 0.574, 3.345] and at or above 0.5 through **4.594
cm** [4.137, 4.907].  The 0.9 distance interval is wide, so it is a measured
trend rather than a stable deployment cutoff.

Only one density bin contains at least five GTs from both tasks.  In that bin,
`(0, 0.5]`, harvesting reaches 9/9 and deleafing reaches 4/7; Fisher's exact
`p=0.0625`.  All higher-density bins are almost exclusively deleafing.  The
dataset therefore does **not** establish that comparable-density task rates
are equal, that density completely explains the task asymmetry, or that task
identity adds no information.

Both the density and distance analyses are GT-centered diagnostics.  Their
thresholds are not available before extraction without knowing the query
location.  A legitimate deployment criterion still requires validation at a
candidate-centered location or as a scene-wide support field.

### Ivy-supported mirror matches

Six of the 34 reached harvesting GTs are ivy-supported.  Their median error is
2.080 cm [IQR 1.293, 2.567], versus 1.379 cm for truss-supported matches.  Their
median `m_peak` is 0.394 [IQR 0.114, 15.458], versus 0.111 for truss-supported
matches.  Ivy is therefore not confined to weak candidates.

Neither current geometric check rejects ivy cleanly:

- A peak-mass threshold low enough to retain all 26 truss-supported matches
  also retains all 6 ivy matches.
- A within-scene rank cutoff retaining all truss-supported matches also retains
  all 6; one ivy match has rank 2.
- Interface-pair distance overlaps almost completely: ivy median 3.249 cm
  versus truss median 3.286 cm.

A semantic truss/ivy consistency filter would remove these cases by definition,
but it uses the diagnostic appearance/position labels and is not admissible in
the geometry-only estimator.  Excluding the six ivy-supported matches gives
the requested transfer-realistic harvesting recall estimate of **28/54 =
0.519**.  Counting only positively identified truss support gives **26/54 =
0.481**; two unresolved matches account for the difference.

The background-rod hypothesis is now resolved and refuted: 26/34 matches are
truss-supported, 0/34 are rod-supported, median alignment toward the rod is
-0.235, and only 4/34 exceed 0.7.  Ivy remains a genuine transfer risk, but it
does not rescue the black-rod hypothesis.

## Annotation side measured from the data

Nearest distances were computed independently to all thresholded removed
primitives (`delta > 0.5`) and all retained candidates.

| task | GTs | nearest retained median | nearest removed median | closer to retained | closer to removed |
|---|---:|---:|---:|---:|---:|
| harvesting | 54 | 4.152 cm | **0.408 cm** | 0 (0%) | **54 (100%)** |
| deleafing | 106 | **0.424 cm** | 0.726 cm | **89 (83.96%)** | 17 (16.04%) |

The measurements unambiguously place harvesting annotations on removed fruit
material.  Deleafing annotations are predominantly on the retained stem-rod
side.  Proximity alone does not resolve the 17 closer-to-removed deleafing
instances, but they do not overturn the task-level retained-side majority.

The canonical target definition is: **harvesting GT marks the tomato apex on
the removed fruit-and-calyx side of the interface with the retained green
truss; deleafing GT marks the magnetic target on the retained stem-rod side of
the petiole interface.**

## Frozen full-scale result

The second estimator revision was frozen before this run as commit `167f9b1`.
All 60 existing Stage-2 caches were then evaluated without re-running Stage 1
or Stage 2.  The evaluation layer selects the retained endpoint for deleafing
and the removed endpoint for harvesting; the estimator itself receives no task
label and returns both sides.  Legacy was re-run from the identical continuous
cache (thresholded at 0.5 for its binary input).

### Head-to-head at 3 cm

| Method | Task | P | R | F1 | median matched error | predictions / GT |
|---|---|---:|---:|---:|---:|---:|
| contact mass, removed endpoint | harvesting | 0.0343 | 0.6296 | 0.0650 | 1.713 cm | 992 / 54 |
| contact mass, retained endpoint | deleafing | 0.0896 | **0.9717** | 0.1641 | **1.070 cm** | 1149 / 106 |
| contact mass | overall | 0.0640 | 0.8562 | 0.1191 | 1.204 cm | 2141 / 160 |
| legacy, identical cache | harvesting | 0.9074 | 0.9074 | 0.9074 | 1.584 cm | 54 / 54 |
| legacy, identical cache | deleafing | 0.9091 | 0.8491 | 0.8780 | 1.182 cm | 99 / 106 |
| legacy, identical cache | overall | 0.9085 | 0.8688 | 0.8882 | 1.309 cm | 153 / 160 |

The peak estimator improves deleafing recall and conditional localization over
legacy, but it does not replace the legacy rejection stage: 2,004 of 2,141
unfiltered candidates are false positives.  Harvesting remains limited to 34
of 54 annotations.  The required harvesting scopes are:

- overall recall: **34/54 = 0.630**;
- excluding six ivy-supported background coincidences: **28/54 = 0.519**;
- positively truss-supported only: **26/54 = 0.481**.

The six ivy-supported matches have median `m_peak = 0.394`, versus roughly
80--98 for the audited true deleafing peaks.  They are background coincidences
and are not expected to transfer.  None of these exclusions is used to improve
the headline metric.

Component abstention (a DBSCAN group has no peak on the evaluated side) is
89/469 = 18.98% for harvesting and 190/515 = 36.89% for deleafing.  No scene
abstains completely.

| Method / task | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| contact / harvesting | 0.000 | 0.0325 | 0.0792 | 0.1466 | 0.4444 |
| contact / deleafing | 0.0734 | 0.1429 | 0.1928 | 0.2698 | 0.5714 |
| legacy / harvesting | 0.5000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| legacy / deleafing | 0.5714 | 0.7768 | 0.8889 | 1.0000 | 1.0000 |

### Rejection signals

Complete PR curves were evaluated in the requested order.  No scores were
combined.

| Signal | Task | true median | false median | best P / R / F1 | unfiltered F1 | conclusion |
|---|---|---:|---:|---:|---:|---|
| pair gap, lower is better | harvesting | 3.144 cm | 1.679 cm | 0.0345 / 0.6296 / 0.0654 | 0.0650 | ordering reversed; six FPs removed |
| pair gap, lower is better | deleafing | 0.587 cm | 2.102 cm | 0.3466 / 0.5755 / 0.4326 | 0.1641 | discriminative |
| bidirectional error, lower is better | harvesting | 0.328 cm | 0.000 cm | 0.0343 / 0.6296 / 0.0651 | 0.0650 | ordering reversed; two FPs removed |
| bidirectional error, lower is better | deleafing | 0.000 cm | 1.281 cm | 0.1658 / 0.9434 / 0.2821 | 0.1641 | discriminative |
| `m_peak`, higher is better | harvesting | 0.135 | 1.209 | 0.0369 / 0.5741 / 0.0694 | 0.0650 | ordering reversed; weak gain |
| `m_peak`, higher is better | deleafing | 56.149 | 0.480 | 0.3490 / 0.8396 / 0.4931 | 0.1641 | discriminative |

Pair gap and bidirectional consistency therefore do not "work" as unified
rejection mechanisms despite literal epsilon-scale F1 increases at thresholds
that remove only six and two harvesting false positives.  `m_peak` is the best
deleafing rejection signal, but it is also reversed on harvesting.  No single
geometry-only rejection score is selected, and no vote is constructed.  This
is the remaining failure of the method.

### Proximity applicability and its GT-free analogue

Retained density is a descriptive testbed property, not the explanatory
variable: the prior isotonic 50% match-rate crossing is 0 primitives/cm.
Nearest-retained proximity is the applicability criterion: the decreasing fit
stays at or above 90% match rate through 0.769 cm and 50% through 4.594 cm.
The task medians are 0.424 cm for deleafing and 4.152 cm for harvesting, which
is consistent with a Gaussian kernel whose scene-typical sigma is about 1.5
cm.

The deployment-time analogue is computed without GT for every DBSCAN group:
minimum distance from its removed-component boundary to a retained primitive.
After associating GTs to their nearest removed component only for this audit,
the medians are:

| Task | GT to nearest retained | component boundary to retained | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|
| harvesting | 4.152 cm | 1.378 cm | 0.353 | 0.259 |
| deleafing | 0.424 cm | 0.085 cm | 0.097 | 0.310 |
| pooled | 0.665 cm | 0.110 cm | 0.675 | 0.642 |

The pooled correlation is moderate, but within-task correlations are weak.
Thus the boundary distance recovers the broad task-level separation but is not
yet validated as an instance-level, GT-free applicability test.  It should be
reported alongside GT-centered proximity, not substituted for it.

## Stage-2 observation

The deleafing smoke scene contains 28 raw DBSCAN change components for four GT
removed modules: approximately 24 are GT-negative change regions rather than
intervention instances.  Stage 2 therefore does not output only physical
removals, even though the benchmark describes ivy and background structure as
unchanged.  The ivy did not move during capture, so ivy-region false change is
reconstruction/registration drift rather than motion.  The legacy consensus
selector was absorbing this upstream noise.

This motivates reporting, but not tuning, a Stage-2 operating frontier.  Three
existing results already show why changing the current score threshold is not
the primary remedy.  First, the identical Stage-2 cache supports legacy F1
0.8882 overall, so sufficient target information is present for a downstream
extractor.  Second, on the smoke scene, the fine-grained setting preserves
recall 1.0 but emits thousands of fragments, while post-filtering to about 22
predictions reduces recall to 0.5; thresholding trades away real intervention
support rather than cleanly separating it.  Third, the two-pass local-drift
prototype below found the scene-global correction already conservative:
localizing it increased component counts and worsened both tasks.  A full
threshold frontier may be added as an ablation, but no operating point is
selected from target F1 and the existing Stage-2 constants remain frozen.

### Read-only continuous replay

Stage 2 was replayed from the original pre/post checkpoints at frozen commit
`167f9b1`, without changing Stage 2 or the frozen Phase-1 caches.  Five scenes
were gated first and the remaining 55 were started only after every gated
`delta` was bit-identical.  All 60 scenes ultimately passed with maximum
absolute `delta` difference exactly `0`; frozen-cache hashes were unchanged.
The isolated replay occupies 2.984 GiB and contains 59 continuous arrays per
scene with no key mismatches, including separate geometry/appearance channels,
best-match kernels and indices, full effective covariances, normal/tangential
displacements, both directions, and reciprocal-match flags.  The run consumed
8.74 aggregate GPU-hours, or about 5.08 wall-clock hours with two GPUs after
the five-scene gate.

The Stage-2 appearance kernel compares only the three SH DC coefficients
(`dc_color`).  Higher-order `sh_rest` coefficients are loaded but do not enter
the appearance score.  Thus view-dependent SH terms are excluded.

### Depth-layer drift diagnostic

For this diagnostic only, a vertical pipe/stem plane was fit from the paired
leaf-socket positions.  Absolute component-centroid distance to that plane was
partitioned by a single three-component GMM over all 984 components.  The
ordered layer means were 4.16, 18.02, and 50.89 cm.  RGB was not used and no
spatial ivy mask was applied.  A true-control component emits at least one
one-to-one matched peak at 3 cm; a false-emitting component emits peaks but no
matched peak.

| Task / component class | Foreground pipe/stem | Mid-ground ivy | Background panel |
|---|---:|---:|---:|
| harvesting false (`n=348`) | 37.36% | 30.46% | 32.18% |
| harvesting true control (`n=27`) | 100.00% | 0.00% | 0.00% |
| deleafing false (`n=213`) | 48.83% | 26.76% | 24.41% |
| deleafing true control (`n=103`) | 92.23% | 7.77% | 0.00% |

Median component-mean `(delta_geo, delta_app)` among false components was
`(0.904, 0.420)`, `(0.968, 0.683)`, and `(0.954, 0.551)` for harvesting
foreground/ivy/background, respectively.  The corresponding deleafing values
were `(0.861, 0.637)`, `(0.945, 0.411)`, and `(0.893, 0.291)`.  Ivy false
components are strongly geometric, but background false components have
similarly high geometry scores.

The median local/global Q0.75 drift ratios were:

| Task | Foreground `(normal, tangential)` | Ivy `(normal, tangential)` | Background `(normal, tangential)` |
|---|---:|---:|---:|
| harvesting false | (3.82, 3.73) | (5.53, 5.31) | (6.08, 5.05) |
| deleafing false | (3.69, 3.28) | (4.77, 4.26) | (4.89, 3.98) |

This first statistic was measured *inside already high-delta components* and
is therefore circularly contaminated by genuine change and drift artifacts.
It motivated the two-pass low-delta experiment below but must not itself be
interpreted as an unbiased local-drift estimate.  In particular, its
under-correction interpretation is superseded by that experiment.

Exact reciprocal best matches do not provide the proposed primitive-level
rejection.  Pooled reciprocal fractions were 0.0797% true versus 0.1528% false
for harvesting and 0.00763% true versus 0.441% false for deleafing.  False ivy
fractions were only 0.0285% and 0.137%, respectively.  The absolute rates are
near zero and the aggregate ordering is opposite the desired separation.
Near-zero exact reciprocity is direct evidence of GS-DIFF representation
non-uniqueness.  Reciprocal matching is closed as a rejection path.

### Peak prominence and component topology

Topographic prominence was computed on each component's retained-primitive
rho graph as the peak value minus the highest saddle on any path to a strictly
higher peak; the component-global maximum uses zero as the field boundary.
The all-scene evaluation contains 2,141 post-NMS peaks from 984 components.

| Task | Peak class | Prominence median [IQR] | `prominence * m_peak` median [IQR] |
|---|---|---:|---:|
| harvesting | true | 0.0619 [0.0122, 0.4192] | 0.0122 [0.0007, 0.6175] |
| harvesting | false | 0.1583 [0.0205, 0.8839] | 0.1876 [0.0026, 5.6354] |
| deleafing | true | 5.2474 [3.7568, 7.3713] | 289.03 [106.90, 662.88] |
| deleafing | false | 0.1771 [0.0209, 1.2888] | 0.0934 [0.0009, 8.1359] |

Prominence is useful for deleafing but reverses on harvesting, so neither it
nor its product with `m_peak` is a unified rejection score.  Best-F1 PR points
for prominence were P/R/F1 = 0.0385/0.6111/0.0724 harvesting and
0.3810/0.8302/0.5223 deleafing.  For the product they were
0.0370/0.6111/0.0698 and 0.3796/0.8774/0.5299.
Together with pair gap, bidirectional consistency, and `m_peak`, this is the
fourth independent rejection signal whose ordering reverses on harvesting.
No further harvesting rejection signals are tested; harvesting is outside the
measured applicability regime.

Before NMS, components contained means of 4.79 harvesting and 5.30 deleafing
peaks; after NMS the means were 2.12 and 2.23.  At the 25%-retained prominence
operating point, the means fell to 0.529 and 0.559 peaks per formed component,
but recall fell from 0.630 to 0.167 for harvesting and from 0.972 to 0.906 for
deleafing.  Within GT-containing components, the matching peak's median raw
rank was 2 for harvesting and 1 for deleafing by both `m_peak` and prominence.
Median 80%-rho radius increased with emitted multiplicity: 1/2/3+ peaks gave
1.33/1.93/2.85 cm for harvesting and 1.21/1.90/3.19 cm for deleafing.

### Geometry-fraction score `struct(c)`

The component score
`sum(omega * delta_geo) / sum(omega * (delta_geo + delta_app))` has the same
ordering in both tasks: true components are lower, not higher.  True/false
medians were 0.510/0.629 for harvesting and 0.503/0.641 for deleafing, so the
ordering does not reverse between tasks.  Ranking low-to-high gives best-F1
P/R/F1 of 0.1136/0.4630/0.1825 for harvesting and
0.3242/0.6698/0.4369 for deleafing; it is not a sufficient unified rejection
mechanism.

The false distribution is bimodal in both tasks rather than unimodal near
zero.  Two-component means/weights are 0.593/0.933 with 79.3%/20.7% weights for
harvesting and 0.019/0.715 with 12.5%/87.5% weights for deleafing.  The
two-component BIC improvements are 41.36 and 132.20, respectively, exceeding
the stated strong-evidence cutoff of 10.  Drift removes cross-reconstruction
explanations and therefore fires `delta_geo`; drift artifacts look structural,
not merely photometric.  The high-`struct` false mode is consequently
interpreted as drift rather than lighting.

### Two-pass local-drift prototype

The local-drift field was prototyped on CPU from the replay cache only.  Pass 1
uses frozen global-drift delta.  Pass 2 estimates Q0.75 normal/tangential
displacement on primitives strictly below the side-specific scene median,
using their `k` nearest low-delta neighbors, then nearest-anchor interpolates
the field to all primitives.  Effective covariance is reconstructed as cached
`Sigma_eff - U_global + U_local(i)`.  Color bandwidth and both scoring
directions are recomputed, and the original 64-neighbor/radius match rule is
re-maximized exactly.  Neither Stage 2 nor Stage 3 source was changed.

A global-covariance CPU control had zero `delta > 0.5` membership flips and
zero component-count difference over all 60 scenes.  Median per-scene median
absolute delta difference from CUDA was 1.86e-8, so grouping changes below are
caused by local covariance rather than CPU/CUDA tie-breaking.

Contrary to the hypothesis, the low-delta local scale is below the global
scale in every layer at the median.  Representative normal/tangential median
ratios are:

| k | Task | Pipe/stem | Ivy | Panel |
|---:|---|---:|---:|---:|
| 64 | harvesting | (0.27, 0.27) | (0.87, 0.88) | (0.44, 0.44) |
| 64 | deleafing | (0.22, 0.22) | (0.67, 0.65) | (0.40, 0.39) |
| 256 | harvesting | (0.26, 0.27) | (0.84, 0.83) | (0.44, 0.44) |
| 256 | deleafing | (0.21, 0.22) | (0.67, 0.64) | (0.40, 0.38) |
| 1024 | harvesting | (0.26, 0.27) | (0.72, 0.70) | (0.42, 0.43) |
| 1024 | deleafing | (0.21, 0.21) | (0.63, 0.59) | (0.39, 0.38) |

Ivy has the largest local/global ratio, but its median is still below one.
The scene-global Q0.75 is conservative, not systematically too small.  Using
the smaller local scale raises rather than suppresses change scores and grows
the DBSCAN population:

| Setting | Task | Components | Precision | Recall | F1 | Median error (cm) |
|---|---|---:|---:|---:|---:|---:|
| frozen global | harvesting | 469 | 0.0343 | 0.6296 | 0.0650 | 1.713 |
| local, k=64 | harvesting | 529 | 0.0077 | 0.1852 | 0.0148 | 2.427 |
| local, k=256 | harvesting | 566 | 0.0069 | 0.1667 | 0.0132 | 2.394 |
| local, k=1024 | harvesting | 603 | 0.0067 | 0.1667 | 0.0128 | 2.406 |
| frozen global | deleafing | 515 | 0.0896 | 0.9717 | 0.1641 | 1.070 |
| local, k=64 | deleafing | 551 | 0.0847 | 0.9623 | 0.1557 | 1.375 |
| local, k=256 | deleafing | 573 | 0.0786 | 0.9528 | 0.1452 | 1.367 |
| local, k=1024 | deleafing | 590 | 0.0765 | 0.9623 | 0.1418 | 1.336 |

All five swept values were run.  Harvesting component count spans 529--603
and deleafing spans 551--590, with counts generally increasing with `k`; there
is no component-count plateau.  F1 ranges are 0.0128--0.0149 harvesting and
0.1418--0.1557 deleafing.  No frozen GT-adjacent component loses all source
support, and zero GTs lose all adjacent components at any `k`.  The recall
loss is therefore downstream peak/contact-field behavior, not deletion of GT
components.

At primary `k=256`, the false `struct(c)` distribution remains bimodal.  Modes
are 0.603/0.897 for harvesting and 0.073/0.744 for deleafing, with BIC
improvements 57.42 and 108.57.  The high structural mode does not collapse;
the localized drift prototype does not remove the drift population.

On frozen deleafing candidates, literal `struct(c) * prominence` gives best
P/R/F1 0.3740/0.8679/0.5227.  This improves over `struct(c)` alone (F1 0.4369)
but is effectively tied with prominence alone (F1 0.5223), so the product adds
no material rejection benefit.

## Oracle-mask localization gate

Parameter-free topographic prominence was frozen first as commit `e478073`.
The commit stores the continuous prominence and ranking only; it does not
hard-code the exploratory deleafing threshold that produced F1 0.5223.  The
full target-extraction test suite passed (60 tests), and stored `xyz`, `m_peak`,
and prominence values on the audited leaf and tomato scenes were bit-identical
to the pre-commit diagnostic.

Two read-only mask oracles were then evaluated on all 60 frozen Stage-2 caches:

1. `legacy_selected`: the underlying dilated support unit selected by the
   reproducible legacy run, reconstructed from its `support_id`;
2. `gt_adjacent_dbscan`: every raw DBSCAN component with removed support within
   `3*sigma` of any GT.  This is a GT-dependent upper-bound diagnostic and is
   independent of legacy selection.

The second oracle does not split a component that contains multiple GTs.
Keeping that representation is intentional: the peak field is supposed to
express multiple contact modes even when DBSCAN merges modules.

| Oracle | Task | Localizer | Recall | Precision | F1 | Median matched error |
|---|---|---|---:|---:|---:|---:|
| legacy-selected masks | harvesting | any contact peak (upper bound) | 0.7593 | 0.1925 | 0.3071 | 1.393 cm |
| legacy-selected masks | harvesting | top-1 `m_peak` | 0.1481 | 0.2500 | 0.1860 | 2.466 cm |
| legacy-selected masks | harvesting | top-1 prominence | 0.1481 | 0.2500 | 0.1860 | 2.521 cm |
| legacy-selected masks | deleafing | any contact peak (upper bound) | 0.8774 | 0.1216 | 0.2135 | 1.020 cm |
| legacy-selected masks | deleafing | top-1 `m_peak` | 0.6698 | 0.7172 | 0.6927 | 1.010 cm |
| legacy-selected masks | deleafing | top-1 prominence | 0.6038 | 0.6465 | 0.6244 | 0.990 cm |
| GT-adjacent DBSCAN | harvesting | any contact peak (upper bound) | 0.7778 | 0.2530 | 0.3818 | 1.394 cm |
| GT-adjacent DBSCAN | harvesting | top-1 `m_peak` | 0.2407 | 0.3939 | 0.2989 | 2.533 cm |
| GT-adjacent DBSCAN | harvesting | top-1 prominence | 0.2963 | 0.4848 | 0.3678 | 2.450 cm |
| GT-adjacent DBSCAN | deleafing | any contact peak (upper bound) | **0.9623** | 0.1579 | 0.2713 | 1.070 cm |
| GT-adjacent DBSCAN | deleafing | top-1 `m_peak` | 0.7642 | 0.7714 | 0.7678 | 1.063 cm |
| GT-adjacent DBSCAN | deleafing | top-1 prominence | 0.6792 | 0.6857 | 0.6825 | 1.043 cm |

The precision of the `any contact peak` row is not a deployable result; that
row asks only whether a geometrically valid mode exists.  Its answer is nearly
yes for deleafing (102/106 GTs) but only 42/54 for harvesting.  Even with the
GT-adjacent mask oracle, the correct mode is top-ranked for only 81/106 leaves
by `m_peak` and 72/106 by prominence, and for 13/54 versus 16/54 fruit targets.
Thus improved mask inference is justified but does not finish Stage 3B.

For GTs with a valid mode, the median within-mask rank is 1 for both scores on
deleafing.  On harvesting it is 2 for both scores, with worst ranks 9
(`m_peak`) and 11 (prominence).  Prominence therefore remains useful as a
continuous deleafing rejection signal, but it is not a universal clean-mask
top-1 rule.

### Nearest-approach control

The scale-free nearest-approach localizer was evaluated in the same pass.  Its
raw endpoint is the removed primitive closest to any retained primitive and
that retained anchor.  Robust variants use the unweighted centroid of removed
primitives in the lowest `{1,5,10,25}` percentile of distance-to-retained.

| Oracle | Task | Variant | Recall | Median matched error |
|---|---|---|---:|---:|
| legacy-selected | harvesting | raw | 0.1481 | 1.958 cm |
| legacy-selected | harvesting | best swept recall (25%) | 0.3519 | 1.886 cm |
| legacy-selected | deleafing | raw | 0.6321 | 1.206 cm |
| legacy-selected | deleafing | best swept recall (1%) | 0.5943 | 1.294 cm |
| GT-adjacent DBSCAN | harvesting | raw | 0.3704 | 2.451 cm |
| GT-adjacent DBSCAN | harvesting | 1/5/10/25% | 0.2593 / 0.2778 / 0.3704 / 0.3704 | 1.559 / 1.766 / 2.036 / 1.803 cm |
| GT-adjacent DBSCAN | deleafing | raw | **0.7736** | 1.207 cm |
| GT-adjacent DBSCAN | deleafing | 1/5/10/25% | 0.6415 / 0.6321 / 0.5943 / 0.5849 | 1.396 / 1.480 / 1.512 / 1.507 cm |

There is no cross-task percentile plateau: larger percentiles help harvesting
but monotonically hurt deleafing.  The nearest-approach hypothesis is therefore
closed as a negative result and the variant is dropped.
With the GT-adjacent oracle, raw nearest approach anchors 24 components on the
truss (70.8% within 3 cm), five on ivy (60.0%), and four on unresolved
structure (0%) for harvesting.  For deleafing it anchors 17 on the green-pipe
proxy (100%), 43 on the hanger/background-rod proxy (97.7%), 43 on ivy
(48.8%), and two unresolved (100%).  This class-conditioned reversal is the
expected background-geometry failure, not evidence for a universal nearest-
approach rule.  Color is used only to label these diagnostic anchors and never
enters either localizer.

### Previously unreached harvesting GTs

All 20 GTs missed by the frozen harvesting mirror lie in a multi-GT DBSCAN
component.  Under either oracle, contact modes recover eight.  Nearest approach
recovers no additional GT that contact peaks do not already recover, leaving
12 recovered by neither localizer.  All 12 have zero retained primitives within
`sigma`; 11/12 also have zero within `3*sigma` (the remaining instance has
five).  Clean component selection therefore fixes 8/20 representation/ranking
failures, but 12/20 remain a genuine missing-anchor limit for this geometry-
only interface formulation.

### Hand/tool-prior coverage and scene-ID resolution

The first manifest-only audit reported 58/60 usable windows because the
30-record phase-review file contains stale scene `2c4c67db08b7`, which has no
frozen benchmark cache, and omits its frozen-cache replacement
`a270c0388f08`.  The replacement has an authoritative local six-phase manifest
and complete MPS products.  After resolving this mapping, all 60/60 benchmark
action windows have at least ten confident samples and a positive empirical
tracking-uncertainty estimate.  No benchmark scene is excluded from the final
action-conditioned gate.  `2c4c67db08b7` is excluded because it is not one of
the 30 frozen benchmark scenes, not because its tracking failed.

MPS contains no explicit tool track and exports a dimensionless hand-confidence
score but no metric hand-position covariance.  The action-likelihood bandwidth
was therefore derived independently for each window as Q0.75 of the 3D
high-frequency residual from each confident pinch-center sample to the
time-linear interpolation of its immediate same-hand neighbors.  Median
bandwidths are 3.59 mm for deleafing and 3.53 mm for harvesting (ranges
2.29--6.36 and 1.34--6.47 mm).  These broad action windows still do not provide
per-target hand/tool association, so coverage does not by itself validate the
prior.  Peak distance is measured to the union of hands with at least ten
confident samples in the window, so the likelihood takes the maximum support
over usable hands without a task-specific hand-side rule.

### Action-conditioned within-component peak ranking

The next gate ranks peaks within each oracle-associated component using:
`S1=m_peak`, `S2=prominence`, `S3=distance to the nearest confident MPS hand
position`, `S4=m_peak*action_likelihood`, and
`S5=prominence*action_likelihood`.  Products are evaluated in log space.  The
bandwidth above is derived solely from MPS trajectory uncertainty and is not
selected from extraction metrics.

| Task | Score | Top-1 | Top-3 | Valid-mode ceiling |
|---|---|---:|---:|---:|
| deleafing | S1 `m_peak` | 81/106 (0.764) | 95/106 (0.896) | 102/106 (0.962) |
| deleafing | S2 prominence | 72/106 (0.679) | 92/106 (0.868) | 102/106 (0.962) |
| deleafing | S3 hand distance only | **92/106 (0.868)** | 95/106 (0.896) | 102/106 (0.962) |
| deleafing | S4 `m_peak` x action | **92/106 (0.868)** | 95/106 (0.896) | 102/106 (0.962) |
| deleafing | S5 prominence x action | **92/106 (0.868)** | 95/106 (0.896) | 102/106 (0.962) |
| harvesting | S1 `m_peak` | 13/54 (0.241) | 36/54 (0.667) | 42/54 (0.778) |
| harvesting | S2 prominence | 16/54 (0.296) | 35/54 (0.648) | 42/54 (0.778) |
| harvesting | S3/S4/S5 (each) | 16/54 (0.296) | 35/54 (0.648) | 42/54 (0.778) |

On deleafing, hand conditioning produces a net gain of 11 toward the 21-target
gap between `m_peak` and the valid-mode ceiling, closing 52.4% of that gap.  It
promotes 15 GTs that S1 misses but demotes four that S1 gets right.  The
ablation is decisive: hand distance alone, `m_peak` times action, and prominence
times action have identical top-k accuracy.  S3 and S4 choose exactly the same
top peak in all 105 GT-adjacent deleafing oracle components; S3 and S5 agree in
104/105.  The gain is therefore entirely attributable to hand position under
the uncertainty-derived bandwidth.  Contact geometry contributes no measurable
within-component top-1 benefit to the product.

This is a useful action prior but does **not** pass the stronger gate for a
geometry-plus-action selector.  It must remain a separately ablated cue; the
paper cannot claim that multiplicative action conditioning solves endpoint
selection geometrically.  The remaining clean-mask selection gap is real and
should be reported rather than hidden inside the hand prior.  Harvesting is
reported only for symmetry: 12/54 targets still have no retained anchor even
under oracle masks, and the prior does not change that applicability limit.

### Geometry-localization gate against hand-only baselines

The geometry gate was run over the same complete manipulation windows and
GT-adjacent DBSCAN oracle associations as the preceding ranking diagnostic.  It
does not segment events.  B1 and B2 are deliberately strong GT-conditioned
hand-trajectory upper bounds; B3--B6 preserve the prior oracle-mask evaluation
protocol.  Median error below is the median over all finite per-GT predictions,
not only matched predictions.

| Task | Baseline | Predictions / GT | Median error | R@3 cm | R@5 cm | R@10 cm |
|---|---|---:|---:|---:|---:|---:|
| deleafing | B1 closest raw hand sample to GT | 106/106 | 8.801 cm | 0.0283 | 0.1981 | 0.7547 |
| deleafing | B2 closest velocity-minimum hand sample to GT | 106/106 | 9.063 cm | 0.0094 | 0.1792 | 0.6321 |
| deleafing | B3 removed primitive nearest the hand | 105/106 | 13.179 cm | 0.0660 | 0.1132 | 0.3585 |
| deleafing | B4 retained primitive nearest the hand | 105/106 | 2.383 cm | 0.6604 | 0.9057 | 0.9245 |
| deleafing | B5 hand-selected raw contact peak | 105/106 | 1.498 cm | 0.8396 | 0.9151 | 0.9151 |
| deleafing | **B6 hand-selected, rho-refined contact peak** | **105/106** | **1.204 cm** | **0.8679** | **0.9151** | **0.9151** |
| harvesting | B1 closest raw hand sample to GT | 54/54 | 5.488 cm | 0.0556 | 0.3148 | 1.0000 |
| harvesting | B2 closest velocity-minimum hand sample to GT | 54/54 | 5.538 cm | 0.0370 | 0.3148 | 1.0000 |
| harvesting | B3 removed primitive nearest the hand | 54/54 | 6.333 cm | 0.0000 | 0.2407 | 0.9074 |
| harvesting | B4 retained primitive nearest the hand | 54/54 | 4.524 cm | 0.1296 | 0.5741 | 1.0000 |
| harvesting | B5 hand-selected raw contact peak | 54/54 | 3.931 cm | 0.2407 | 0.8519 | 1.0000 |
| harvesting | B6 hand-selected, rho-refined contact peak | 54/54 | 3.818 cm | 0.2963 | 0.8148 | 1.0000 |

For deleafing, B6 improves over B3 by a paired median 10.537 cm (95% paired
bootstrap interval 7.639--12.735 cm; exact sign-test
`p=1.50e-20`).  More importantly, refinement itself improves B6 over the same
raw peak B5 by 0.339 cm (95% interval 0.198--0.409 cm; `p=7.44e-7`).  Contact
geometry therefore adds measurable localization beyond hand position and raw
primitive selection on the applicable deleafing task.

For harvesting, B6 improves over B3 by 3.019 cm (95% interval
2.152--3.442 cm; `p=3.80e-11`), but its refinement over B5 is only 0.041 cm;
the interval reaches zero (`0.000--0.164` cm improvement) and the sign test is
not significant at 0.05 (`p=0.0541`).  This is consistent with the previously
measured harvesting anchor/annotation limitation, not evidence for a strong
harvesting refinement claim.

### Five-scene manipulation-event gate

A GT-blind velocity/dwell segmenter was frozen before count comparison on the
first five benchmark prefixes lexicographically.  Per-hand centered speed was
split by a two-component log-speed GMM; contiguous low-speed runs of at least
five samples were treated as dwells, and temporally overlapping left/right
dwells were merged.

| Scene prefix | GT actions | Events, 3-sample floor | Events, 5-sample primary | Events, 10-sample floor |
|---|---:|---:|---:|---:|
| `040bc88b3760` | 4 | 10 | 5 | 5 |
| `16b2d1bf9fa9` | 4 | 12 | 10 | 5 |
| `173cb38286e9` | 4 | 10 | 7 | 4 |
| `17ef081b9d6c` | 4 | 10 | 9 | 6 |
| `35aa89c0e852` | 3 | 11 | 9 | 6 |

The primary count is exact in 0/5 scenes, has total absolute count error 21,
and is stable across the three reporting floors in 0/5 scenes.  Trajectory
availability therefore does not imply event separability.  Global per-event
assignment is not built from this signal; the supported action mechanism is
window-level association, and its oracle conditioning must remain explicit.

### All-scene component-feature separability audit

All 984 frozen DBSCAN components were labeled GT-adjacent only for evaluation.
The first 20 base-scene prefixes lexicographically are development and the final
10 are held out, with the identical split for both tasks.  There are 105/410
true/false deleafing components and 33/436 true/false harvesting components;
the lower harvesting component count relative to 54 GTs reflects merged-target
components and is not a localization ceiling.

All feature directions are crop-agnostic and frozen from their physical
meaning.  Previously side-dependent contact quantities use a symmetric
bilateral aggregation.  No classifier or feature weights are fit.

| Development rank | Feature | Deleaf AUPRC | Harvest AUPRC | Macro AUPRC | Ordering reversal |
|---:|---|---:|---:|---:|---:|
| 1 | hand-trajectory proximity | 0.8632 | 1.0000 | **0.9316** | no |
| 2 | within-scene removed-mass rank | 0.9325 | 0.4817 | 0.7071 | no |
| 3 | total removed mass | **0.9674** | 0.3546 | 0.6610 | no |
| 4 | within-scene component-size rank | 0.9301 | 0.3821 | 0.6561 | no |
| 5 | component size | 0.9641 | 0.2974 | 0.6308 | no |
| 6 | post-pose Fisher confidence at pre positions | 0.3741 | 0.4690 | 0.4215 | no |

Reverse-change new-surface support is sparse: only 25/515 leaf components and
14/469 fruit components have any, and the true and false medians and IQRs are
all zero.  The stricter empty-match auxiliary is zero everywhere.  Thus F1 is
non-discriminative under either auditable counterpart definition, rather than
the expected high-value cue.  Empty-neighbor fraction, objectness, and `struct`
are weak at scale.  Low contact fraction, bilateral
`conf_c`, and bilateral `m_peak` reverse their true/false median ordering across
tasks and cannot serve as unified rejection signals.

Absolute size and removed mass genuinely separate the uniform printed leaf
modules: dev AUPRC is 0.9641/0.9674.  The same cues are much weaker on fruit,
so the one-scene perfect size margin does not generalize across tasks.  Size is
reported honestly but remains a testbed cue requiring potted-plant validation.

The top features are not fully redundant.  On pooled development components,
Spearman correlation between hand proximity and within-scene removed-mass rank
is 0.337 (0.653 leaves, 0.236 fruit).  This motivated exactly one scale-free
product: hand-proximity rank times removed-mass rank, representing substantial
disappearance at the manipulated location.

| Frozen held-out component filter | Task | AUPRC | P@R95 | P/R/F1 at pooled-dev threshold |
|---|---|---:|---:|---:|
| hand proximity | deleafing | 0.7113 | 0.5000 | 0.4923 / 0.9143 / 0.6400 |
| hand proximity | harvesting | 0.8899 | 0.6500 | 0.7059 / 0.9231 / 0.8000 |
| hand proximity | pooled | 0.7608 | 0.5349 | 0.5366 / 0.9167 / 0.6769 |
| mass-rank x hand-rank | deleafing | **0.9211** | 0.6182 | 0.9000 / 0.7714 / 0.8308 |
| mass-rank x hand-rank | harvesting | 0.8548 | 0.1368 | 0.6000 / 0.9231 / 0.7273 |
| mass-rank x hand-rank | pooled | **0.8567** | 0.4946 | 0.7800 / 0.8125 / 0.7959 |

The unified development threshold for hand distance is 7.732 cm, selected as
the highest precision point with pooled component recall at least 0.95.  The
product threshold is 0.6694 under the same rule.  Held-out end-to-end target
metrics show why component-level AUPRC alone is insufficient:

| Filter | Task | TP/FP/FN | P/R/F1 | Median matched error |
|---|---|---:|---:|---:|
| none | overall | 49/551/5 | 0.0817 / 0.9074 / 0.1498 | 1.248 cm |
| hand proximity | deleafing | 32/65/3 | 0.3299 / 0.9143 / **0.4848** | 1.226 cm |
| hand proximity | harvesting | 14/28/5 | 0.3333 / 0.7368 / **0.4590** | 1.289 cm |
| hand proximity | overall | 46/93/8 | 0.3309 / 0.8519 / **0.4767** | 1.249 cm |
| mass-rank x hand-rank | deleafing | 27/33/8 | 0.4500 / 0.7714 / **0.5684** | 1.248 cm |
| mass-rank x hand-rank | harvesting | 14/62/5 | 0.1842 / 0.7368 / **0.2947** | 1.289 cm |
| mass-rank x hand-rank | overall | 41/95/13 | 0.3015 / 0.7593 / **0.4316** | 1.249 cm |

Hand proximity is the only strong cross-task single feature and improves
held-out overall target F1 by 3.18x without changing localization error.  The
product is complementary for leaf-component rejection but is not a better
unified target filter: it sacrifices leaf recall and passes fruit components
that emit more peaks, worsening overall and harvesting end-to-end F1.  It is
therefore reported but not selected.  The common measurable property of GT
components is proximity to the manipulation trajectory; high removed mass is a
secondary, module-dependent property.  This audit addresses component rejection
only and neither splits merged instances nor selects a mode within a component.

## Final held-out component/peak experiment

This is the last method experiment.  It tests the previously missing sequential
composition: manipulation-hand proximity filters DBSCAN components, then
topographic prominence ranks the peaks emitted by each surviving component.
The split remains fixed at the first 20 scene prefixes per task for development
and the final 10 per task for held-out evaluation.  No held-out metric selects
an operating point.

To avoid a metric-distance cutoff, the component gate ranks components by
nearest-hand distance within each scene; rank 1 denotes the nearest component.
The development component-recall rule selects rank quantile `q=0.75`, retaining
the nearest 25% of components.  Development selects the single most prominent
peak per component for both prominence-only and sequential variants.  These
choices are then frozen for the held-out table.

| Variant | Task | TP/FP/FN | P | R | F1 | Median matched error |
|---|---|---:|---:|---:|---:|---:|
| no filter | deleafing | 35/297/0 | 0.1054 | **1.0000** | 0.1907 | 1.204 cm |
| no filter | harvesting | 14/254/5 | 0.0522 | **0.7368** | 0.0976 | 1.289 cm |
| no filter | overall | 49/551/5 | 0.0817 | **0.9074** | 0.1498 | 1.248 cm |
| hand rank only | deleafing | 26/43/9 | 0.3768 | 0.7429 | 0.5000 | 1.131 cm |
| hand rank only | harvesting | 14/74/5 | 0.1591 | **0.7368** | 0.2617 | 1.289 cm |
| hand rank only | overall | 40/117/14 | 0.2548 | 0.7407 | 0.3791 | 1.249 cm |
| prominence only | deleafing | 25/63/10 | 0.2841 | 0.7143 | 0.4065 | 1.204 cm |
| prominence only | harvesting | 7/104/12 | 0.0631 | 0.3684 | 0.1077 | 2.372 cm |
| prominence only | overall | 32/167/22 | 0.1608 | 0.5926 | 0.2530 | 1.311 cm |
| hand rank + prominence | deleafing | 22/7/13 | **0.7586** | 0.6286 | **0.6875** | 1.131 cm |
| hand rank + prominence | harvesting | 7/29/12 | **0.1944** | 0.3684 | 0.2545 | 2.372 cm |
| hand rank + prominence | overall | 29/36/25 | **0.4462** | 0.5370 | **0.4874** | 1.374 cm |

The two-level composition is real on deleafing: relative to hand rank alone it
removes 36 false positives at a cost of four true positives and raises F1 from
0.5000 to 0.6875.  It does not transfer to harvesting.  Prominence removes half
of the harvesting true positives and raises median error from 1.289 to 2.372 cm,
consistent with its previously measured reversed true/false ordering on that
task.  Consequently, hand plus prominence is not adopted as a unified
crop-agnostic rule.

The complete held-out curves sweep every attainable within-scene hand-rank
quantile and every prominence top-k.  Only the sequential variant attains 90%
precision at nonzero recall: deleafing reaches recall 0.400 at precision 0.933
(`q=0.9231`, top-1).  Recall at precision at least 0.95 is zero for every
variant and task; harvesting and overall recall at both 0.90 and 0.95 precision
are also zero.  The full 15,252-point table and plotted curves are retained as
artifacts rather than reducing this result to a best-F1 point.

### Within-scene rank decision

The rank gate does not reproduce the absolute 7.732 cm hand gate.  Relative to
the frozen absolute-distance result, hand-rank-only F1 changes by +0.0152 for
deleafing, -0.1973 for harvesting, and -0.0975 overall.  It therefore fails the
predeclared five-point comparability check and is **not adopted**.  The evidence
does not support reducing the declared parameter count by replacing the metric
cutoff with a rank quantile.  If the hand filter is retained in a deployable
method, 7.732 cm remains an explicit third tuned value; otherwise hand proximity
should be presented as an ablated diagnostic cue rather than silently selecting
a favorable held-out rank.

## Action-specific large/high-score mask gate

After relaxing the original universal-selector constraint, the proposed mask
rule was tested directly: accept a DBSCAN component only when it is both large
and has high mean Stage-2 score.  Size and mean-score cutoffs were selected
separately for each action on the fixed development scenes by maximizing
component precision subject to component recall at least 0.95, then frozen on
the same 10-scene-per-task held-out split.  This is an explicitly
action-conditioned four-threshold rule, not a parameter-free unified method.

Development selects:

- deleafing: size at least 1,409 primitives and mean `delta` at least 0.8204;
- harvesting: size at least 395 primitives and mean `delta` at least 0.9008.

| Evaluation level | Task | TP/FP/FN | P | R | F1 |
|---|---|---:|---:|---:|---:|
| held-out components | deleafing | 27/1/8 | 0.9643 | 0.7714 | 0.8571 |
| held-out components | harvesting | 9/0/4 | **1.0000** | 0.6923 | 0.8182 |
| held-out targets | deleafing | 27/52/8 | 0.3418 | 0.7714 | 0.4737 |
| held-out targets | harvesting | 12/18/7 | 0.4000 | 0.6316 | 0.4898 |
| held-out targets | overall | 39/70/15 | 0.3578 | 0.7222 | 0.4785 |

This cleanly separates the two Stage-3 subproblems.  The action-specific rule
does select high-quality masks: it retains only one false deleafing component
and no false harvesting components on held-out scenes.  End-to-end precision
remains low because the retained components emit multiple refined peaks.  Mask
rejection is therefore no longer the dominant error under this rule;
within-component peak selection is.

The visual impression that no false harvesting region is tomato-sized does not
hold over all scenes when size is used alone.  False harvesting components reach
5,064 primitives on development and 4,438 on held-out, versus true maxima of
1,527 and 1,080.  The mean-score conjunction, not size alone, removes those
large drift components.  Conversely, a held-out GT-adjacent harvesting
component contains only 15 primitives, explaining part of the recall loss.

For reference, task-specific mean `delta` alone gives held-out target F1
0.5238 deleafing, 0.4727 harvesting, and 0.5083 overall.  It has better target
F1 but substantially worse component purity; this happens because it passes a
different population with fewer emitted peaks, not because it identifies masks
more accurately.  The held-out legacy values on the same 20 scenes remain
0.8615/0.8947/0.8738 F1 for deleafing/harvesting/overall.

### All-30 threshold oracle

Both size and mean-score cutoffs were subsequently swept over all 30 scenes per
task, with independent values allowed for deleafing and harvesting.  This is an
explicit in-sample diagnostic oracle.  It measures the attainable operating
region but is not unbiased held-out performance and cannot replace the fixed
split table above in a paper.

The mask/component-F1 optima are:

| Task | Minimum size | Minimum mean `delta` | Component P/R/F1 | Target P/R/F1 at same point |
|---|---:|---:|---:|---:|
| deleafing | 1,278 | 0.8106 | 0.9703 / 0.9333 / **0.9515** | 0.2914 / 0.8962 / 0.4398 |
| harvesting | 205 | 0.9008 | 0.9412 / 0.9697 / **0.9552** | 0.3820 / 0.6296 / 0.4755 |

These values support using a lower harvesting size threshold: 32/33
GT-adjacent harvesting components are retained with only two false components.
The thresholded Stage-2 components can therefore be selected accurately under
an action-specific rule.

Optimizing exact one-to-one target F1 instead selects size/mean cutoffs of
702/0.8716 for deleafing and 313/0.9198 for harvesting.  The corresponding
target results are 0.3583/0.8113/0.4971 and
0.4103/0.5926/0.4848 P/R/F1, or 0.3711/0.7375/0.4937 overall.  Target-optimal
thresholds deliberately discard valid masks because each accepted mask can
emit several false peaks.  This is further evidence that mask selection and
within-mask point selection are distinct problems; tuning the mask cutoff
cannot repair the peak multiplicity error.

### One peak per accepted component

The component-optimal masks were also evaluated with exactly one output per
component, selected either by maximum local contact mass (`m_peak`) or maximum
prominence.  This rule is beneficial for deleafing but structurally invalid for
harvesting.

| Setting | Task | P/R/F1 |
|---|---|---:|
| all peaks after NMS | deleafing | 0.2914 / 0.8962 / 0.4398 |
| top-1 `m_peak` per component | deleafing | **0.7327 / 0.6981 / 0.7150** |
| top-1 prominence per component | deleafing | 0.6535 / 0.6226 / 0.6377 |
| all peaks after NMS | harvesting | 0.3820 / 0.6296 / 0.4755 |
| top-1 `m_peak` per component | harvesting | 0.4545 / 0.2778 / 0.3448 |
| top-1 prominence per component | harvesting | 0.5152 / 0.3148 / 0.3908 |

The same conclusion holds on the fixed held-out split: top-1 `m_peak` raises
deleafing F1 from 0.4737 to 0.6984 but lowers harvesting F1 from 0.4898 to
0.3571.  Across all scenes, no deleafing DBSCAN component is adjacent to more
than one GT.  In harvesting, 23 components are each adjacent to two GT fruit
targets, accounting for 46/54 harvesting annotations.  A one-output rule thus
has a harvesting structural recall ceiling of 31/54 = 0.5741 even with oracle
peak selection.  Harvesting needs within-component multi-mode output or an
additional instance split; deleafing can defensibly use one peak per accepted
component.

### Deleafing within-component selector

Every retained-side peak in a GT-adjacent deleafing component was audited on
the fixed 20-scene development / 10-scene held-out split.  The audit contains
646 peaks in 105 components; 109 peaks are within 3 cm of a target, and a valid
peak exists for 102/106 targets.  Thus candidate generation is nearly complete,
but a purely geometric maximum does not reliably identify the acted-on mode.

| Split | Candidate ceiling | max `m_peak` top-1 | max prominence top-1 | nearest-hand top-1 |
|---|---:|---:|---:|---:|
| development | 67/71 (0.944) | 52/71 (0.732) | 47/71 (0.662) | **58/71 (0.817)** |
| held-out | 35/35 (1.000) | 29/35 (0.829) | 25/35 (0.714) | **34/35 (0.971)** |
| all 30 | 102/106 (0.962) | 81/106 (0.764) | 72/106 (0.679) | **92/106 (0.868)** |

With one-to-one matching at 3 cm, emitting the nearest-hand refined peak once
per oracle component gives 34/1/1 TP/FP/FN on held-out scenes
(P/R/F1 = 0.9714/0.9714/0.9714; median error 1.140 cm).  Across all 30 scenes
it gives 92/13/14 (0.8762/0.8679/**0.8720**; median error 1.043 cm).  These are
oracle-component peak-selection metrics, not deployable end-to-end numbers,
because every evaluated component is GT-adjacent by construction.

At this intermediate gate, the strongest previously tested criterion was to
choose the refined contact mode with minimum Euclidean distance to a confident
MPS pinch-center sample in the manipulation window.  It is retained as an
action-association ablation; the final all-30 decision below supersedes it with
the geometry-only interface-boundary selector.  The existing sigma-local
rho-weighted refinement remains responsible for the final geometric position.

On held-out scenes the action cue improves top-1 accuracy by 0.143 over maximum
`m_peak`; the scene-bootstrap 95% interval is [0.031, 0.256].  It corrects five
`m_peak` failures and introduces none, although the exact paired sign-test is
only p=0.0625 because there are five discordant held-out cases.  One held-out
target remains incorrectly selected despite a valid peak: its correct mode is
ranked 11th by hand distance.  Products of `m_peak` or prominence with the
uncertainty-derived action likelihood do not improve the rule: they choose the
same top peak as hand distance in 105/105 and 104/105 components.  Pair gap is
not a viable primary alternative despite high peak-level separability because
it is defined in only 60% of components and has lower held-out top-1 accuracy.

This result is an oracle of within-component selection only.  Component
acceptance/rejection remains separate, and four GTs with no valid contact mode
cannot be recovered by any ranker.

### Exploratory geometry-only interface-boundary selector

An independent all-peak audit identified a stronger crop-agnostic geometric
property shared by the correct deleafing modes.  For retained-side candidate
`p` and removed component `c`, define the removed-mass directional moment

`b(p) = ||sum_i delta_i unit(mu_i-p)|| / sum_i delta_i`

and score it by the estimator's existing Gaussian proximity kernel at the
nearest removed primitive:

`S_boundary(p) = b(p) exp(-d(p,c)^2 / (2 sigma^2))`.

The first term is high when the removed material lies predominantly on one
side of the candidate; the second requires the candidate to touch the removed
component.  It therefore describes a generic interface boundary rather than a
leaf endpoint, uses no color or hand tracking, introduces no fitted cutoff,
and reuses the existing kernel scale.

Within GT-adjacent components, `S_boundary` selects a valid peak for 65/70
development components and 35/35 held-out components.  Exact all-30 oracle
metrics are 100/5/6 TP/FP/FN, or 0.9524/0.9434/**0.9479** P/R/F1 with 1.072 cm
median error.  For comparison, one-sidedness without proximity gives F1
0.8910; hand proximity gives 0.8720; maximum `m_peak` gives 0.7678.

The deployable held-out experiment applies the development-selected component
gate (`size >= 1409` and mean `delta >= 0.820429`) before peak selection.  It
accepts 28 components for 35 GT targets.  `S_boundary` then gives 27/1/8,
or **0.9643/0.7714/0.8571** P/R/F1 with 1.056 cm median error.  This equals the
component gate's attainable TP count and is close to the identical-split legacy
F1 of 0.8615.  Continuity-plus-hand gives 26/1/9 (F1 0.8387), so hand tracking
is not required for this held-out pipeline.

The total removed-evidence gate `M_c = sum_i delta_i` was subsequently evaluated
under four normalizations, with threshold selection and evaluation both using
all 30 scenes.  These are explicitly **in-sample discovery results**, not
held-out validation.  DBSCAN, the interface-boundary selector, refinement, and
matching were fixed throughout.

| Gate score | Best threshold | TP/FP/FN | P/R/F1 | median error | scene-bootstrap F1 95% CI |
|---|---:|---:|---:|---:|---:|
| `M_c / sum_scene(delta)` | 0.048462 | 96/6/10 | 0.9412/0.9057/**0.9231** | 1.043 cm | [0.8767, 0.9633] |
| within-scene percentile rank | 0.800000 | 94/12/12 | 0.8868/0.8868/**0.8868** | 1.066 cm | [0.8421, 0.9275] |
| `M_c / N_scene_overlap` | 0.003775 | 98/9/8 | 0.9159/0.9245/**0.9202** | 1.060 cm | [0.8756, 0.9600] |
| absolute `M_c` | 638.554857 | 98/9/8 | 0.9159/0.9245/**0.9202** | 1.060 cm | [0.8756, 0.9600] |
| GT-adjacent-component oracle ceiling | -- | 100/5/6 | 0.9524/0.9434/**0.9479** | 1.072 cm | -- |

At the maximum-recall points satisfying precision at least 0.90, scene-total
delta normalization and the overlap-count/absolute forms all give 98/9/8
(recall 0.9245); within-scene rank gives 86/7/20 (recall 0.8113).  At precision
at least 0.85, the two density-normalized forms reach recall 0.9528, while rank
reaches 0.9151 and the absolute form remains at 0.9245.  The complete 0.85,
0.90, and 0.95 operating points, including scene-bootstrap confidence intervals
for precision, recall, F1, and median error, are stored with the full curves.

Threshold flatness was compared in a common accepted-component coordinate.  The
within-one-F1-point plateau covers 3.30% of all components for within-scene rank,
2.33% for overlap-count normalization, 1.17% for absolute mass, and 0.97% for
scene-total-delta normalization.  After perturbing the accepted fraction by
approximately +/-5 percentage points, the worst F1 losses are 0.0340, 0.0586,
0.0792, and 0.1319, respectively.  Leave-one-scene-out threshold reselection
also gives the smallest worst full-data F1 loss for rank (0.00095).  Thus
**within-scene rank is the least threshold-sensitive form**, although it trades
away 3.63 F1 points relative to the best scene-total-delta result.  It is kept as
a robustness ablation rather than the primary gate.

The adopted primary gate is **`M_c / N_scene_overlap >= 0.003774612`**.  Its F1
0.9202 is within the nearly identical bootstrap intervals of scene-total-delta
normalization's 0.9231, while its one-point plateau is 2.4 times wider and its
local +/-5% sensitivity is 2.25 times smaller.  It also avoids a structural
circularity: total scene delta increases with the amount removed, so the
quantity being detected appears in the scene-delta denominator.  Overlap count
measures reconstruction size independently of how much changed.  Scene-delta
normalization remains in the table only as an explicitly circular ablation.

All operating points in this sweep are in-sample.  Fresh-scene or potted-plant
data is required to validate any selected threshold.

The final all-30 oracle-component ablation isolates the two factors in the
interface-boundary selector:

| Selector | TP/FP/FN | P/R/F1 | median error |
|---|---:|---:|---:|
| one-sidedness `b(p)` only | 94/11/12 | 0.8952/0.8868/**0.8910** | 1.076 cm |
| Gaussian nearest-distance only | 85/20/21 | 0.8095/0.8019/**0.8057** | 1.063 cm |
| **`b(p)` x Gaussian nearest-distance** | **100/5/6** | **0.9524/0.9434/0.9479** | **1.072 cm** |

Distance alone therefore does not explain the result.  Proximity identifies
contact adjacency, while one-sidedness distinguishes an interface boundary
from nearby interior or clutter modes; both are required.  Here `d(p,c)` is
distance to the nearest primitive in the thresholded DBSCAN component, the sum
uses all primitives in that component (all already satisfy `delta > tau_c`),
and `sigma` is exactly the scene-derived contact-kernel sigma.  Replacing the
nearest-primitive definition with either an unweighted or delta-weighted
component centroid lowers both distance-only and product F1 to 0.6446.

This feature was discovered after repeated inspection of the benchmark and the
nominal held-out scenes; the Gaussian form was checked after a preliminary
rational proximity form.  The result is consequently exploratory, not clean
untouched-test evidence.  It requires fresh-scene or potted-plant confirmation
before becoming the paper's headline.  It should be presented as a proposed
interface-boundary principle with a full ablation (`b`, proximity, product),
not as a post-hoc final benchmark claim.

## Final method decision

| Method, all 30 deleafing scenes | Precision | Recall | F1 | median error | declared tuned parameters |
|---|---:|---:|---:|---:|---:|
| **Ours: `M_c/N_overlap` + boundary contact peak** | **0.9159** | **0.9245** | **0.9202** | **1.060 cm** | **3** |
| Legacy, identical Stage-2 cache | 0.9091 | 0.8491 | 0.8780 | 1.182 cm | approximately 20 |
| GT-adjacent-component oracle ceiling | 0.9524 | 0.9434 | 0.9479 | 1.072 cm | -- |

For deleafing, the final in-sample Stage-3 pipeline is continuous Stage-2 change
evidence, standard DBSCAN, normalized component evidence
`M_c/N_scene_overlap`, retained-side contact modes, the one-sided/proximal
interface-boundary selector, and sigma-local rho refinement.  On all 30 scenes
it gives 98/9/8 TP/FP/FN: precision 0.9159, recall 0.9245, F1 **0.9202**, and
median matched error **1.060 cm**.  The superseded 1.038 cm value came from a
different held-out operating point.  Identical-cache legacy gives 0.9091 /
0.8491 / **0.8780** and 1.182 cm; the GT-adjacent-component ceiling is 0.9524 /
0.9434 / **0.9479** and 1.072 cm.

The requested table declares three tuned Stage-3 parameters versus legacy's
approximately 20 Table-6 parameters.  This count treats the fixed `delta > 0.5`
mask as the shared Stage-2 input convention; Stage 3 contributes DBSCAN radius,
DBSCAN density floor, and the normalized mass threshold.  Derived sigma and the
3 cm evaluator/tool resolution are not tuned.  If `tau_c=0.5` is instead counted
as a Stage-3 re-tuning liability, the literal count is four; this accounting
choice is disclosed rather than hidden.  Legacy's approximately 20 values were
also selected on this dataset, so the all-30 in-sample comparison is fair, but
neither method's operating point is externally validated.

The result supports replacing the crop-specific deleafing consensus stack with
a compact geometric selector on this benchmark.  It does not establish a
universal harvesting replacement: harvesting has a different annotation side,
multi-target DBSCAN components, and 12/54 oracle targets without retained
contact support.  Fresh-scene or potted-plant data is required before the
deleafing threshold or the post-hoc boundary selector can be claimed to
transfer.  Method development stops at this result.

## Closed negative results

| Hypothesis or mechanism | Measured outcome | Decision |
|---|---|---|
| two-pass local drift | Local scales were below the global scale; at `k=256`, components grew from 469 to 566 for harvesting and 515 to 573 for deleafing, while F1 fell from 0.0650 to 0.0132 and 0.1641 to 0.1452. | The global drift correction is already conservative; do not localize it. |
| nearest approach | Under GT-adjacent masks, raw recall was 0.3704 harvesting and 0.7736 deleafing; the 1/5/10/25% robust sweep had opposite task trends and no common plateau.  It recovered none of the 12 anchorless harvesting GTs. | Kernel decay was not the main failure; drop the localizer. |
| multiplicative action fusion | On deleafing oracle components, hand distance alone, `m_peak` times action likelihood, and prominence times action likelihood all obtained top-1 92/106; hand alone and `m_peak` fusion selected the same peak in all 105 evaluable components. | The improvement is entirely the hand prior, not geometric fusion. |
| velocity/dwell event segmentation | On five deleafing scenes, the primary event counts were 5/10/7/9/9 versus GT counts 4/4/4/4/3; 0/5 counts were exact and 0/5 stable across reporting floors. | Tracking coverage does not imply event separability; do not build global event assignment. |
| revealed-new-surface cue | Reverse-change support was nonzero in only 25/515 leaf and 14/469 fruit components; true/false development medians and IQRs were all zero.  The strict empty-match version was zero everywhere. | The proposed revealed-surface signal is too sparse to reject components. |
| reciprocal matching | Exact reciprocal fractions were 0.0797% true versus 0.1528% false for harvesting and 0.00763% versus 0.441% for deleafing. | Rates are near zero and ordered in the wrong direction; this instead documents 2DGS representation non-uniqueness. |
| component-score products | `struct(c) * prominence` reached deleafing F1 0.5227 versus 0.5223 for prominence alone; hand-rank times removed-mass-rank reduced held-out overall F1 from 0.4767 for hand alone to 0.4316. | Products add no robust cross-task benefit; no learned or hand-built vote is introduced. |
| post-side retained field | The audited fruit sockets remained unsupported or weak in the post reconstruction, and the post-side peak reformulation did not repair harvesting. | Paired reconstruction does not remove the measured anchor limit in this dataset. |
| prominence as a universal selector | Fixed held-out top-1 prominence halves harvesting TP from 14 to 7; true prominence is lower than false prominence on harvesting but higher on deleafing. | Retain prominence as a continuous diagnostic/deleafing cue, not a universal rejection mechanism. |
| pre-structure continuity gate | It passes 97.2% of valid deleafing peaks but also 60.9% of other peaks; adding it leaves held-out oracle top-1 selection unchanged and removes only one end-to-end false positive. | It is nonspecific and unnecessary in the primary pipeline. |

These null results are part of the contribution: they rule out plausible
parameter-free repairs and delimit precisely which evidence is supplied by
contact geometry, action proximity, Stage-2 change masks, and the testbed.

## Artifact index

- `contact_mass_runs/phase1/interim_results.md`: frozen Phase-1 checkpoint.
- `contact_mass_runs/diagnostics_dbscan_score_audit/deleafing_tau_0.5_clusters.md`:
  required 28-row S1–S6 table.
- `contact_mass_runs/diagnostics_dbscan_score_audit/score_summary.csv`: ranks,
  margins, and precision-at-recall results for both tasks and both tau values.
- `contact_mass_runs/diagnostics_dbscan_score_audit/legacy_cluster_mapping.csv`:
  raw-to-stored membership comparison.
- `contact_mass_runs/diagnostics_dbscan_score_audit/legacy_primary_proposal_mapping.csv`:
  selected legacy output correspondence.
- `contact_mass_runs/diagnostics_peak_reformulation/diagnostics.md`: two-scene
  peak report and metric comparison.
- `contact_mass_runs/diagnostics_peak_reformulation/deleafing_pr_curve.csv` and
  `harvesting_pr_curve.csv`: complete local-peak-mass PR curves.
- `contact_mass_runs/diagnostics_peak_reformulation/deleafing_rho_peaks.png` and
  `harvesting_rho_peaks.png`: retained-side rho renderings with local peaks,
  globally surviving peaks, GT annotations, and the old centroid.
- `contact_mass_runs/diagnostics_peak_reformulation/gt_signal_refinement.md`:
  GT-centered side measurements, harvesting displacement directions, and
  local-refinement metrics; its original A1 interpretation is superseded.
- `contact_mass_runs/diagnostics_peak_reformulation/gt_signal_measurements.csv`,
  `harvesting_peak_directions.csv`, and `refinement_metrics.csv`:
  machine-readable forms of the follow-up diagnostic.
- `contact_mass_runs/diagnostics_peak_reformulation/post_side_diagnostic.md`:
  pre/post GT coverage, validated post confidence reconstruction, and the
  post-retained-side peak comparison.
- `contact_mass_runs/diagnostics_peak_reformulation/post_side_coverage.csv`,
  `post_side_peaks.csv`, and `post_side_metrics.csv`: machine-readable post-side
  results.
- `contact_mass_runs/diagnostics_all_gt_signal/diagnostics.md`: all-60-scene
  retained-proximity summary; its original observability interpretation is
  superseded by the annotation-side correction.
- `contact_mass_runs/diagnostics_all_gt_signal/per_gt_signal.csv`: all 160
  per-GT values, including retained counts/mass, rho mass, `sigma`, and nearest
  retained distance.
- `contact_mass_runs/diagnostics_all_gt_signal/task_summary.csv`: task-level
  retained-proximity counts and distances, retained for numerical provenance.
- `contact_mass_runs/diagnostics_mirror_interface/diagnostics.md`: all-scene
  E3-3 mirror, fair legacy comparison, interface-pair thickness, and measured
  annotation-side summary.
- `contact_mass_runs/diagnostics_mirror_interface/per_gt_mirror.csv`: per-GT
  mirror ranks, raw/refined errors, pair endpoints and distances, and nearest
  retained/removed measurements.
- `contact_mass_runs/diagnostics_mirror_interface/mirror_pr_curve.csv`: complete
  task-level mirror peak-mass precision-recall sweeps.
- `contact_mass_runs/diagnostics_truss_status/diagnostics.md`: all-scene A--D
  truss status, retained-neighbor semantics, mirror orientation, and mock-
  testbed density contrast.
- `contact_mass_runs/diagnostics_truss_status/truss_per_gt.csv` and
  `truss_distance_bins.csv`: per-GT and per-distance-bin truss-corridor
  measurements.
- `contact_mass_runs/diagnostics_truss_status/nearest20_retained_classes.csv`
  and `mirror_matched_orientation.csv`: all B/C semantic classes, positions,
  colors, distances, and direction vectors.
- `contact_mass_runs/diagnostics_truss_status/density_per_gt.csv`,
  `density_per_scene.csv`, and `density_paired_scenes.csv`: D at target, scene,
  and paired-scene levels.
- `contact_mass_runs/diagnostics_applicability/diagnostics.md`: status/reach
  cross-tab, pooled density and distance curves, and ivy-risk findings.
- `contact_mass_runs/diagnostics_applicability/applicability_curves.png`: match
  rate against unified retained density and nearest-retained distance, with
  95% Wilson intervals and monotone fits.
- `contact_mass_runs/diagnostics_applicability/per_gt_applicability.csv`,
  `density_bins.csv`, `nearest_distance_bins.csv`, and
  `density_task_comparison.csv`: complete applicability measurements.
- `contact_mass_runs/diagnostics_applicability/ivy_supported_matches.csv`:
  per-instance ivy errors, scores, ranks, pair distances, and filter survival.
- `contact_mass_runs/full_scale_167f9b1/results.json` and `results.md`: frozen
  all-60-scene head-to-head, abstention, transfer context, and proximity audit.
- `contact_mass_runs/full_scale_167f9b1/per_scene_head_to_head.csv`: requested
  per-scene P/R/F1 inputs and per-task side convention.
- `contact_mass_runs/full_scale_167f9b1/pair_gap_pr_curve.csv`,
  `bidirectional_pr_curve.csv`, and `m_peak_pr_curve.csv`: all three complete
  rejection curves in their required order; matching PNGs contain plots.
- `contact_mass_runs/full_scale_167f9b1/candidates.csv` and
  `per_gt_proximity.csv`: per-candidate scores/labels and paired GT-centered /
  GT-free proximity measurements.
- `contact_mass_runs/caches_replay_167f9b1/`: isolated all-60 continuous replay
  cache and the five-scene/two-shard fidelity manifests.
- `contact_mass_runs/diagnostics_drift_prominence_167f9b1/diagnostics.md`:
  prominence distributions, PR summaries, operating points, ranks, component
  histograms, and extent diagnostics.
- `contact_mass_runs/diagnostics_drift_prominence_167f9b1/drift_diagnostics.md`:
  depth-layer change channels, local/global drift, reciprocal matching, and
  `struct(c)` over all scenes; adjacent CSV/PNG files contain the complete
  machine-readable tables and curves.
- `contact_mass_runs/diagnostics_local_drift_167f9b1/all60/results.md`: CPU
  two-pass local-drift sweep, frozen-control regression, component/metric and
  adjacency tables, post-fix `struct(c)` modes, and the deleafing joint curve.
- `contact_mass_runs/diagnostics_local_drift_167f9b1/all60/npz/`: 60 isolated
  derived caches containing local fields and continuous pre/post geometry,
  appearance, and combined delta channels for all five `k` values.
- `contact_mass_runs/oracle_masks_e478073/results.md`: dual-oracle contact and
  nearest-approach gate; adjacent CSV files contain all 320 per-GT comparisons,
  masks, peaks, nearest anchors, and the unreached-20 decomposition.
- `contact_mass_runs/hand_tracking_coverage_e478073/results.md`: 30-scene MPS
  manipulation-window coverage audit and phase-manifest mismatch.
- `contact_mass_runs/oracle_action_ranking_e478073/results.md`: resolved
  60-window MPS action-ranking gate; adjacent per-GT, per-peak, trajectory, and
  scene-mapping CSVs contain all S1--S5 results and the data-hygiene audit.
- `contact_mass_runs/geometry_gate_e478073/results.md`: B1--B6 hand/contact
  geometry gate with paired bootstrap intervals, sign tests, and per-GT output.
- `contact_mass_runs/event_gate_e478073/results.md`: five-scene GT-blind
  velocity/dwell event-count gate and dwell-candidate audit.
- `contact_mass_runs/feature_separability_e478073/results.md`: fixed-split
  all-984-component F1--F10 separability audit, held-out PR curves, correlation
  matrices, and end-to-end component-filter results.
- `contact_mass_runs/feature_separability_e478073/hand_prominence_results.md`:
  final fixed held-out component/peak experiment and rank-cutoff decision.
- `contact_mass_runs/feature_separability_e478073/hand_prominence_full_pr_curve.csv`
  and `hand_prominence_full_pr_curve.png`: complete hand-rank, prominence-rank,
  and sequential held-out operating curves.
- `contact_mass_runs/task_specific_filter_e478073/results.md`: fixed-split
  action-specific size, mean-score, total-score, and size-plus-mean component
  filters with component-level and end-to-end held-out metrics.
- `contact_mass_runs/task_specific_filter_e478073/all30_sweet_spot.md`: all-30
  in-sample size/mean threshold oracle, reporting component- and target-optimal
  operating points separately; adjacent PNGs contain the full F1 surfaces.
- `contact_mass_runs/task_specific_filter_e478073/one_peak_per_cluster.md`:
  all-30 and held-out top-1 `m_peak`/prominence evaluations plus the measured
  multi-GT-component structural ceiling.
- `contact_mass_runs/deleafing_peak_selection_e478073/mc_normalizations_all30_results.md`:
  explicitly in-sample all-30 `M_c` normalization sweep, constrained operating
  points, scene-bootstrap intervals, and near-optimum threshold-stability audit.
- `contact_mass_runs/deleafing_peak_selection_e478073/mc_normalizations_all30_pr_curves.png`:
  the four complete precision-recall curves with the GT-adjacent-component
  oracle marked; adjacent `mc_pr_curve_all30_*.csv` files contain every point.
- `contact_mass_runs/deleafing_peak_selection_e478073/final_boundary_ablation_results.md`:
  the all-30 one-sidedness/distance/product ablation and exact distance-definition
  sensitivity; adjacent CSV files contain selected peaks and every candidate score.
- `contact_mass_runs/deleafing_peak_selection_e478073/final_deleafing_results.md`:
  publication-ready primary/legacy/oracle comparison, adopted gate rationale,
  boundary ablation, parameter accounting, and closed negative-result ledger.
