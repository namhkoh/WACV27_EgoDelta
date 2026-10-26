# Mechanics-conditioned Stage-3 target extraction

## Method

EgoDelta represents an intervention by an interface pair. The retargeting
primitive already chooses which endpoint the tool acts on, so Stage 3 routes by
the a-priori endpoint-side convention rather than by crop identity:

1. **Retained endpoint (sever/cut).** The retained structure persists after the
   action. What distinguishes the target from other retained structure is
   contact with the removed material; a retained-side contact field is therefore
   required. We detect its modes, refine them locally, and select the mode that
   is both adjacent to and one-sided with respect to the removed component.
2. **Removed endpoint (detach/grasp payload).** The object itself is present in
   the pre-action reconstruction, and the target is its own attachment feature.
   Its geometry and orientation suffice. Dense removed-object modes seed an
   exhaustive nearest-core partition, followed by a locally refined
   gravity-facing surface point.

Both decoders start from the same continuous Stage-2 change score and DBSCAN
component proposal. Routing uses only the endpoint side already stored by the
tool/retargeting primitive. It does not use task identity, crop identity, color,
PCA endpoints, crop-specific shape rules, or a consensus vote.

The world `+Z` direction in the removed-side decoder is a physical prior for
hanging fruit. It will fail on horizontally trained trusses and must be checked
on potted plants before making a general horticultural claim.

## Final all-scene results

All operating thresholds are in-sample over 30 scenes per task. The legacy
extractor's approximately 20 Table-6 parameters were also selected on this
dataset, making this an in-sample comparison rather than fresh-scene validation.

**In-sample disclosure.** We used the same 30 scenes per task for method
development and final reporting. Inspection of these scenes selected the
one-sidedness-times-Gaussian boundary-selector form, the deleafing normalized
removed-mass threshold `M_c/N_overlap=0.0037746118219953335`, the harvesting
component mean-change threshold `0.9008104936197888` and component-size
threshold `205`, the harvesting dense-mode `min_cluster_size=120`, and the use
of a scene-relative Tukey density fence `Q3+1.5 IQR`. The legacy extractor's
approximately 20 Table-6 parameters were likewise fitted on this benchmark;
accordingly, comparisons are in-sample for both methods, and validation on
fresh scenes and potted plants remains outstanding.

| Task | Method | Precision | Recall | F1 | Median matched error | TP / FP / FN |
|---|---|---:|---:|---:|---:|---:|
| Deleafing | mechanics-conditioned geometry | **91.59%** | **92.45%** | **92.02%** | **1.060 cm** | 98 / 9 / 8 |
| Deleafing | legacy | 90.91% | 84.91% | 87.80% | 1.182 cm | 90 / 9 / 16 |
| Harvesting | mechanics-conditioned geometry | **94.44%** | **94.44%** | **94.44%** | **1.395 cm** | 51 / 3 / 3 |
| Harvesting | legacy | 90.74% | 90.74% | 90.74% | 1.584 cm | 49 / 5 / 5 |

Parameter counts under the frozen reporting convention are 3 for the retained
decoder and 6 for the removed decoder, versus approximately 20 in legacy. The
removed decoder adds two fitted component-adequacy constants and the fitted
`min_cluster_size=120`. The shared three are the sigma multiplier, `tau_c`, and
the normalized removed-mass gate. DBSCAN `eps=0.02 m, min_samples=20` is the
unchanged standard/legacy instance proposal and is listed as inherited rather
than silently folded into either decoder's count. Physical/domain conventions
are reported separately.

| Class | Retained decoder | Removed decoder |
|---|---|---|
| Fitted | sigma multiplier 2; `tau_c=0.5`; normalized removed-mass threshold | the retained/shared three, plus component mean-delta threshold; component-size threshold; mode minimum-cluster size |
| Inherited instance proposal | DBSCAN `eps=0.02 m`, `min_samples=20` | same |
| Derived / conventional | scene median nearest-neighbor scale | scene median nearest-neighbor scale; scene Tukey density upper fence |
| Domain / mechanics | 3 cm tool/evaluation resolution | gravity direction; removed endpoint chosen by detach/grasp primitive |

## Cross-application establishes necessity

Each task's selected component gate was held fixed while only the geometric
decoder was swapped.

| Evaluation task | Endpoint decoder | Precision | Recall | F1 | Median error |
|---|---|---:|---:|---:|---:|
| Deleafing | retained contact | 91.59% | 92.45% | **92.02%** | 1.060 cm |
| Deleafing | removed density/surface | 0.10% | 0.94% | **0.18%** | 2.958 cm |
| Harvesting | retained contact | 16.13% | 9.26% | **11.76%** | 1.563 cm |
| Harvesting | removed density/surface | 94.44% | 94.44% | **94.44%** | 1.395 cm |

The 11.76% and 0.18% cross-task F1 scores show a definitional difference, not
a tuning difference. Retained contact identifies an interface on persistent
support; removed density identifies an attachment feature on the removed
payload.

## Proximity is an applicability diagnostic, not a router

Two distance statistics appear in the analysis and must not be conflated:

- **GT-centered distance:** distance from the annotated GT point to its nearest
  retained primitive. Harvesting has a 4.152 cm median. This evaluation-only
  statistic produced the earlier applicability curve: recovery stayed at least
  90% through 0.769 cm and at least 50% through 4.594 cm.
- **Component-boundary distance:** minimum distance over *all* removed
  primitives in a GT-adjacent DBSCAN component to any pre-action primitive with
  positive retained weight. This is computable without GT. Its median is 0.084
  cm for deleafing and 1.281 cm for harvesting.

The second number is smaller because it takes a minimum over the entire changed
component rather than measuring at the annotated endpoint. Its task IQRs do not
overlap—deleafing 0.052--0.141 cm and harvesting 0.163--2.561 cm—but the full
ranges do overlap. The binned decoder-success curves cross around 0.93 cm,
whereas the in-sample threshold optimum is 0.388--0.431 cm. A threshold switch
there succeeds on 128/138 GT-adjacent components (92.8%), below the a-priori
mechanics convention at 134/138 (97.1%). Proximity therefore supports the
geometric interpretation but is not used to choose the decoder.

## Recurring retained-neighbor contamination

Nearest-retained measurements are repeatedly contaminated by background ivy:

- approximately 35% of nearest-retained support was classified as ivy;
- 6/34 mirror matches were ivy-supported background coincidences;
- replacing world-up with the direction to the nearest retained primitive
  reduced harvesting F1 to 77.78% and oracle-partition F1 to 68.52%;
- low component-boundary distances incorrectly route several fruit components
  under the experimental proximity switch.

This is one consistent observation: nearest arbitrary scene geometry is not a
reliable proxy for the intervention support. It explains why retained-facing
orientation and proximity routing are omitted.

## Closed negative results

| Mechanism | Measured failure and cause | Decision |
|---|---|---|
| Local drift | At `k=256`, component counts increased from 515/469 to 573/566 (leaf/fruit); deleafing F1 fell from 0.1641 to 0.1452. Local low-delta neighborhoods underestimated the conservative global drift rather than removing drift artifacts. | Drop. |
| Nearest approach | Oracle-component recall was 77.36% for deleafing and 37.04% for harvesting. The closest retained primitive is often ivy/background, and no percentile plateau appeared. | Drop. |
| Multiplicative fusion | `m_peak x action` and `prominence x action` did not improve the selected oracle leaf count beyond the action cue alone. Correlated evidence cannot repair missing/incorrect modes. | Drop. |
| Event segmentation | Velocity/dwell recovered 5/10/7/9/9 events against 4/4/4/4/3 GTs; 0/5 scene counts were correct. Multi-action windows are not reliably separable by this signal. | Drop. |
| Revealed-surface evidence | Nonzero in only 25/515 leaf and 14/469 fruit components; true and false development medians and IQRs were zero. Reverse support is too sparse. | Drop. |
| Reciprocal matching | Absolute reciprocity was near zero and reversed (true/false leaf 0.00763%/0.441%). Independently optimized 2DGS primitives are not one-to-one correspondences. | Drop. |
| Absence depth | 71.2--94.1% of component primitives were capped by the search radius. The feature was censored and its true/false ordering was opposite the hypothesis. | Drop. |
| Scene-relative quantiles | Best pooled component F1 was 53.57%; the supposedly structural quantiles remained dataset-selected and lost most deleafing recall. | Drop. |
| Footprint area / voxel extent | Physical extent avoided count inversion but remained weak on harvesting (best task-specific component F1 66.67%); disc overlap double-counts surface, while voxelization did not separate fruit from drift well enough. | Drop. |
| Mean delta alone | Deleafing end-to-end F1 fell to 79.35%. Small high-delta drift fragments look fully unexplained even though they are not removed modules. | Drop. |
| Retained-facing direction | Harvesting F1 fell to 77.78% and oracle F1 to 68.52% because the nearest retained primitive is frequently lateral ivy/background rather than truss support. | Drop. |
| Proximity switch | Task IQRs separate, but ranges overlap; the binned crossover (0.93 cm) conflicts with the threshold optimum (0.4 cm) and loses six component successes relative to mechanics routing. | Diagnostic only. |
| Prominence for within-component selection | Oracle top-1 deleafing F1 was 68.25%, below both `m_peak` and the boundary selector. Prominence identifies isolated ridge bumps, not the acted-on interface endpoint. | Diagnostic only. |

No further rejection or routing mechanism is introduced.

## Scope

These are discovery results from all 30 scenes per task, not untouched test
performance. Fresh scenes—especially potted plants, thin natural stems, and
horizontally trained trusses—are required to validate the thresholds, gravity
prior, and transfer beyond the mock testbed.

The full prediction-affecting constant audit, including inherited DBSCAN
settings and code-level legacy constants absent from the archived headline
set, is in `PARAMETER_AUDIT.md`. Reproduction commands and qualitative case
selection are in `REPRODUCTION.md` and `paper_assets/`, respectively.
