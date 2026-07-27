# Experiment 3 report: sink-state ablation

## Status

Full GPU matrix not yet run. No result or significance claim should be made from
smoke tests. Populate this report from `results.json` after status is `complete`.

## Protocol

Main historical pixel protocol: Multi-Room, 84x84 RGB, horizon 300, 50,000 reward-free
interactions, gamma 0.99, replay minibatch 1,024, operator batch 5,000, actor update
every 1,500 interactions, 250 PMD steps, and seeds 1-5. Partial-support cross:
operator batch 1,024 for no-sink and paper-schedule conditions only. All remaining
hyperparameters are frozen. Time-limited design uses one pretraining seed per cell
and three independent adaptation seeds: 8 pretraining plus 24 adaptation runs.
Encoder replay is episode-balanced sampling with replacement. Operator anchors are
sampled uniformly without replacement from all accumulated transitions, with a
fixed initial-state anchor.

Exact paper-run sink schedule:

```text
epsilon(t) = 0.004 * clip(t / 50000, 0, 1).
```

Historical run configs support this value. Later `rover_paper.yaml` refactor says
`0.05 * clip(t / 50000, 0, 1)`; this does not match historical pixel configs.
"Close to one" is qualitative motivation for sink penalty, not numeric run value.
Sensitivity results resolve selection toward small epsilon under finite support.

## Metrics

Report mean, standard error, and paired seed differences for final-policy coverage,
largest consecutive-round visitation regression, first downstream reward, success
AUC, estimated sink mass, and low-support operator-query fraction. Primary interaction
test compares paired sink-minus-no-sink effects between batch 1,024 and batch 5,000.

## Results

Pending full matrix.

## Restricted conclusion

Pending. Conclusion must address only whether ROVER's learned operator suffers
coverage expansion-collapse under incomplete fitting support and whether sink
augmentation stabilizes its downstream policy initialization. It must not attribute
collapse in RND, APT, SMM, CIC, or unrelated intrinsic-reward methods to sink absence.

## Reviewer response drafts

### Reviewer 1vr1 Q2

We added a direct pixel-based sink-state ablation under the same ROVER protocol as
the main Multi-Room experiment. We compare epsilon=0, fixed small/intermediate
values, the historical schedule, and a deliberately large negative control, using
shared seeds and both the main 5,000-sample operator batch and a 1,024-sample
partial-support batch. We report final-policy coverage, round-wise coverage
regression, downstream time-to-reward/AUC, estimated sink mass, and low-support
query frequency. [Insert completed paired results.]

### Reviewer u2jK Q4

The exact historical schedule is epsilon(t)=0.004 clip(t/50000,0,1). We found a
reproducibility-refactor discrepancy: the later aggregate agent YAML states a 0.05
endpoint. We now document both and use the historical schedule for the main-setting
ablation. The paper's "close to one" wording describes the conceptual penalty for
unsupported occupancy, not the selected numeric scale; finite-sample sensitivity
instead favors small epsilon, especially at partial support. [Insert results.]

### Reviewer D9Z7 Q3

We now quantify the proposed failure mode directly: largest loss of visitation mass
on any previously covered state between consecutive policy-optimization rounds.
We also record the fraction of state-action operator queries with fewer than two
supporting samples. The paired batch interaction tests whether sink benefit increases
when support is incomplete. [Insert completed regression and interaction estimates.]

### Reviewer LSRN Q1

The added experiment uses pixel observations and downstream online SAC adaptation,
not only the tabular diagnostic. All non-sink settings and random seeds are matched.
Our conclusion is deliberately narrow: it tests stabilization of ROVER's
operator-based coverage update. It does not claim sink absence explains collapse in
other intrinsic-reward algorithms. [Insert completed coverage and adaptation AUC.]
