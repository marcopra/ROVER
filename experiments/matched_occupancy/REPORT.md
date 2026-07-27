# Experiment 1 report

## Status

**NOT RUN.** Infrastructure and preregistered matching rule are implemented. Candidate
checkpoints and buffers are not present in this checkout, so matched pairs and
statistical evidence do not yet exist.

## Methods

Five-room MultiRooms; fixed one-hot states; ROVER and CIC candidates; replay-buffer
support/distribution measured from stored transitions; final-policy support and
discounted occupancy measured from 100 rollouts at gamma 0.99. Pairing is blind to
downstream results: replay support gap ≤5% (target ≤2%) and final effective-support gap
≥10% of state count. Downstream goals span rooms 1, 3, and 5 with identical seeds 1–5.
DDPG adaptation holds architecture, transferred encoder, optimizer, rollout horizon,
pre-update collection budget, evaluation schedule, and 60k-interaction budget fixed.

Metrics: zero-shot success, interactions to first reward, success before first update,
success AUC, and interactions to 80% success. Pearson and Spearman correlations compare
buffer support and final effective occupancy against success AUC; 95% bootstrap
intervals use 10,000 resamples.

## Results

Pending. `analyze.py` produces `matched_pair_table.csv`, `results.json`,
`coverage_vs_adaptation.png`, and `per_goal_learning_curves.png`.

## Limitations

Pair existence is empirical and not guaranteed. MultiRooms is finite-state and does not
establish continuous-control scalability. Checkpoint comparisons may retain
algorithm-specific pretraining differences; within-pair downstream controls reduce but
do not remove this confound. Five adaptation seeds give limited correlation power.

## Exact claim

No claim currently supported. Claim may be stated only if frozen matched pairs exist,
final-occupancy/adaptation association exceeds replay-support association, and confidence
intervals exclude a practically null contrast.

## Paper-ready reviewer paragraph

Reviewer u2jK Q1 asks why a covering final policy matters when a diverse buffer exists;
this preregistered experiment directly holds buffer support approximately fixed and tests
online rewards that cannot be retrospectively relabeled. Reviewer D9Z7 Q2 asks whether
pixel gains come from encoder or policy transfer; fixed one-hot inputs isolate behavioral
policy transfer while downstream architecture and encoder handling remain controlled.
Reviewer LSRN Q3 asks whether coverage consistently predicts faster learning across goal
distributions; identical near, intermediate, and distant goals and five seeds test this
with first-reward, success-AUC, threshold-time, and correlation estimates. **Results are
pending, so this paragraph makes no positive empirical claim.**
