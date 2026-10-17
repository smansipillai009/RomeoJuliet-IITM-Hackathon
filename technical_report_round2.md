# Technical Report (Round 2): Allocation Under Scarce Supply for Sequential Reciprocal Matching

**Team:** LaLaLand · **Kit:** The Sequential Matching Problem 1.0.0 (synthetic data only)
**Main outcome:** completed meeting and mutual second-meeting intention (MSMI). Alongside it: calibration and who is left waiting.

## 1. Summary

We built a standard-library policy that (i) asks for hard constraints in order of expected feasible pairs unlocked, never for members who declined one; (ii) scores feasible pairs with a probability model learned from our own rollouts; and (iii) chooses introductions by exact maximum-weight matching. Every ask and pair is re-validated against the public contract, and any internal error degrades to "wait".

What the evidence supports, in order of strength:

1. **Clarification is the only large effect.** Removing it cuts expected MSMI by about 64% (0.946 to 0.339).
2. **Our policy gives a small gain over the supplied baseline, about +3%.** Pooled over 24 paired episodes (four variants), expected MSMI is 0.902 against 0.877 (+0.025 [+0.009, +0.042]). Individual comparisons are noisy, and in 10 development seeds alone the baseline's point estimate was slightly higher.
3. **A simple configuration matches the full one.** List-order asks, exact matching and an agreement count score 0.899, indistinguishable from the shipped policy (+0.003 [−0.012, +0.017]). The gain over the baseline therefore most likely comes from exact allocation, not from learned scoring or smarter asking. The pooled `simple` vs `starter` comparison isolates the matcher: both use list-order asks, and they differ only in exact maximum-weight matching on an agreement-count weight (with a base value per edge) versus the starter's greedy agreement selection. That difference is +0.022 [+0.012, +0.034] in expected MSMI (Section 6). It is the best-supported single source of the gain, but it is one analysis path over 24 paired episodes, and the 10-seed development ablation alone (greedy instead of exact, −0.024 [−0.052, +0.004]) does not exclude zero.
4. **Learned scoring helps when there is more to learn and more to choose from:** denser pools and stronger preference signal.
5. **Realised MSMI is dominated by luck** (about one event per episode), so we compare policies on expected MSMI computed from hidden true probabilities (analysis only) and cross-check on realised MSMI and the official evaluator.

Headroom is small by construction. The baseline already introduces 82–92% of the feasible pairs that exist, and about 35% of members cannot be served at all.

## 2. Changes from the Round 1 proposal

| Proposed | What happened |
|---|---|
| H1: value-ordered asks bring coverage forward | Tested: same cost as list order, coverage +0.002, no detectable gain. Kept as the ask rule (never asks declined members), reported as neutral |
| H2: soft asks on spare budget | Tested: no reliable gain at the official size; cost +90 units. Kept because the member score in denser pools uses them |
| H3: exact allocation | Implemented and verified. Likely the source of the small gain |
| H4: probability-weighted edges and member propensities | Implemented. Member history looked weak in cross-validation (pair AUC 0.623 to 0.629) but was best in dense pools. A second-meeting model on soft fields scored at chance (AUC 0.52) |
| H5: cross-variant prior and rate monitoring | **Not built.** Per-variant differences were small relative to noise, and simple scoring matched learned scoring, so added complexity was not justified by the evidence |
| Waiting threshold | **Not built**, for the same reason |

## 3. Method as built

- **Feasibility and preferences.** Only `feasible` pairs are proposed, so hard constraints and stated preferences are never overridden; the score only ranks pairs that already pass `eligibility()`. Declined fields are never re-asked or bypassed.
- **Asks.** Constraint bundles (3 units) ranked by expected feasible pairs unlocked; spare budget buys soft fields (1 unit) for members already in a feasible pair, restricted to fields with non-trivial learned weight.
- **Edge score (default `member`).** Logistic model on 7 soft-field agreement features plus shrunk member response and yes rates computed from feedback visible in the request (no policy memory). Options kept: `fit`, `agree`, `uniform`.
- **Allocation.** Exact maximum-weight matching in a general graph (component-wise exact search, deterministic state cap, greedy fallback); verified against `networkx` with zero weight gap on 184 real daily states, slowest call under 2 ms.
- **Safety and isolation.** A final guard removes any invalid ask or pair. Analysis code that reads hidden simulator state lives in `analysis/`; two tests fail the build if `policy.py` references hidden names or imports it. A third audit checks every introduction against members' true hard constraints after the fact: it passes across four variants.
- **Training.** Weights are learned from declared rollouts (random-feasible exploration, development variant only): 30 episodes (seeds 3001–3030) for the agreement model, 120 episodes (5001–5120, 21,846 directional rows) for the member model. No simulator parameter is encoded in the policy.

## 4. Official evaluator (seeds 101–103, six variants, 18 episodes; all valid)

| Policy | Primary (MSMI/100) | Coverage | Mutual acc./100 | Mean ask cost | MSMI events (of 18) |
|---|---:|---:|---:|---:|---:|
| Greedy baseline | 0.389 | 0.372 | 5.50 | 206.7 | 14 |
| Random feasible | 0.278 | 0.377 | 5.33 | 206.7 | 10 |
| No clarification | 0.139 | 0.143 | 1.75 | 0 | 5 |
| **Submitted** | 0.361 | 0.377 | 6.03 | 283.5 | 13 |

This is a smoke comparison; 13 vs 14 events is well inside Poisson noise. Total inference per 120-call episode was at most 15.8 s, far below the per-call 10 s limit.

### 4.1 Scenario-level outcomes (same seeds 101–103, 3 episodes per scenario)

| Scenario | Policy | MSMI/100 | MSMI events (3 ep.) | Coverage | Mutual acc./100 | Ask cost | Missing feedback | Mean first-intro wait (days, served) | Unserved members | Max inference (s/episode) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| development | Greedy | 0.500 | 3 | 0.432 | 6.50 | 200.0 | 60.0 | 6.0 | 113.7 | 12.0 |
| development | Random feasible | 0.333 | 2 | 0.440 | 6.50 | 200.0 | 54.7 | 6.1 | 112.0 | 12.1 |
| development | No clarification | 0.167 | 1 | 0.183 | 2.17 | 0.0 | 18.0 | 4.3 | 163.3 | 12.8 |
| development | Submitted | 0.500 | 3 | 0.437 | 8.00 | 288.3 | 50.7 | 5.5 | 112.7 | 14.0 |
| sparse | Greedy | 0.167 | 1 | 0.145 | 2.33 | 196.0 | 8.3 | 6.2 | 171.0 | 12.6 |
| sparse | Random feasible | 0.167 | 1 | 0.145 | 2.17 | 196.0 | 8.3 | 6.1 | 171.0 | 13.0 |
| sparse | No clarification | 0.000 | 0 | 0.043 | 0.50 | 0.0 | 2.0 | 4.2 | 191.3 | 13.2 |
| sparse | Submitted | 0.000 | 0 | 0.145 | 1.17 | 207.3 | 6.0 | 4.4 | 171.0 | 15.8 |
| cold_start | Greedy | 0.167 | 1 | 0.360 | 4.67 | 244.0 | 30.3 | 8.1 | 128.0 | 12.0 |
| cold_start | Random feasible | 0.167 | 1 | 0.363 | 4.50 | 244.0 | 31.7 | 8.4 | 127.3 | 12.1 |
| cold_start | No clarification | 0.000 | 0 | 0.080 | 1.17 | 0.0 | 4.7 | 3.6 | 184.0 | 12.9 |
| cold_start | Submitted | 0.000 | 0 | 0.373 | 4.17 | 330.3 | 33.7 | 7.5 | 125.3 | 14.5 |
| delayed | Greedy | 0.500 | 3 | 0.430 | 6.83 | 200.0 | 55.3 | 6.5 | 114.0 | 11.7 |
| delayed | Random feasible | 0.167 | 1 | 0.438 | 6.83 | 200.0 | 50.0 | 6.4 | 112.3 | 11.3 |
| delayed | No clarification | 0.333 | 2 | 0.183 | 2.50 | 0.0 | 19.0 | 4.3 | 163.3 | 12.6 |
| delayed | Submitted | 0.833 | 5 | 0.433 | 8.17 | 290.7 | 47.0 | 5.3 | 113.3 | 13.8 |
| shift | Greedy | 0.500 | 3 | 0.432 | 6.50 | 200.0 | 58.7 | 6.0 | 113.7 | 11.8 |
| shift | Random feasible | 0.500 | 3 | 0.433 | 5.83 | 200.0 | 52.0 | 6.2 | 113.3 | 11.9 |
| shift | No clarification | 0.167 | 1 | 0.183 | 2.00 | 0.0 | 18.3 | 4.1 | 163.3 | 12.6 |
| shift | Submitted | 0.333 | 2 | 0.438 | 7.17 | 296.0 | 51.0 | 5.6 | 112.3 | 14.2 |
| drift | Greedy | 0.500 | 3 | 0.432 | 6.17 | 200.0 | 60.3 | 6.0 | 113.7 | 11.9 |
| drift | Random feasible | 0.333 | 2 | 0.440 | 6.17 | 200.0 | 54.7 | 6.1 | 112.0 | 11.6 |
| drift | No clarification | 0.167 | 1 | 0.183 | 2.17 | 0.0 | 18.0 | 4.3 | 163.3 | 12.6 |
| drift | Submitted | 0.500 | 3 | 0.437 | 7.50 | 288.3 | 51.3 | 5.5 | 112.7 | 14.0 |

Reading notes. (a) The submitted policy's overall mean first-introduction wait is 5.76 days against 6.45 for greedy, with 124.6 and 125.7 unserved members per episode. The no-clarification row has the shortest wait (4.17 days) only because it serves far fewer members, so waits are not comparable across policies without the unserved counts. (b) Missing feedback (non-responses) averages 39.9 per episode for the submitted policy, 45.5 for greedy and 13.3 for no clarification, and tracks the number of introductions. (c) The submitted policy recorded 0 MSMI events in the cold_start and sparse scenarios (greedy: 1 event each) and the highest count in delayed (5 vs 3). With 3 episodes per scenario, none of these per-scenario differences is distinguishable from luck; Sections 5 and 6 use expected MSMI over more seeds for that reason. (d) Ask cost is higher for the submitted policy (about 284 units against 207) because it spends spare budget on soft fields; cost is only a late tie-break in the ranking, and we report it as a cost of the method, not a benefit.

## 5. The three questions from the Round 1 feedback

### 5.1 Cold-start decrease: timing, selection or model error?

**It was luck.** Round 1 showed cold_start MSMI 0.48 events per episode lower for our policy. On 40 fresh cold_start seeds (4001–4040) the realised gap reversed. This measurement used the Round 1 policy (score `fit`):

| | Baseline | Policy | Difference (95% paired bootstrap) |
|---|---:|---:|---|
| Realised MSMI | 0.525 | 0.850 | +0.325 [+0.025, +0.625] |
| Expected MSMI (luck removed) | 0.712 | 0.748 | **+0.035 [+0.014, +0.055]** |
| Introductions | 71.5 | 73.0 | +1.5 [−0.18, +3.28] |

- **Design.** Each introduction's true success probability is computed from hidden state by quadrature over the pair-level draw, validated against Monte Carlo (p_mutual 0.0702 vs 0.0722; p_msmi 0.00336 vs 0.00347). Realised minus expected is outcome luck.
- **Decomposition of the expected gain:** volume +0.015, selection +0.020; luck (realised − expected) was −0.19 for the baseline and +0.10 for the policy, with per-episode SD about 0.7.
- **Timing:** mean assignment day 21.0 vs 19.9, not an explanation.
- **Selection:** true P(mutual) and true P(MSMI) rank pairs almost identically (Spearman 0.93), so ranking by acceptance does not select for worse second meetings.
- **Model error is the open part:** the policy score ranks true P(MSMI) with Spearman 0.46.

### 5.2 When do learned funnel scores beat simple scores?

Asking and allocation fixed; only the edge score varies. Fresh populations, 8 paired seeds per cell, weights trained only at signal ×1, n=200. Expected MSMI per episode for uniform, and differences versus uniform:

| Population | Uniform | Simple agree | Learned fit | Member model | Per-day true-P reference |
|---|---:|---|---|---|---|
| n=200, signal ×0 | 0.785 | −0.026 | +0.018 | +0.005 | +0.002 |
| n=200, signal ×1 | 0.972 | +0.028 | +0.014 | +0.044 | +0.049 |
| n=200, signal ×2 | 1.160 | +0.029 | +0.062 | **+0.092** [+0.011, +0.170] | +0.099 |
| n=100, signal ×1 | 0.325 | +0.011 [+0.003, +0.018] | +0.008 | +0.008 | **−0.012** [−0.026, −0.000] |
| n=300, signal ×1 | 1.709 | **+0.089** [+0.021, +0.163] | **+0.084** [+0.015, +0.148] | **+0.145** [+0.089, +0.194] | +0.143 |

- With no preference signal or very sparse pools, no score helps.
- At n=300 the member model is about +8.5% over uniform and reaches the per-day true-probability reference.
- **The reference is not an upper bound:** at n=100 it does worse than uniform, because when feasible pairs are scarce, using more of them beats choosing the best.
- At the official size (n=200, 12 paired seeds per variant), differences in expected MSMI versus `fit`:

| Variant | member | agree |
|---|---|---|
| development | +0.013 [−0.009, +0.035] | +0.009 [−0.036, +0.048] |
| shift | +0.009 [−0.016, +0.034] | +0.021 [−0.021, +0.058] |
| drift | +0.015 [+0.001, +0.028] | +0.003 [−0.030, +0.030] |
| cold_start | −0.005 [−0.027, +0.023] | +0.006 [−0.022, +0.036] |

We ship `member` as a judgement call: never clearly worse, best in denser pools.

**What the comparison taught us.** At the official size the simple unlearned score matches the learned one, so most of the learned model's value there is "use soft-field agreement". Learning matters only with stronger signal and denser pools. Member history looked weak in cross-validation but was best in dense pools; our untested explanation is more observed history per member there.

### 5.3 Directional calibration and unequal service

**Calibration.** Model for P(actor says yes), grouped-by-seed cross-validation on 21,846 rows (development): ECE 0.0037, Brier 0.2229 vs 0.2298 for a constant; the product of the two directions has ECE 0.0057 against realised mutual acceptance. On other variants (trained on development, scored on fresh episodes):

| Variant | Directional ECE | Brier (constant) | Predicted vs observed yes | Pair product vs observed mutual |
|---|---:|---:|---|---|
| shift | 0.0294 | 0.2201 (0.2233) | 0.354 vs 0.337 | 0.131 vs 0.132 |
| drift | 0.0106 | 0.2201 (0.2264) | 0.356 vs 0.346 | 0.133 vs 0.136 |

Calibration degrades modestly under shift (directional yes is over-predicted) while pair-level aggregates stay close. We did not attempt recalibration.

**Who gets left waiting.** Servable members only (arrived, no declined hard field, at least one truly feasible partner; this label uses hidden truth, analysis only).
Paired intervals, shipped policy minus baseline, 16 fresh development seeds (1,220 servable members each):

| Measure | Baseline | Submitted | Difference [95% CI] |
|---|---:|---:|---|
| Never served | 2.8% (34) | 2.0% (24) | −0.63 per episode [−1.38, 0.00] |
| Mean wait (days) | n/a | n/a | **−1.04 [−1.41, −0.68]** |
| Wait Gini | n/a | n/a | **+0.026 [+0.011, +0.041]** |

Typical members wait about a day less, while the distribution of waits becomes slightly more unequal. The never-served share is borderline, and an earlier 10-seed run showed it rising (2.4% to 2.9%), so we treat it as noise. A 10-seed run across variants (first-cut table in `results/equity.json`) shows the same shorter-mean-wait pattern in cold_start and shift. Group tables (gender, age band, zone) show no consistent worst-served group, but non-binary, zone_c and zone_d have fewer than 100 servable members in that run and cannot be read. No subgroup significance tests were run.

## 6. Ablation on the final policy (development, 10 fresh seeds 7301–7310)

Difference in expected MSMI per episode versus the shipped policy (0.946):

| Cell | Expected MSMI | Difference [95% CI] | Ask cost |
|---|---:|---|---:|
| Starter-equivalent (list-order asks, greedy agreement) | 0.970 | +0.024 [−0.002, +0.052] | 193 |
| A1 no clarification | 0.339 | **−0.607 [−0.702, −0.517]** | 0 |
| A2 list-order asks | 0.946 | −0.001 [−0.040, +0.041] | 193 |
| A3 hard asks only | 0.935 | −0.012 [−0.030, +0.003] | 193 |
| A4 greedy instead of exact matching | 0.923 | −0.024 [−0.052, +0.004] | 284 |
| A5a score = uniform | 0.967 | +0.021 [−0.002, +0.043] | 282 |
| A5b score = agree | 0.963 | +0.016 [−0.007, +0.040] | 289 |
| A5c score = fit | 0.949 | +0.003 [−0.036, +0.040] | 287 |

Only clarification matters. All other differences are within about 3% and not distinguishable from zero in these 10 seeds.

**Pooled across variants** (6 fresh seeds × development, shift, drift, cold_start = 24 paired episodes; seeds 7101–7106): starter 0.877, simple 0.899, agree 0.900, shipped 0.902. Shipped − starter +0.025 [+0.009, +0.042]; simple − starter +0.022 [+0.012, +0.034]; agree − starter +0.023 [+0.002, +0.047]; shipped − simple +0.003 [−0.012, +0.017].

**Sparse** (10 seeds): shipped 0.235; starter +0.002 [−0.003, +0.008]; agree +0.002 [−0.004, +0.009]. **Delayed** (10 seeds): shipped 0.799; starter −0.011 [−0.058, +0.031]; agree +0.032 [−0.002, +0.068]. In sparse pools about 99% of feasible pairs are used by any policy, so no difference is expected. For delayed, our expected-MSMI calculation ignores the 30-day date window, so it can overstate expected values in that variant and the delayed expected-MSMI figures above are indicative only. Realised MSMI events per episode in the same 10 delayed seeds (`results/ablation_delayed.json`) were 0.9 for the shipped policy, 1.2 for the starter-equivalent and 0.2 for the `agree` score; at about one event per episode these realised counts cannot rank the cells either.

## 7. Why headroom is small

- About 35% of members have a declined hard field; many others have no feasible partner. In 5 development episodes (hidden truth, analysis only): 66 declined-hard, 57 no feasible partner, 77 with a partner, 75.6 of those served.
- Official evaluator, development seeds 101–106: 108, 96, 95, 100, 60 and 74 assignments against 125, 116, 103, 122, 69 and 85 feasible edges at day 60 (82–92% used; 86% pooled). The count visible to a policy equals the oracle count in all six episodes. On two other sets of eight seeds the share was 79%.
- MSMI is about one event per episode; a 120-episode mean has a standard error near 9% of the mean (Poisson assumption).

## 8. Limitations

- Weights are trained on the development variant only; shift and drift transfer is measured, not trained for.
- Most comparisons use 6–12 paired seeds, so intervals are wide; the pooled test has 24 paired episodes and one analysis path.
- Expected-MSMI comparisons use hidden simulator internals (analysis only); every claim is cross-checked on realised MSMI and the official evaluator, which cannot resolve differences this small.
- The Section 5.1 decomposition measured the Round 1 policy (score `fit`), not the shipped default.
- Calibration is out-of-fold on exploration-policy data, not on the deployed policy's own distribution, and is weaker under shift.
- **Container status.** On 17 October 2026, in GitHub Codespaces (Docker 29.8.0), `docker build -t sequential-policy:submission .` succeeded from the unchanged `Dockerfile` (base `python:3.12.7-slim`, copying only `kit.py` and `policy.py`), and the full test suite passed there (34 tests OK). The official evaluator was then run against the built image: `python evaluate.py --image sequential-policy:submission --seeds 101 --output results/container_check.json`. That run covers one episode (seed 101, development variant): valid, MSMI 1 event (0.5 per 100 arrived members), coverage 0.465, mutual acceptances 10.0 per 100, ask cost 310. These figures are identical to the same seed and variant run without the container (Section 4), so the image reproduces the in-process behaviour for that episode. Total inference time in the container was 64.6 s for the episode, against 14.0 s outside it; the extra time is container start-up on each call (about 0.54 s per call averaged over the episode's 120 calls), well inside the 10 s per-call limit, although we did not record the slowest single call. The container check covers only that one episode; we did not run the other five variants through the image.
- Realised MSMI is about one event per episode, so no per-scenario row in Section 4.1 supports a performance claim on its own.
- The wait statistics in Sections 4.1 and 5.3 describe first-introduction waits of served members; the hidden-truth servable-member labels in Section 5.3 are analysis only.

## 9. Reproducibility

`reproduce.sh` runs tests, the official evaluator for baselines and the submitted policy, training, and every experiment above, in order, with approximate run times on one core. The policy needs only the standard library (`Dockerfile` unchanged). Analysis tooling is pinned in `requirements-analysis.txt` (numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, networkx 3.6.1). Seeds: baselines and edge usage 101–106; training 3001–3030, 3001–3180, 5001–5120; cold-start decomposition 4001–4040; score grid 6001–6008; default-score and first-cut equity 7001–7012 and 7001–7010; pooled 7101–7106; equity intervals 7201–7216; ablation 7301–7310, sparse 7401–7410, delayed 7501–7510; calibration 8001–8025. Training seeds are disjoint from every evaluation seed. All 34 tests pass (22 kit, 9 policy, 2 hidden-isolation, 1 hard-constraint audit).

### 9.1 Environment, randomness and compute

- **Python and container.** Policy container: `python:3.12.7-slim`, copying only `kit.py` and `policy.py`; no network, GPU or third-party package is needed at inference. Local development used Python 3.12.
- **Randomness.** The submitted policy uses no random number generator, so there is no inference random seed; its action stream is a deterministic function of the request (`action_sha256` is recorded per episode in the results JSON).
- **Training compute.** Weights were fitted offline on one CPU core, no GPU: 30 exploration episodes (3001–3030) for the agreement model and 120 (5001–5120, 21,846 directional rows) for the member model. Approximate wall-clock times for each step are in the comments of `reproduce.sh` (for example about 10 minutes for the 120-episode collection). Trained weights are embedded as constants in `policy.py`, so the container loads no model files.

### 9.2 Provenance and licences

| Item | Source and use | Licence |
|---|---|---|
| Kit code and data | Organiser starter kit 1.0.0, synthetic records only | Kit `LICENSE` (MIT) and `DATA_LICENSE.md` (additional provenance conditions on the supplied records) |
| Training and exploration data | Generated by our own simulator rollouts; no external datasets, no pretrained models | Our own project licence (`LICENSE`) |
| Python standard library | Policy runtime | PSF License |
| numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, networkx 3.6.1 | Analysis and training only; not used by the policy or container | BSD-3-Clause each (as published by the projects; confirm with the installed package metadata) |
| `python:3.12.7-slim` | Container base image | Python licence for the interpreter; Debian packages carry their own licences |
| Claude (Anthropic) | Coding and analysis assistant, declared in Section 10 | Not redistributed; no model weights are included |

## 10. Tooling declaration

Claude (Anthropic) was used as a coding and analysis assistant for implementation, experiments and drafting. All reported numbers come from runs of the code in this archive.
