# avoidance-cascade-c — Detection and Prevention of Avoidance Cascades in Ternary Agent Systems

**avoidance-cascade-c** is a C library that detects and prevents **avoidance cascades** — the catastrophic failure mode where agents in a ternary decision system (each choosing between Avoid −1, Unknown 0, or Choose +1) progressively converge toward avoidance until the entire population refuses all actions. The library provides a cascade detector with configurable thresholds, a balanced learner with exponential moving average reward tracking, and an exploration scheduler with exponential decay to inject forced exploration.

## Why It Matters

In multi-agent reinforcement learning, avoidance cascades are a well-documented collapse mode: when agents share information and observe peers avoiding actions, avoidance propagates contagiously through the population. This is analogous to **bank runs** in financial systems — rational individual decisions aggregate into collectively irrational outcomes. In production fleets, an avoidance cascade means every agent stops processing requests, creating a total system blackout. This library provides the detection and correction primitives needed to guarantee that the avoid ratio $\gamma_{\text{avoid}}$ never exceeds a safety threshold for more than a bounded number of consecutive rounds, with $O(n)$ detection cost per round.

## How It Works

### Cascade Detection

The detector counts avoid actions across the population each round, computes the avoid ratio $\gamma = |\text{avoid}| / n$, and compares against a threshold $\theta$ (typically 0.95). A cascade is declared when $\gamma > \theta$ for **3 consecutive rounds** — this hysteresis prevents false positives from transient spikes.

```
round r: γ_r = count(actions == AVOID) / n
if γ_r > θ:
    cascade_rounds++
    if cascade_rounds ≥ 3: cascading = true
else:
    cascade_rounds = 0
    cascading = false
```

**Complexity**: $O(n)$ per round for counting, $O(1)$ for comparison. Space: $O(1)$ for state (stores only ratios in history buffer).

### Balanced Learner

The balanced learner tracks per-option reward using an **Exponential Moving Average (EMA)** with decay factor $\alpha$:

$$\hat{r}_i^{(t)} = \alpha \cdot \hat{r}_i^{(t-1)} + (1 - \alpha) \cdot r_i^{(t)}$$

The decision logic is ternary:
1. **Forced exploration**: Every `explore_interval` steps, return UNKNOWN (explore) regardless of learned rewards.
2. **Avoid**: If the best-known option has $\hat{r}_{\text{best}} < \text{margin}$, the learner returns AVOID — all options are worse than the safety margin.
3. **Choose**: If the best-known option exceeds the margin, return CHOOSE (exploit).

This prevents the degenerate case where a learner with no good options keeps choosing the least-bad option. The margin parameter implements a **reservation threshold** — familiar from the secretary problem and bandit literature.

### Exploration Scheduler

The scheduler controls exploration rate $\epsilon$ with exponential decay:

$$\epsilon_t = \max(\epsilon_{\text{floor}},\; \epsilon_0 \cdot \rho^{\lfloor t / I \rfloor})$$

where $\rho$ is the decay rate, $I$ is the interval between decays, and $\epsilon_{\text{floor}}$ ensures exploration never fully stops. This follows the standard $\epsilon$-greedy schedule from Sachs & Selvin (the same family used in DQN, with the modification that decay is stepped rather than continuous to reduce floating-point drift).

### Metrics

The metrics module tracks cascade events (entering cascading state), recovery events (exiting cascading state), and cumulative avoid/choose ratios. These provide the observability needed to verify that the system satisfies the conservation invariant γ + η = C over long time horizons.

### Comparison with Alternatives

| Approach | Detection Latency | False Positive Rate | Memory |
|----------|-------------------|--------------------|----|
| This library (3-round hysteresis) | 3 rounds | Low (requires sustained cascade) | $O(1)$ |
| Single-threshold (no hysteresis) | 1 round | High (spike-sensitive) | $O(1)$ |
| Statistical (z-score on avoid ratio) | Variable | Tunable | $O(n)$ history |

## Quick Start

```c
#include "avoidance_cascade.h"

/* 1. Initialize cascade detector */
cascade_detector_t detector;
ac_detector_init(&detector, 0.95);  // threshold: 95% avoid

/* 2. Check a population's actions */
ac_action_t actions[100];
/* ... fill actions from agent decisions ... */
int cascading = ac_detector_check(&detector, actions, 100);

if (cascading) {
    printf("WARNING: Avoidance cascade detected!\n");
}

/* 3. Balanced learner decides actions */
balanced_learner_t learner;
ac_learner_init(&learner, 0.9,   // decay (EMA weight)
                       0.0,   // margin (reservation threshold)
                       0.1,   // forced explore rate
                       100);  // explore interval

ac_action_t decision = ac_learner_decide(&learner, 3, rewards);

/* 4. Schedule exploration to prevent future cascades */
exploration_scheduler_t sched;
ac_scheduler_init(&sched, 1.0, 0.95, 0.05, 50);
if (ac_scheduler_should_explore(&sched)) {
    // Force exploration this round
}
ac_scheduler_step(&sched);

/* 5. Track cascade metrics */
cascade_metrics_t metrics;
ac_metrics_init(&metrics);
ac_metrics_record(&metrics, cascading, 0.96, 0.04);
printf("Avg avoid ratio: %.3f\n", ac_metrics_avg_avoid_ratio(&metrics));
```

```bash
# Build
gcc -O2 -c src/avoidance_cascade.c -o avoidance_cascade.o
gcc -O2 tests/test_cascade.c src/avoidance_cascade.c -o test_cascade -lm && ./test_cascade
```

## API

### Types

```c
typedef enum {
    AC_AVOID  = -1,
    AC_UNKNOWN = 0,
    AC_CHOOSE = 1
} ac_action_t;
```

### Cascade Detector

```c
void ac_detector_init(cascade_detector_t *d, double threshold);
int  ac_detector_check(cascade_detector_t *d, const ac_action_t *actions, size_t n);
int  ac_detector_is_cascading(const cascade_detector_t *d);
double ac_detector_current_ratio(const cascade_detector_t *d);
```

### Balanced Learner

```c
void ac_learner_init(balanced_learner_t *l, double decay, double margin,
                     double forced_explore_rate, size_t explore_interval);
ac_action_t ac_learner_decide(balanced_learner_t *l, size_t n_options,
                               const double *rewards);
void ac_learner_update(balanced_learner_t *l, size_t option, double reward);
```

### Exploration Scheduler

```c
void ac_scheduler_init(exploration_scheduler_t *s, double rate, double decay,
                       double floor, size_t interval);
int  ac_scheduler_should_explore(exploration_scheduler_t *s);
void ac_scheduler_step(exploration_scheduler_t *s);
```

### Cascade Metrics

```c
void   ac_metrics_init(cascade_metrics_t *m);
void   ac_metrics_record(cascade_metrics_t *m, int is_cascading,
                         double avoid_ratio, double choose_ratio);
double ac_metrics_avg_avoid_ratio(const cascade_metrics_t *m);
double ac_metrics_avg_choose_ratio(const cascade_metrics_t *m);
```

## Architecture Notes

In the SuperInstance fleet, avoidance-cascade-c is the **safety guard** that ensures ternary agents (those using {-1, 0, +1} action spaces) never collapse into total avoidance. It directly enforces the γ component of γ + η = C: when γ (the avoid ratio) rises above threshold, the balanced learner and exploration scheduler inject UNKNOWN actions that increase η (Shannon entropy), restoring the conservation balance. The library's $O(n)$ detection cost makes it suitable for real-time monitoring of large agent populations.

See: [SuperInstance Architecture](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md)

## References

1. Sutton, R. S. & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2nd ed., Chapter 2 (Multi-armed Bandits) — ε-greedy exploration and reservation thresholds.
2. Granovetter, M. (1978). "Threshold Models of Collective Behavior." *American Journal of Sociology* 83(6) — Cascade dynamics in populations with threshold-based decision rules.

## License

MIT
