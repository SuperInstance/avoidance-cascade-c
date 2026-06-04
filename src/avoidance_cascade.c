/* avoidance-cascade-c: Implementation */
#include "avoidance_cascade.h"
#include <string.h>
#include <math.h>

/* --- Cascade Detector --- */

void ac_detector_init(cascade_detector_t *d, double threshold) {
    memset(d, 0, sizeof(*d));
    d->threshold = threshold;
}

int ac_detector_check(cascade_detector_t *d, const ac_action_t *actions, size_t n) {
    if (d->count >= AC_MAX_HISTORY) return d->cascading;
    size_t avoid = 0;
    for (size_t i = 0; i < n; i++) {
        if (actions[i] == AC_AVOID) avoid++;
    }
    double ratio = (double)avoid / (double)n;
    d->avoid_ratios[d->count++] = ratio;
    
    if (ratio > d->threshold) {
        d->cascade_rounds++;
        if (d->cascade_rounds >= 3) {
            d->cascading = 1;
        }
    } else {
        d->cascade_rounds = 0;
        d->cascading = 0;
    }
    return d->cascading;
}

int ac_detector_is_cascading(const cascade_detector_t *d) {
    return d->cascading;
}

double ac_detector_current_ratio(const cascade_detector_t *d) {
    if (d->count == 0) return 0.0;
    return d->avoid_ratios[d->count - 1];
}

/* --- Balanced Learner --- */

void ac_learner_init(balanced_learner_t *l, double decay, double margin,
                     double forced_explore_rate, size_t explore_interval) {
    memset(l, 0, sizeof(*l));
    l->decay = decay;
    l->margin = margin;
    l->forced_explore_rate = forced_explore_rate;
    l->explore_interval = explore_interval;
}

ac_action_t ac_learner_decide(balanced_learner_t *l, size_t n_options,
                               const double *rewards) {
    if (n_options == 0) return AC_UNKNOWN;
    
    /* Increment step counter */
    size_t current_step = l->step_count++;

    /* Forced exploration */
    if (current_step > 0 && current_step % l->explore_interval == 0) {
        return AC_UNKNOWN;
    }
    
    /* Find best option by average reward + margin */
    double best_score = -1e9;
    size_t best_idx = 0;
    size_t known = 0;
    
    for (size_t i = 0; i < n_options && i < 256; i++) {
        if (l->memory[i] != 0.0) {
            known++;
            if (l->memory[i] > best_score) {
                best_score = l->memory[i];
                best_idx = i;
            }
        }
    }
    
    l->step_count++;
    
    /* If best score is below threshold, avoid */
    if (known > 0 && best_score < l->margin) {
        return AC_AVOID;
    }
    
    /* If we know the options, choose best */
    if (known > 0) {
        return AC_CHOOSE;
    }
    
    /* Otherwise explore */
    return AC_UNKNOWN;
}

void ac_learner_update(balanced_learner_t *l, size_t option, double reward) {
    if (option >= 256) return;
    /* Exponential moving average with decay */
    if (l->memory[option] == 0.0) {
        l->memory[option] = reward;
    } else {
        l->memory[option] = l->decay * l->memory[option] + (1.0 - l->decay) * reward;
    }
    l->memory_count++;
}

/* --- Exploration Scheduler --- */

void ac_scheduler_init(exploration_scheduler_t *s, double rate, double decay,
                       double floor, size_t interval) {
    s->rate = rate;
    s->decay = decay;
    s->floor = floor;
    s->interval = interval;
    s->steps = 0;
}

int ac_scheduler_should_explore(exploration_scheduler_t *s) {
    int result = (s->steps % s->interval == 0) && (s->rate > s->floor);
    return result;
}

void ac_scheduler_step(exploration_scheduler_t *s) {
    s->steps++;
    if (s->steps % s->interval == 0) {
        s->rate *= s->decay;
        if (s->rate < s->floor) s->rate = s->floor;
    }
}

/* --- Cascade Metrics --- */

void ac_metrics_init(cascade_metrics_t *m) {
    memset(m, 0, sizeof(*m));
}

void ac_metrics_record(cascade_metrics_t *m, int is_cascading,
                       double avoid_ratio, double choose_ratio) {
    if (is_cascading) {
        m->cascade_events++;
    } else if (m->cascade_events > 0 && !is_cascading) {
        m->recovery_events++;
    }
    m->total_rounds++;
    m->total_avoid_ratio += avoid_ratio;
    m->total_choose_ratio += choose_ratio;
}

double ac_metrics_avg_avoid_ratio(const cascade_metrics_t *m) {
    if (m->total_rounds == 0) return 0.0;
    return m->total_avoid_ratio / (double)m->total_rounds;
}

double ac_metrics_avg_choose_ratio(const cascade_metrics_t *m) {
    if (m->total_rounds == 0) return 0.0;
    return m->total_choose_ratio / (double)m->total_rounds;
}
