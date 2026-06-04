/* avoidance-cascade-c: Detect and prevent avoidance cascades in ternary agents */
#ifndef AVOIDANCE_CASCADE_H
#define AVOIDANCE_CASCADE_H

#include <stddef.h>

/* Ternary action */
typedef enum {
    AC_AVOID  = -1,
    AC_UNKNOWN = 0,
    AC_CHOOSE = 1
} ac_action_t;

/* Cascade detector */
#define AC_MAX_HISTORY 10000

typedef struct {
    double avoid_ratios[AC_MAX_HISTORY];
    size_t count;
    double threshold;
    int cascading;
    size_t cascade_rounds;
} cascade_detector_t;

void ac_detector_init(cascade_detector_t *d, double threshold);
int ac_detector_check(cascade_detector_t *d, const ac_action_t *actions, size_t n);
int ac_detector_is_cascading(const cascade_detector_t *d);
double ac_detector_current_ratio(const cascade_detector_t *d);

/* Balanced learner (v5 fix) */
typedef struct {
    double avg_reward_history[AC_MAX_HISTORY];
    double memory[256];  /* per-option reward memory */
    double decay;
    double margin;
    double forced_explore_rate;
    size_t memory_count;
    size_t explore_interval;
    size_t step_count;
} balanced_learner_t;

void ac_learner_init(balanced_learner_t *l, double decay, double margin,
                     double forced_explore_rate, size_t explore_interval);
ac_action_t ac_learner_decide(balanced_learner_t *l, size_t n_options,
                               const double *rewards);
void ac_learner_update(balanced_learner_t *l, size_t option, double reward);

/* Exploration scheduler */
typedef struct {
    double rate;
    double decay;
    double floor;
    size_t interval;
    size_t steps;
} exploration_scheduler_t;

void ac_scheduler_init(exploration_scheduler_t *s, double rate, double decay,
                       double floor, size_t interval);
int ac_scheduler_should_explore(exploration_scheduler_t *s);
void ac_scheduler_step(exploration_scheduler_t *s);

/* Cascade metrics */
typedef struct {
    size_t cascade_events;
    size_t recovery_events;
    size_t total_rounds;
    double total_avoid_ratio;
    double total_choose_ratio;
} cascade_metrics_t;

void ac_metrics_init(cascade_metrics_t *m);
void ac_metrics_record(cascade_metrics_t *m, int is_cascading,
                       double avoid_ratio, double choose_ratio);
double ac_metrics_avg_avoid_ratio(const cascade_metrics_t *m);
double ac_metrics_avg_choose_ratio(const cascade_metrics_t *m);

#endif /* AVOIDANCE_CASCADE_H */
