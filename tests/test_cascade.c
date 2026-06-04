/* Tests for avoidance-cascade-c */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "../src/avoidance_cascade.h"

#define ASSERT_FEQ(a, b, tol) do { \
    if (fabs((a) - (b)) > (tol)) { \
        fprintf(stderr, "FAIL %s:%d: %f != %f\n", __FILE__, __LINE__, (double)(a), (double)(b)); \
        return 1; \
    } \
} while(0)

int test_detector_no_cascade() {
    cascade_detector_t d;
    ac_detector_init(&d, 0.95);
    
    /* 50% avoidance — well below threshold */
    ac_action_t actions[100];
    for (int i = 0; i < 50; i++) actions[i] = AC_AVOID;
    for (int i = 50; i < 100; i++) actions[i] = AC_CHOOSE;
    
    int result = ac_detector_check(&d, actions, 100);
    assert(!result);
    assert(!ac_detector_is_cascading(&d));
    ASSERT_FEQ(ac_detector_current_ratio(&d), 0.5, 0.01);
    return 0;
}

int test_detector_cascade() {
    cascade_detector_t d;
    ac_detector_init(&d, 0.95);
    
    /* 98% avoidance — above threshold */
    ac_action_t actions[100];
    for (int i = 0; i < 98; i++) actions[i] = AC_AVOID;
    actions[98] = AC_UNKNOWN;
    actions[99] = AC_CHOOSE;
    
    /* Need 3 consecutive rounds to trigger */
    ac_detector_check(&d, actions, 100);
    assert(!ac_detector_is_cascading(&d));
    ac_detector_check(&d, actions, 100);
    assert(!ac_detector_is_cascading(&d));
    int result = ac_detector_check(&d, actions, 100);
    assert(result);
    assert(ac_detector_is_cascading(&d));
    return 0;
}

int test_detector_recovery() {
    cascade_detector_t d;
    ac_detector_init(&d, 0.95);
    
    /* Trigger cascade */
    ac_action_t high_avoid[100];
    for (int i = 0; i < 98; i++) high_avoid[i] = AC_AVOID;
    high_avoid[98] = AC_CHOOSE;
    high_avoid[99] = AC_CHOOSE;
    ac_detector_check(&d, high_avoid, 100);
    ac_detector_check(&d, high_avoid, 100);
    ac_detector_check(&d, high_avoid, 100);
    assert(ac_detector_is_cascading(&d));
    
    /* Recover with normal distribution */
    ac_action_t normal[100];
    for (int i = 0; i < 50; i++) normal[i] = AC_AVOID;
    for (int i = 50; i < 100; i++) normal[i] = AC_CHOOSE;
    ac_detector_check(&d, normal, 100);
    assert(!ac_detector_is_cascading(&d));
    return 0;
}

int test_learner_choose_best() {
    balanced_learner_t l;
    ac_learner_init(&l, 0.9, 0.0, 0.1, 100);
    
    /* Train: option 0 is bad, option 1 is good */
    double rewards[] = {-1.0, 1.0};
    ac_learner_update(&l, 0, -1.0);
    ac_learner_update(&l, 1, 1.0);
    
    ac_action_t decision = ac_learner_decide(&l, 2, rewards);
    assert(decision == AC_CHOOSE);  /* knows options, best is above margin */
    return 0;
}

int test_learner_avoid_all_bad() {
    balanced_learner_t l;
    ac_learner_init(&l, 0.9, 0.5, 0.1, 100);
    
    /* All options are bad */
    double rewards[] = {-0.8, -0.5, -0.9};
    ac_learner_update(&l, 0, -0.8);
    ac_learner_update(&l, 1, -0.5);
    ac_learner_update(&l, 2, -0.9);
    
    ac_action_t decision = ac_learner_decide(&l, 3, rewards);
    assert(decision == AC_AVOID);  /* best is -0.5, below margin 0.5 */
    return 0;
}

int test_learner_explore_unknown() {
    balanced_learner_t l;
    ac_learner_init(&l, 0.9, 0.0, 0.1, 100);
    
    double rewards[] = {1.0, 1.0};
    /* No memory yet */
    ac_action_t decision = ac_learner_decide(&l, 2, rewards);
    assert(decision == AC_UNKNOWN);
    return 0;
}

int test_learner_forced_exploration() {
    balanced_learner_t l;
    ac_learner_init(&l, 0.9, 0.0, 0.1, 5);  /* explore every 5 steps */
    
    double rewards[] = {1.0, 1.0};
    ac_learner_update(&l, 0, 1.0);
    ac_learner_update(&l, 1, 1.0);
    
    /* Steps 1-4: choose */
    for (int i = 0; i < 5; i++) {
        ac_action_t d = ac_learner_decide(&l, 2, rewards);
        assert(d == AC_CHOOSE);
    }
    /* Step 6: forced explore (step_count=5, 5%5==0) */
    ac_action_t d = ac_learner_decide(&l, 2, rewards);
    assert(d == AC_UNKNOWN);
    return 0;
}

int test_learner_memory_decay() {
    balanced_learner_t l;
    ac_learner_init(&l, 0.5, 0.0, 0.1, 100);
    
    /* Initial good memory */
    ac_learner_update(&l, 0, 1.0);
    ASSERT_FEQ(l.memory[0], 1.0, 0.01);
    
    /* Bad experience with decay */
    ac_learner_update(&l, 0, -1.0);
    /* Expected: 0.5 * 1.0 + 0.5 * (-1.0) = 0.0 */
    ASSERT_FEQ(l.memory[0], 0.0, 0.01);
    return 0;
}

int test_scheduler_periodic() {
    exploration_scheduler_t s;
    ac_scheduler_init(&s, 0.5, 0.9, 0.01, 3);
    
    /* Steps 0, 3, 6 should trigger exploration */
    assert(ac_scheduler_should_explore(&s));  /* step 0 */
    ac_scheduler_step(&s);
    assert(!ac_scheduler_should_explore(&s)); /* step 1 */
    ac_scheduler_step(&s);
    assert(!ac_scheduler_should_explore(&s)); /* step 2 */
    ac_scheduler_step(&s);
    assert(ac_scheduler_should_explore(&s));  /* step 3 */
    return 0;
}

int test_scheduler_decay() {
    exploration_scheduler_t s;
    ac_scheduler_init(&s, 1.0, 0.5, 0.1, 2);
    
    double initial_rate = s.rate;
    ac_scheduler_step(&s);
    ac_scheduler_step(&s);  /* triggers decay */
    assert(s.rate < initial_rate);
    ASSERT_FEQ(s.rate, 0.5, 0.01);
    return 0;
}

int test_scheduler_floor() {
    exploration_scheduler_t s;
    ac_scheduler_init(&s, 0.2, 0.5, 0.1, 1);
    
    /* Decay many times */
    for (int i = 0; i < 20; i++) {
        ac_scheduler_step(&s);
    }
    assert(s.rate >= 0.1);  /* floor */
    return 0;
}

int test_metrics_basic() {
    cascade_metrics_t m;
    ac_metrics_init(&m);
    
    ac_metrics_record(&m, 0, 0.5, 0.3);  /* normal */
    ac_metrics_record(&m, 1, 0.96, 0.02); /* cascade */
    ac_metrics_record(&m, 0, 0.6, 0.25);  /* recovery */
    
    assert(m.cascade_events == 1);
    assert(m.recovery_events == 1);
    assert(m.total_rounds == 3);
    ASSERT_FEQ(ac_metrics_avg_avoid_ratio(&m), (0.5 + 0.96 + 0.6) / 3.0, 0.01);
    ASSERT_FEQ(ac_metrics_avg_choose_ratio(&m), (0.3 + 0.02 + 0.25) / 3.0, 0.01);
    return 0;
}

int test_metrics_no_cascade() {
    cascade_metrics_t m;
    ac_metrics_init(&m);
    
    for (int i = 0; i < 10; i++) {
        ac_metrics_record(&m, 0, 0.5, 0.3);
    }
    assert(m.cascade_events == 0);
    assert(m.recovery_events == 0);
    ASSERT_FEQ(ac_metrics_avg_avoid_ratio(&m), 0.5, 0.01);
    return 0;
}

int test_metrics_empty() {
    cascade_metrics_t m;
    ac_metrics_init(&m);
    ASSERT_FEQ(ac_metrics_avg_avoid_ratio(&m), 0.0, 0.001);
    ASSERT_FEQ(ac_metrics_avg_choose_ratio(&m), 0.0, 0.001);
    return 0;
}

int test_cascade_death_spiral_simulation() {
    /* Simulate pure avoidance learning → should cascade to 100% avoid */
    cascade_detector_t d;
    ac_detector_init(&d, 0.95);
    
    for (int round = 0; round < 20; round++) {
        /* Each round, agents learn to avoid more */
        int avoid_count = 90 + round / 2;  /* 90, 90, 91, 91, ... */
        if (avoid_count > 100) avoid_count = 100;
        
        ac_action_t actions[100];
        for (int i = 0; i < avoid_count; i++) actions[i] = AC_AVOID;
        for (int i = avoid_count; i < 100; i++) actions[i] = AC_UNKNOWN;
        
        ac_detector_check(&d, actions, 100);
    }
    
    /* Should detect cascade */
    assert(ac_detector_is_cascading(&d));
    return 0;
}

int test_balanced_learner_prevents_cascade() {
    /* Balanced learner should NOT cascade to 100% avoid */
    balanced_learner_t l;
    ac_learner_init(&l, 0.9, 0.0, 0.1, 5);
    
    cascade_metrics_t m;
    ac_metrics_init(&m);
    
    int avoid_count = 0;
    for (int round = 0; round < 50; round++) {
        double rewards[] = {-0.5, 0.5};
        ac_action_t d = ac_learner_decide(&l, 2, rewards);
        
        /* Update with actual rewards */
        if (d == AC_AVOID) {
            ac_learner_update(&l, 0, -0.5);
            avoid_count++;
        } else if (d == AC_CHOOSE) {
            ac_learner_update(&l, 1, 0.5);
        }
        
        double avoid_ratio = (d == AC_AVOID) ? 1.0 : 0.0;
        double choose_ratio = (d == AC_CHOOSE) ? 1.0 : 0.0;
        ac_metrics_record(&m, 0, avoid_ratio, choose_ratio);
    }
    
    /* Should NOT have cascaded to 100% avoid */
    assert(avoid_count < 50);
    return 0;
}

int main(void) {
    int failures = 0;
    
    #define RUN(name) do { \
        int r = test_##name(); \
        if (r == 0) printf("  PASS: %s\n", #name); \
        else { printf("  FAIL: %s\n", #name); failures++; } \
    } while(0)
    
    printf("Running tests...\n");
    RUN(detector_no_cascade);
    RUN(detector_cascade);
    RUN(detector_recovery);
    RUN(learner_choose_best);
    RUN(learner_avoid_all_bad);
    RUN(learner_explore_unknown);
    RUN(learner_forced_exploration);
    RUN(learner_memory_decay);
    RUN(scheduler_periodic);
    RUN(scheduler_decay);
    RUN(scheduler_floor);
    RUN(metrics_basic);
    RUN(metrics_no_cascade);
    RUN(metrics_empty);
    RUN(cascade_death_spiral_simulation);
    RUN(balanced_learner_prevents_cascade);
    
    printf("\n%d/%d passed\n", 16 - failures, 16);
    return failures;
}
