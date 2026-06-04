# avoidance-cascade-c

C implementation of avoidance cascade detection and prevention in ternary agent systems.

Cross-language companion to [avoidance-cascade](https://github.com/SuperInstance/avoidance-cascade) (Rust).

## API

- `cascade_detector_t` — monitors avoidance ratio, detects cascade (>95% threshold, 3 rounds)
- `balanced_learner_t` — v5 fix: average reward with margin, forced exploration, memory decay
- `exploration_scheduler_t` — periodic forced exploration with decay and floor
- `cascade_metrics_t` — track cascade events, recovery, ratios

## Build & Test

```bash
gcc -o test_cascade tests/test_cascade.c src/avoidance_cascade.c -lm -Wall -O2
./test_cascade
```

16 tests passing. No external dependencies.

## License

MIT
