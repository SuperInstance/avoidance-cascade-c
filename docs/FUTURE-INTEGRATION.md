# Future Integration: avoidance-cascade-c

## Current State
C implementation of avoidance cascade detection and prevention for ternary agents. Detects when avoidance (-1 state) begins cascading through a population, models spread dynamics, identifies tipping points, and simulates interventions.

## Integration Opportunities

### With ternary-cell (Embedded Cascade Detection)
The C port enables real-time cascade detection on microcontrollers. A `CellGcStrategy` using this library runs on ESP32 alongside the ternary cell engine. When the `gc` phase starts pruning cells, the cascade detector monitors the avoidance rate. If it crosses the tipping point, the GC strategy switches from aggressive to conservative.

### With compiled-policy-c
Avoidance cascade detection IS a safety policy. Compile the cascade detector into a policy that runs on every edge device. The policy triggers when avoidance exceeds threshold — a zero-dependency early warning system for fleet instability.

### With ternary-failure-c
Cascade detection feeds failure analysis. An avoidance cascade is a `FailureMode::Critical` event. The C port enables failure analysis on-device without network connectivity — critical for isolated ESP32 nodes.

## Potential in Mature Systems
In room-as-codespace, the C port runs on edge hardware (ESP32, Jetson) that monitors local room populations. When a cascade starts, the edge device can intervene immediately — no round-trip to PLATO required. This is the fast path: local detection, local response, global reporting.

## Cross-Pollination Ideas
- Tipping point detection as a room health metric — how close is this room to cascade failure?
- C port enables cross-language room operations: Rust room logic calls C cascade detection via FFI
- Intervention simulation for testing cascade recovery strategies offline

## Dependencies for Next Steps
- FFI bindings for Rust integration (ternary-cell calls C cascade detector)
- Integration with compiled-policy-c for policy deployment
- Embedded testing on ESP32 hardware
