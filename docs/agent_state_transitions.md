# Agent State Transitions

This document visualizes the runtime modes of the trading agent and supporting
services (tests-when-closed and learner), plus the healthwatch scheduler that
starts/stops services around market hours.

## Trading Agent Runtime

```mermaid
stateDiagram-v2
    [*] --> Booting
    Booting --> MarketClosedIdle: init complete
    MarketClosedIdle --> MarketOpenActive: market open or within start_before_minutes
    MarketOpenActive --> MarketClosedIdle: market closed

    MarketOpenActive --> RiskHalt: circuit breaker / var-cvar / exposure caps / limits
    RiskHalt --> MarketOpenActive: risk clears + cooldown

    MarketOpenActive --> KillSwitchSleep: kill_switch.force_sleep armed
    MarketClosedIdle --> KillSwitchSleep: kill_switch.force_sleep armed
    KillSwitchSleep --> MarketClosedIdle: force_sleep cleared

    MarketOpenActive --> Liquidating: kill_switch.force_liquidate armed
    Liquidating --> MarketClosedIdle: liquidation complete
```

## Healthwatch Market Scheduler

```mermaid
stateDiagram-v2
    [*] --> SchedulerStart
    SchedulerStart --> Running: market open or within start_before_minutes
    SchedulerStart --> WaitingReport: market closed + daily report pending
    SchedulerStart --> Sleeping: market closed + report done

    Running --> Sleeping: market closed + report done
    Running --> WaitingReport: market closed + report pending
    WaitingReport --> Sleeping: report done
    Sleeping --> Running: next_open within start_before_minutes
    Running --> ForceSleep: kill_switch.force_sleep armed
    ForceSleep --> Sleeping: force_sleep cleared
```

## Tests-When-Closed Service

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> RunningTests: market closed
    RunningTests --> RunningBacktest: backtest.run_when_closed true
    RunningBacktest --> RunningBenchmarks: benchmarking.run_when_closed true
    RunningTests --> Idle: sleep interval
    RunningBacktest --> Idle: sleep interval
    RunningBenchmarks --> Idle: sleep interval
    Idle --> Idle: market open (skip)
```

## Learner Service (Online Updates)

```mermaid
stateDiagram-v2
    [*] --> LearnerIdle
    LearnerIdle --> OnlineTraining: learning.online.enabled true
    OnlineTraining --> Evaluate: update steps complete
    Evaluate --> LearnerIdle: interval sleep
```
