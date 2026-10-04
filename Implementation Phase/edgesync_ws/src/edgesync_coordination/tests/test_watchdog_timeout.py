"""Unit tests for Liveness Watchdog (Quality Gate 6).

Verifies:
1. Continuous feeds maintain armed watchdog without timeout.
2. Missing 3 heartbeats (300 ms) fires cutoff callback.
3. Disarming stops timeout triggers.
"""
from edgesync_coordination.protocol.liveness_watchdog import LivenessWatchdog


def test_watchdog_feeds_prevent_timeout():
    timeout_triggered = []

    def on_timeout():
        timeout_triggered.append(True)

    dog = LivenessWatchdog(heartbeat_period_ms=100.0, max_missed_heartbeats=3, on_timeout_callback=on_timeout)
    dog.arm()

    # Simulate 5 consecutive 10 Hz heartbeats arriving every 100 ms
    current_time = 1000.0
    for _ in range(5):
        current_time += 0.100
        dog.feed()
        assert dog.check(current_time=current_time) is False

    assert len(timeout_triggered) == 0


def test_watchdog_timeout_fires_at_300ms():
    timeout_triggered = []

    def on_timeout():
        timeout_triggered.append(True)

    dog = LivenessWatchdog(heartbeat_period_ms=100.0, max_missed_heartbeats=3, on_timeout_callback=on_timeout)
    dog.arm()

    current_time = 1000.0
    dog._last_heartbeat_time = current_time

    # Advance time by 200 ms (2 missed heartbeats -> no timeout yet)
    assert dog.check(current_time=current_time + 0.200) is False
    assert len(timeout_triggered) == 0

    # Advance time by 300 ms (3 missed heartbeats -> TIMEOUT TRIGGERED!)
    assert dog.check(current_time=current_time + 0.301) is True
    assert len(timeout_triggered) == 1
    assert dog.is_armed is False
