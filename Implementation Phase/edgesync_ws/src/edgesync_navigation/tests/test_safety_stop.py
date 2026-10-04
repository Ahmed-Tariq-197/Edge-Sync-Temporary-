"""Unit tests for Safe-Stop & Motion Safety (Quality Gate 7).

Verifies:
1. Immediate software safe-stop braking on lost localization / emergency.
2. 150 ms command watchdog cutoff activation.
3. Smooth deceleration ramp-down to zero velocity.
"""
import pytest
from edgesync_navigation.core.safety_supervisor import SafetySupervisor


def test_immediate_safe_stop_trigger():
    supervisor = SafetySupervisor(max_linear_decel=2.0, max_angular_decel=3.0)

    # Initial motion: moving forward at 0.6 m/s, turning at 1.0 rad/s
    vx, vy, wz = 0.6, 0.0, 1.0

    supervisor.trigger_safe_stop("Obstacle detected or heartbeat lost")
    assert supervisor.is_stopped is True

    # First cycle of deceleration (dt = 0.05s)
    cmd_vx, cmd_vy, cmd_wz, stopped = supervisor.filter_command(
        req_vx=0.6, req_vy=0.0, req_wz=1.0,
        current_vx=vx, current_vy=vy, current_wz=wz,
        dt=0.05
    )

    assert stopped is True
    assert cmd_vx < vx  # Decelerated
    assert cmd_wz < wz


def test_velocity_ramp_down_to_zero():
    supervisor = SafetySupervisor(max_linear_decel=2.0, max_angular_decel=2.0)
    supervisor.trigger_safe_stop("Emergency test")

    cur_vx = 0.5
    cur_wz = 0.5
    dt = 0.05

    # Simulate braking loop until stopped
    for _ in range(20):
        cur_vx, _, cur_wz, stopped = supervisor.filter_command(
            0.5, 0.0, 0.5, cur_vx, 0.0, cur_wz, dt=dt
        )

    # Must be completely stopped at zero
    assert pytest.approx(cur_vx, abs=1e-5) == 0.0
    assert pytest.approx(cur_wz, abs=1e-5) == 0.0


def test_command_watchdog_150ms_cutoff():
    supervisor = SafetySupervisor(cmd_vel_watchdog_ms=150.0)
    now = 100.0
    supervisor._last_valid_command_time = now

    # At 100ms: within 150ms limit -> NOT stopped
    _, _, _, stopped = supervisor.filter_command(0.4, 0.0, 0.0, 0.4, 0.0, 0.0, current_time=now + 0.100)
    assert stopped is False

    # At 160ms: exceeds 150ms -> WATCHDOG CUTOFF ENGAGED!
    _, _, _, stopped = supervisor.filter_command(0.4, 0.0, 0.0, 0.4, 0.0, 0.0, current_time=now + 0.160)
    assert stopped is True
    assert "watchdog" in supervisor.stop_reason.lower()
