#!/usr/bin/env python3
"""Standalone CLI Runner for EdgeSync Fault Injection Suite (Tier 1 Simulation - WP-8).

Executes Scenarios D, F, G, and Quality Gate 8 conformance tests.
"""
import sys
import traceback
from edgesync_coordination.tests.test_fault_injection import (
    test_scenario_d_candidate_crash_and_watchdog_cutoff,
    test_scenario_f_network_loss_and_confirmation_retries_success,
    test_scenario_f_total_network_severance_and_claim_reclaim,
    test_scenario_g_stale_epoch_rejection,
    test_quality_gate_8_zero_dual_ownership_invariant_under_random_loss
)


def run_scenario(name: str, test_fn):
    print(f"[*] Running {name} ... ", end="", flush=True)
    try:
        test_fn()
        print("PASSED ✅")
        return True
    except Exception as e:
        print("FAILED ❌")
        traceback.print_exc()
        return False


def main():
    print("=" * 70)
    print("🚀 EdgeSync Tier 1 Simulation & Fault Injection Harness (WP-8)")
    print("Aligned with SRS v1.3, SAD v2.3, & Quality Gate 8")
    print("=" * 70)

    tests = [
        ("Scenario D: Candidate Crash & 300ms Watchdog Cutoff", test_scenario_d_candidate_crash_and_watchdog_cutoff),
        ("Scenario F (Part 1): Network Loss with 3x Retries within T_ack", test_scenario_f_network_loss_and_confirmation_retries_success),
        ("Scenario F (Part 2): Total Network Severance & T_claim Reclaim (e+2)", test_scenario_f_total_network_severance_and_claim_reclaim),
        ("Scenario G: Stale Epoch e-1 Rejection & Version Mismatch Gate", test_scenario_g_stale_epoch_rejection),
        ("Quality Gate 8: 50-Trial Property-Based Stress Test (Zero Dual-Ownership)", test_quality_gate_8_zero_dual_ownership_invariant_under_random_loss),
    ]

    all_passed = True
    for name, fn in tests:
        if not run_scenario(name, fn):
            all_passed = False

    print("=" * 70)
    if all_passed:
        print(">>> ALL FAULT INJECTION SCENARIOS PASSED 100%! QUALITY GATE 8 MET! <<<")
        print(">>> ZERO DUAL-OWNERSHIP VIOLATIONS CONFIRMED. PROTOCOL ROBUST. <<<")
        return 0
    else:
        print(">>> SOME SCENARIOS FAILED. CHECK TRACEBACK ABOVE. <<<")
        return 1


if __name__ == '__main__':
    sys.exit(main())
