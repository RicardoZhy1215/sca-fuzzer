#!/usr/bin/env bash
#
# perf_env.sh — restore the low-latency measurement environment for F+R fuzzing.
#
# A reboot resets all of these to defaults, which makes F+R timing noisy: measurements stop
# stabilizing, so Revizor re-runs them (the "R:" counter in the stats climbs) and throughput
# drops several-fold. Run this ONCE after each reboot, before fuzzing. Requires sudo.
#
# What it pins:
#   - SMT off                         (sibling hyperthread contends L1D/BTB)
#   - nmi_watchdog off                (periodic NMI perturbs timing)
#   - cpufreq governor = performance  (stop the clock drifting down between measurements)
#   - EPP = performance               (same, for the amd-pstate-epp driver)
#   - deep C-states (C2/C3) disabled  (wakeup latency is the main F+R jitter source)
#
# It does NOT touch the kernel module or microcode (those are set via the kernel cmdline:
# dis_ucode_ldr keeps microcode at 0x..0206, the revision that leaves the cross-VM BTB channel open).
set -u
echo "== restoring low-latency measurement env (sudo) =="

echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1
echo "SMT            : $(cat /sys/devices/system/cpu/smt/control 2>/dev/null)"

echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog >/dev/null 2>&1
echo "nmi_watchdog   : $(cat /proc/sys/kernel/nmi_watchdog 2>/dev/null)"

for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee "$g" >/dev/null 2>&1; done
echo "governor       : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)"

for e in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo performance | sudo tee "$e" >/dev/null 2>&1; done
echo "EPP            : $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null)"

# disable every idle state deeper than C1 (keep POLL + C1 so the box doesn't cook)
n=0
for d in /sys/devices/system/cpu/cpu*/cpuidle/state[2-9]/disable; do echo 1 | sudo tee "$d" >/dev/null 2>&1; n=$((n+1)); done
echo "deep C-states  : disabled on $n state files (kept POLL + C1)"
echo "cpu0 states now:"; for s in /sys/devices/system/cpu/cpu0/cpuidle/state*/; do
    echo "   $(cat "$s/name") disable=$(cat "$s/disable")"; done

echo "== done. Re-run the fuzzer; the R: (rerun) counter should stay near 0. =="
