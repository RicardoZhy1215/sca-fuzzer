#!/usr/bin/env bash
#
# reload_until_leak.sh — recover a physical layout that reproduces THIS hpa_gpa/L1TF violation.
#
# The leak is physical-memory-layout dependent (hpa_gpa makes guest GPAs = physical addresses, whose
# cache/EPT aliasing is what leaks). A reboot re-rolled the layout, so the saved TC stopped
# reproducing. This reloads the module (each insmod re-rolls the layout) until one BOTH:
#   (a) loads with no EPT collision, AND
#   (b) reproduces the violation.
# Then it STOPS with that module loaded, ready for your experiment.
#
# Needs sudo (rmmod/insmod each iteration — run in a shell where sudo is cached, or `sudo -v` first).
#
# Usage:
#   bash reload_until_leak.sh                 # default = u2u violation, up to 40 reloads
#   MAX=80 REPRO_TRIES=4 bash reload_until_leak.sh
#   V=<u2k-dir> CONFIG=<...> bash reload_until_leak.sh
set -uo pipefail
REPO=/home/mluo/sca-fuzzer
KO=$REPO/rvzr/executor_km/rvzr_executor.ko
PY=${PY:-/home/mluo/miniconda3/bin/python3}
V=${V:-$REPO/campaigns/vmscape-repro/out-fuzz-u2u/violation-260814-125938}
CONFIG=${CONFIG:-$V/reproduce.yaml}
MAX=${MAX:-40}
REPRO_TRIES=${REPRO_TRIES:-3}   # a leaking layout reproduces ~9/10, so a few tries confirm it

[ -f "$V/program.asm" ] || { echo "no program.asm in $V"; exit 2; }
sudo -v || { echo "need sudo for rmmod/insmod"; exit 2; }   # cache sudo so the loop never prompts

# robust reload: rmmod can lag, so retry until the module is actually gone, THEN insmod
reload_module () {
    local a w
    for a in 1 2 3 4 5 6; do
        # unload, then WAIT for /sys/module/rvzr_executor to fully disappear (lsmod lags; insmod
        # races the teardown and returns "File exists" if the kobject still exists)
        [ -d /sys/module/rvzr_executor ] && { sudo rmmod rvzr_executor 2>/dev/null || true; }
        w=0
        while [ -d /sys/module/rvzr_executor ] && [ "$w" -lt 25 ]; do
            sudo rmmod rvzr_executor 2>/dev/null || true; sleep 0.2; w=$((w+1))
        done
        if [ -d /sys/module/rvzr_executor ]; then sleep 0.5; continue; fi   # still loaded -> retry cycle
        # fully unloaded -> load; retry if insmod races
        sudo insmod "$KO" 2>/dev/null && return 0
        sleep 0.6
    done
    echo "  reload failed after retries (refcnt=$(cat /sys/module/rvzr_executor/refcnt 2>/dev/null || echo n/a))"
    return 2
}

# low-latency env once (persists across module reloads; F+R needs it)
echo off | sudo tee /sys/devices/system/cpu/smt/control >/dev/null 2>&1
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance | sudo tee "$g" >/dev/null 2>&1; done
for e in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do echo performance | sudo tee "$e" >/dev/null 2>&1; done
for d in /sys/devices/system/cpu/cpu*/cpuidle/state[2-9]/disable; do echo 1 | sudo tee "$d" >/dev/null 2>&1; done

echo "searching for a layout that reproduces $V (up to $MAX reloads)..."
collided=0; loaded_noleak=0
for i in $(seq 1 "$MAX"); do
    reload_module || { echo "  reload $i: module reload failed — aborting"; exit 2; }

    b=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)
    hit=0
    for t in $(seq 1 "$REPRO_TRIES"); do
        timeout 90 "$PY" "$REPO/revizor.py" reproduce -s "$REPO/base.json" -c "$CONFIG" -I "$V" \
            -t "$V/program.asm" -i "$V"/input_*.bin >/dev/null 2>&1
        [ $? -eq 1 ] && { hit=1; break; }
    done
    a=$(journalctl -k -b 2>/dev/null | grep -c "set_extended_page_tables] Assertion" || true)

    if [ "$a" -gt "$b" ]; then
        collided=$((collided+1)); printf "  reload %2d: EPT collision (unusable)\n" "$i"
    elif [ "$hit" -eq 1 ]; then
        echo "=================================================================="
        echo " *** LEAK REPRODUCED on reload $i *** — module is loaded and ready."
        echo "   srcversion: $(cat /sys/module/rvzr_executor/srcversion)"
        echo "   verify:  N=10 bash campaigns/vmscape-repro/replay_fuzz.sh"
        echo "   (do NOT reload the module now, or you lose this layout)"
        echo "=================================================================="
        exit 0
    else
        loaded_noleak=$((loaded_noleak+1)); printf "  reload %2d: loads, no leak\n" "$i"
    fi
done
echo "=================================================================="
echo " no leaking layout in $MAX reloads (collided=$collided, loaded-no-leak=$loaded_noleak)."
echo " => this exact TC's layout is rare. Better path: re-fuzz under the current layout to get a"
echo "    currently-valid violation:  bash campaigns/vmscape-repro/run_fuzz.sh u2u"
echo "=================================================================="
exit 1
