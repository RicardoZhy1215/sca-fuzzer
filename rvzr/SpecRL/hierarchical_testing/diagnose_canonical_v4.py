"""
Diagnose whether this machine + kernel + microcode + fuzzer config can
actually trigger Spectre v4 on the canonical hand-written gadget.

Mirrors SpecEnv._full_obs_program's executor/model/analyser pipeline but
bypasses the hardcoded my_test_case.asm path inside X86Fuzzer.start_SpecRL.
We directly parse canonical v4.asm, assemble, trace unfenced+fenced through
X86IntelExecutor, and apply the same analyser checks. Extra diagnostics
are printed (raw htrace diffs, per-input equivalence, per-rep fire counts)
that _full_obs_program throws away.

Usage (from hierarchical_testing/):
    python diagnose_canonical_v4.py                     # default: tests/x86_tests/asm/spectre_v4.asm
    python diagnose_canonical_v4.py path/to/other.asm
"""
import os
import sys
from collections import Counter

_this_dir = os.path.dirname(os.path.abspath(__file__))
_specrl_root = os.path.dirname(_this_dir)
_rvzr_root = os.path.dirname(_specrl_root)
_repo_root = os.path.dirname(_rvzr_root)
for p in (_repo_root, _rvzr_root, _specrl_root, _this_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from hi_SpecEnv import SpecEnv  # noqa: E402
from rvzr.arch.x86.fuzzer import _create_fenced_test_case  # noqa: E402


DEFAULT_CANONICAL = "/home/hz25d/sca-fuzzer/tests/x86_tests/asm/spectre_v4.asm"
N_TRIALS = 5
N_REPS = 20


def _htrace_raw(ht):
    """Return tuple of raw trace samples for stable counting."""
    try:
        return tuple(ht.get_raw_traces())
    except Exception:
        return tuple()


def _fire_summary(h_raw, f_raw):
    """
    How often do unfenced samples pick up a trace value that never appears
    in fenced samples? That's the direct SSB fire-rate estimator.
    """
    fenced_set = set(f_raw)
    fired = sum(1 for x in h_raw if x not in fenced_set)
    return fired, len(h_raw)


def main():
    canonical = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CANONICAL
    assert os.path.exists(canonical), f"asm not found: {canonical}"

    env_config = {
        "sequence_size": 50,
        "num_inputs": 50,
        "vulnerability_type": "spectre_v4",
        "pattern_reward_scale": 5.0,
        "leak_reward": 600.0,
        "observable_bonus": 50.0,
        "observable_penalty": 0.0,
        "step_penalty": 0.05,
        "min_steps_before_end": 15,
        "early_end_penalty": 20.0,
        "trace_divergence_reward_scale": 100.0,
    }
    env = SpecEnv(env_config)
    env._force_kernel_ssb_vulnerable()

    print("=" * 72)
    print("DIAGNOSTIC: can this machine trigger v4 on the canonical gadget?")
    print(f"asm     : {canonical}")
    print(f"trials  : {N_TRIALS}")
    print(f"n_reps  : {N_REPS}")
    print(f"n_inputs: {env_config['num_inputs']}")
    print("=" * 72)

    # Parse canonical asm via the same pipeline RL uses.
    canonical_copy = os.path.join(_this_dir, "canonical_v4_probe.asm")
    with open(canonical, "r") as src, open(canonical_copy, "w") as dst:
        dst.write(src.read())

    # parse_file already runs assign_obj + assemble + populate_elf_data,
    # so we must NOT do it again.
    probe = env.asm_parser.parse_file(
        canonical_copy, env.generator, env.elf_parser
    )

    # Build fenced version once (deterministic w.r.t. canonical asm).
    fenced_asm = os.path.join(_this_dir, "canonical_v4_probe.fenced.asm")
    fenced_tc = _create_fenced_test_case(
        canonical_copy, fenced_asm, env.asm_parser, env.generator, env.elf_parser
    )

    analyser = env.analyser
    any_leak = 0
    any_observable = 0
    div_totals = []

    for trial in range(N_TRIALS):
        env.inputs = env.input_gen.generate(env_config["num_inputs"], n_actors=1)

        env.executor.load_test_case(probe)
        h = env.executor.trace_test_case(env.inputs, n_reps=N_REPS)

        env.executor.load_test_case(fenced_tc)
        fh = env.executor.trace_test_case(env.inputs, n_reps=N_REPS)

        observable = h != fh

        # Per-input equivalence (analyser view).
        div_count = 0
        n = min(len(env.inputs), len(h), len(fh))
        per_input_details = []
        for i in range(n):
            eq = analyser.htraces_are_equivalent(h[i], fh[i])
            h_raw = _htrace_raw(h[i])
            f_raw = _htrace_raw(fh[i])
            fired, total = _fire_summary(h_raw, f_raw)
            extra_uniq = len(set(h_raw) - set(f_raw))
            per_input_details.append((eq, fired, total, extra_uniq))
            if not eq:
                div_count += 1
        div_score = div_count / float(n or 1)

        # Summarize: how many inputs had ANY new trace value in unfenced?
        inputs_with_fire = sum(1 for _, fired, _, _ in per_input_details if fired > 0)
        inputs_with_extra_line = sum(1 for _, _, _, ex in per_input_details if ex > 0)

        print(
            f"[trial {trial}] observable={observable} "
            f"div_score={div_score:.3f} "
            f"inputs_with_new_trace_val={inputs_with_fire}/{n} "
            f"inputs_with_extra_cache_line={inputs_with_extra_line}/{n}"
        )
        # Show top 3 most-diverging inputs.
        top = sorted(
            [(i, d) for i, d in enumerate(per_input_details)],
            key=lambda kv: -kv[1][3],
        )[:3]
        for i, (eq, fired, total, extra) in top:
            if extra == 0 and fired == 0:
                break
            print(
                f"    input {i}: analyser_eq={eq} "
                f"new_trace_samples={fired}/{total} "
                f"unique_unfenced_trace_values_not_in_fenced={extra}"
            )

        any_observable += int(observable)
        any_leak += int(div_score > 0)
        div_totals.append(div_score)

    mean_div = sum(div_totals) / len(div_totals)
    print("-" * 72)
    print(
        f"summary: observable_trials={any_observable}/{N_TRIALS}, "
        f"trials_with_div_score>0={any_leak}/{N_TRIALS}, "
        f"mean_div_score={mean_div:.3f}"
    )

    if any_leak > 0:
        print(
            "\n=> HW is vulnerable AND executor pipeline works on canonical. "
            "Any remaining training failure is on the RL side — push the "
            "agent to longer slow chains on the store base (gate full_gadget "
            "behind chain_len >= 5)."
        )
    elif any_observable > 0:
        print(
            "\n=> Raw htrace differs but analyser collapses the difference "
            "to zero. Either lower analyser_stat_threshold further (try 0.01) "
            "or switch to `analyser: bitmaps` with "
            "`analyser_subsets_is_violation: true` + "
            "`analyser_outliers_threshold: 0.05`. Another option: bump n_reps "
            f"to 100+ (currently {N_REPS})."
        )
    else:
        print(
            "\n=> Even the canonical v4 gadget produces ZERO observable diff "
            "on this machine. Kernel/microcode has killed SSB. "
            "Verify the '[SSBD] wrote 0' line appeared above; if yes, the "
            "microcode is enforcing SSBD regardless. Options: boot with "
            "`spec_store_bypass_disable=off nosmt=off mitigations=off`, or "
            "try a CPU with older microcode."
        )


if __name__ == "__main__":
    main()
