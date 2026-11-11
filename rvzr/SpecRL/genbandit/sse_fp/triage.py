"""
Hardware-only triage of SSE-FP violations: is each one a real FPVI-family leak, or something else
(an architectural difference, e.g. a Unicorn-vs-Zen4 mismatch the pipeline let through)?

Port of side-channel-fuzzer sse_fp/triage_ext.py to this fork. Replays the program and variants
of it on the violation's own counterexample pair (the saved inputs named in report.txt), so no
model is involved:

  orig              the program as saved
  fence-fp          lfence after every FP op (arithmetic, cvt*, round*, dp*, comis*)
  fence-all         lfence after every instruction (no transient execution left)
  ftzdaz            MXCSR = FTZ|DAZ before the program: stops denormal FPVI and the
                    no-denormal variant (VVI)
  ftzdaz+fence-fp, ftzdaz+fence-all

Verdict:
  not-reproduced    orig shows no difference on the saved pair
  architectural     fence-all still differs: no speculation needed (NOT a speculative leak)
  FPVI-family       ftzdaz removes it
  CANDIDATE         survives ftzdaz, gone with ftzdaz+fence-all (transient, not FP underflow/denormal)
  ftz-architectural ftzdaz+fence-all differs (FTZ/DAZ changed architectural results; inconclusive)

Also prints, per violation, the cache sets only one input touches (orig) and the FP ops in the
program, a first look at whether violations are variants of one gadget.

  cd ~/sca-fuzzer   (executor free)
  python rvzr/SpecRL/genbandit/sse_fp/triage.py <violation-dir>... [--reps 50] [--out verdicts.tsv]
         [--include <dir the reproduce.yaml's !include files are in>]
"""
import os
import re
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.getcwd())
from rvzr.config import CONF  # noqa: E402
from rvzr import factory  # noqa: E402

FP_OP = re.compile(r"^\s*((add|sub|mul|div|sqrt|hadd|hsub|addsub|min|max)(ps|ss|pd|sd)|cvt\w+"
                   r"|round(ps|ss|pd|sd)|dp(ps|pd)|u?comis[sd])\b")
FP_NAME = re.compile(r"^\s*((add|sub|mul|div|sqrt|hadd|hsub|addsub)(ps|ss|pd|sd)|cvt\w+|round\w\w"
                     r"|blend\w+|insertps|extractps)\b")
MXCSR_SLOT = "dword ptr [r14 + 0x1ffc]"
VARIANTS = ["orig", "fence-fp", "fence-all", "ftzdaz", "ftzdaz+fence-fp", "ftzdaz+fence-all"]


def fenced(lines, which):
    out, inside = [], False
    for line in lines:
        out.append(line)
        s = line.strip()
        if s.startswith(".macro.measurement_start"):
            inside = True
        elif s.startswith(".macro.measurement_end"):
            inside = False
        elif inside and s and not s.startswith((".", "#")):
            if which == "all" or FP_OP.match(s):
                out.append("lfence\n")
    return out


def ftzdaz(lines):
    out = []
    for line in lines:
        out.append(line)
        if line.startswith(".macro.measurement_start"):
            out += [f"mov {MXCSR_SLOT}, 0x9fc0\n", f"ldmxcsr {MXCSR_SLOT}\n"]
    return out


def build(lines, name):
    if name.startswith("ftzdaz"):
        lines = ftzdaz(lines)
    if name.endswith("fence-fp"):
        lines = fenced(lines, "fp")
    if name.endswith("fence-all"):
        lines = fenced(lines, "all")
    return lines


def verdict(d):
    if not d["orig"]:
        return "not-reproduced"
    if d["fence-all"]:
        return "architectural"
    if not d["ftzdaz"]:
        return "FPVI-family"
    if d["ftzdaz+fence-all"]:
        return "ftz-architectural"
    return "CANDIDATE"


def dominant(h):
    raw = [int(v) for v in h.get_raw_traces()]
    return Counter(raw).most_common(1)[0][0]


def sets(v):
    return {s for s in range(64) if (v >> (63 - s)) & 1}


def main():
    args, reps, out, inc = sys.argv[1:], 50, None, None
    for flag in ("--reps", "--out", "--include"):
        if flag in args:
            i = args.index(flag)
            val = args[i + 1]
            del args[i:i + 2]
            if flag == "--reps":
                reps = int(val)
            elif flag == "--out":
                out = val
            else:
                inc = os.path.abspath(val)
    fuzzer = None
    rows = []
    for vdir in map(os.path.abspath, args):
        if fuzzer is None:
            CONF.load(os.path.join(vdir, "reproduce.yaml"), inc or vdir)
            fuzzer = factory.get_fuzzer(os.path.join(os.getcwd(), "base.json"), tempfile.mkdtemp(), "", None)
        ids = [int(x) for x in re.findall(r"^Input #(\d+)", open(os.path.join(vdir, "report.txt")).read(), re.M)][:2]
        inputs = fuzzer.data_gen.load([os.path.join(vdir, f"input_{i:04}.bin") for i in ids])
        lines = open(os.path.join(vdir, "program.asm")).readlines()
        fps = sorted({m.group(1) for l in lines if (m := FP_NAME.match(l))})
        os.makedirs(os.path.join(vdir, "triage"), exist_ok=True)
        differs, diffsets = {}, {}
        for name in VARIANTS:
            path = os.path.join(vdir, "triage", f"{name}.asm")
            open(path, "w").writelines(build(lines, name))
            tc = fuzzer.asm_parser.parse_file(path, fuzzer.code_gen, fuzzer.elf_parser)
            fuzzer.executor.load_test_case(tc)
            h = fuzzer.executor.trace_test_case(inputs, reps)
            differs[name] = not fuzzer.analyser.htraces_are_equivalent(h[0], h[1])
            sa, sb = sets(dominant(h[0])), sets(dominant(h[1]))
            diffsets[name] = (sorted(sa - sb), sorted(sb - sa))
        v = verdict(differs)
        flags = "".join("D" if differs[n] else "-" for n in VARIANTS)
        a_only, b_only = diffsets["orig"]
        row = (os.path.basename(os.path.dirname(vdir)), os.path.basename(vdir), f"#{ids[0]}/#{ids[1]}",
               v, flags, f"A-only {a_only} B-only {b_only}", " ".join(fps))
        print("\t".join(row), flush=True)
        open(os.path.join(vdir, "triage", "verdict.txt"), "w").write("\t".join(row) + "\n")
        rows.append(row)
    if out:
        with open(out, "w") as f:
            f.write("run\tviolation\tpair\tverdict\tflags(" + ",".join(VARIANTS) + ")\torig sets\tFP ops\n")
            for r in rows:
                f.write("\t".join(r) + "\n")


if __name__ == "__main__":
    main()
