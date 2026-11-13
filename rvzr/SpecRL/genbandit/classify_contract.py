"""
Which known speculation explains each Spectre v4 / v1 violation? Two independent checks per violation,
on its own saved counterexample pair (the two inputs named in report.txt):

  model     contract traces of the pair (nesting 1, then 5) under a permissive contract: CT-BPAS (store bypass is allowed)
            for v4, CT-COND (branch misprediction is allowed) for v1. The pair is a CT-SEQ violation, so
            its CT-SEQ traces are equal; if they DIFFER under the permissive contract, the model says that
            speculation explains the leak. (The two clauses are used separately: the cond-bpas composition
            has a known bug.)
  hardware  replay the whole saved input sequence (the pair's neighbours train the predictors) and compare
            the pair's hardware traces, without and with the mitigation for that speculation: v4 with SSBD on (x86_executor_enable_ssbp_patch),
            v1 with an lfence at the start of every basic block (both successors of every branch). If the
            hardware traces become equivalent, the mitigation removes the leak.

class:  explained       both checks agree it is this speculation
        model-only      the model explains it, the mitigation does not remove it
        hw-only         the mitigation removes it, the model does not explain it
        unexplained     neither (look at these by hand)
        not-reproduced  the original pair shows no difference on the hardware now

  cd ~/sca-fuzzer   (executor free)
  python rvzr/SpecRL/genbandit/classify_contract.py v4 <out.tsv>
  python rvzr/SpecRL/genbandit/classify_contract.py v1 <out.tsv>
"""
import os
import re
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../.."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)
from rvzr.config import CONF  # noqa: E402
from rvzr import factory  # noqa: E402

BB = re.compile(r"^\s*\.(bb_\d+\.\d+|exit_\d+):\s*$")


def fence_branches(lines):
    """lfence at the start of every basic block (incl. the exit block): both successors of every
    conditional branch start with one. (Not right after the jcc: a block must end in its terminators.)"""
    out = []
    for l in lines:
        out.append(l)
        if BB.match(l):
            out.append("lfence\n")
    return out


class Checker:
    def __init__(self, target, first_vdir, reps=50):
        self.target, self.reps = target, reps
        CONF.load(os.path.join(first_vdir, "reproduce.yaml"), first_vdir)
        self.f = factory.get_fuzzer(os.path.join(ROOT, "base.json"), tempfile.mkdtemp(), "", None)
        saved = list(CONF.contract_execution_clause)
        CONF.contract_execution_clause = ["bpas" if target == "v4" else "cond"]
        self.alt = factory.get_model(self.f.executor.read_base_addresses())
        CONF.contract_execution_clause = saved
        self.tmp = tempfile.mkdtemp()

    def hw_differs(self, lines, name, trials=2):
        path = os.path.join(self.tmp, name + ".asm")
        open(path, "w").writelines(lines)
        tc = self.f.asm_parser.parse_file(path, self.f.code_gen, self.f.elf_parser)
        self.f.executor.load_test_case(tc)
        hits = 0
        for _ in range(trials):
            # the whole saved input sequence, as in the campaign: the predictors' state depends on the
            # inputs around the pair (replaying the pair alone loses most v4 leaks)
            h = self.f.executor.trace_test_case(self.inputs, self.reps)
            a, b = self.pair_idx
            hits += not self.f.analyser.htraces_are_equivalent(h[a], h[b])
        return hits, tc

    def ssbd(self, on):
        CONF.x86_executor_enable_ssbp_patch = on
        self.f.executor._set_vendor_specific_features()

    def check(self, vdir):
        ids = [int(x) for x in re.findall(r"^Input #(\d+)", open(os.path.join(vdir, "report.txt")).read(), re.M)][:2]
        n = len([x for x in os.listdir(vdir) if re.match(r"input_\d{4}\.bin$", x)])
        self.inputs = self.f.data_gen.load([os.path.join(vdir, f"input_{i:04}.bin") for i in range(n)])
        self.pair_idx = ids
        pair = [self.inputs[i] for i in ids]
        lines = open(os.path.join(vdir, "program.asm")).readlines()
        orig, tc = self.hw_differs(lines, "orig")
        seq = self.f.model
        seq.load_test_case(tc)
        ct_seq = seq.trace_test_case(pair, 1)
        self.alt.load_test_case(tc)
        # nesting 1 first, as Revizor's fast path does; 5 only if 1 does not explain it (30, the
        # configured maximum, is far too slow with BPAS: every store opens a speculative path)
        nest = 1
        ct_alt = self.alt.trace_test_case(pair, nest)
        if ct_alt[0] == ct_alt[1]:
            nest = 5
            ct_alt = self.alt.trace_test_case(pair, nest)
        model_explains = ct_seq[0] == ct_seq[1] and ct_alt[0] != ct_alt[1]
        if self.target == "v4":
            self.ssbd(True)
            try:
                mit, _ = self.hw_differs(lines, "mitigated")
            finally:
                self.ssbd(False)
        else:
            mit, _ = self.hw_differs(fence_branches(lines), "mitigated")
        if orig == 0:
            cls = "not-reproduced"
        elif model_explains and mit == 0:
            cls = "explained"
        elif model_explains:
            cls = "model-only"
        elif mit == 0:
            cls = "hw-only"
        else:
            cls = "unexplained"
        return {"pair": f"#{ids[0]}/#{ids[1]}", "seq_equal": ct_seq[0] == ct_seq[1], "alt_differs": ct_alt[0] != ct_alt[1],
                "nesting": nest, "orig_hw": orig, "mitigated_hw": mit, "class": cls}


def main():
    target, out = sys.argv[1], sys.argv[2]
    from violation_similarity import sets
    groups = sets()[target]
    first = next(iter(groups.values()))[0]
    chk = Checker(target, first)
    done = set()
    if os.path.exists(out):
        done = {l.split("\t")[1] for l in open(out) if not l.startswith("tool\t")}
    new = not os.path.exists(out)
    with open(out, "a", buffering=1) as f:
        if new:
            f.write("tool\tviolation\tpair\tseq_equal\tpermissive_differs\tnesting\torig_hw(of 2)\tmitigated_hw(of 2)\tclass\n")
        for tool, dirs in groups.items():
            for vdir in dirs:
                key = os.path.relpath(vdir, HERE)
                if key in done:
                    continue
                try:
                    r = chk.check(vdir)
                    row = [tool, key, r["pair"], r["seq_equal"], r["alt_differs"], r["nesting"], r["orig_hw"], r["mitigated_hw"], r["class"]]
                except Exception as e:                       # keep going; record it
                    row = [tool, key, "", "", "", "", "", "", f"error: {type(e).__name__}: {e}"[:150]]
                f.write("\t".join(str(x) for x in row) + "\n")
    if target == "v4":
        chk.ssbd(False)


if __name__ == "__main__":
    main()
