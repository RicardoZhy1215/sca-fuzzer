---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- _paginate: false -->

# A *Delayed-Exception* Fault-Shadow Leak on Intel Sapphire Rapids

### Found by leakage-contract fuzzing (SpecRL / Revizor)

**`violation-260721-212748`** — minimized to 8 instructions, mechanism pinned by single-variable `asm` controls

<br>

Intel Xeon **w5-3423** (Sapphire Rapids) · kernel 6.8.0-124 · Prime+Probe · **all named transient CVEs "Not affected"**

---

## Leakage-contract fuzzing in one slide

- **Contract** = (observation clause, execution clause). Here: **`ct` + `delayed-exception-handling` (DEH)**.
  - `ct` — the model exposes the **address** of every memory access + the PC.
  - `deh` — the model keeps executing **transiently past a faulting instruction**.
- **Violation** = two inputs with the **same contract trace** but a **different hardware trace** (Prime+Probe cache footprint) → hardware leaks *more* than the contract models.

> Contract lattice: `seq` ⊂ `deh` (fault-based, Meltdown-family).
> A violation under `ct+deh` ⇒ the CPU's **fault-shadow** behavior is **unsound** w.r.t. DEH.

---

## Setup — one transient source by construction

- **Faulty page `present = false`** → in-sandbox accesses can hit a non-present page → **#PF**. *This is the DEH trigger.*
- **SSBD ON** (`x86_executor_enable_ssbp_patch: true`) → store bypass mitigated → **Spectre-v4 excluded**.
- **Single basic block, no branches** → no misprediction entry → **Spectre-v1 excluded**.
- **Observation**: L1D **Prime+Probe**.

> With v1 and v4 excluded, the **only** path to transient execution is the fault.
> This is squarely a **delayed-exception (DEH)** question.

---

## The finding — same ctrace, different htrace

Fuzzing flagged **test case 2350**: two inputs share one contract-trace hash but diverge on hardware. After minimization the violating pair separates **cleanly and stably**:

```
^^.^...^..^................^.^............^....................^ | A:481  B:0
^^.^...^...................^.^............^....................^ | A:0    B:476
```

- One cache set differs between the two dominant traces (~480 / 500 samples each, mutually exclusive).
- **10 / 10** independent reproductions — survives the full slow path (sample size → 500). **Not noise.**

---

## The minimized gadget (51 → 8 instructions)

```asm
      not  rdi
      and  rdi, 0x1F00
①     xchg [r14+rdi], rbx      ; faulting access — read component opens the window
      sub  rbx, rbx            ; ← DISCARDS ①'s value (no Meltdown-style forwarding)
      and  rbx, 0x1FE0         ; rbx = 0
②     mov  rdi, [r14+0]        ; normal, PRESENT load of the program's own data
      and  rdi, 0x1FE0         ; mask loaded value → offset
③     mov  [r14+rdi], 0        ; data-dependent STORE → encodes value into a cache set
```

- ① is a load / RMW; ③'s **address** is derived from the value loaded at ②.
- ①'s **own** loaded value is thrown away by `sub rbx,rbx`.

---

## Mechanism — a store in the fault shadow

1. **① opens a transient window.** The #PF on the non-present page is not retired immediately; younger instructions execute transiently in its shadow — as every OoO CPU does.
2. **② reads present data.** Inside the shadow, `mov rdi,[r14+0]` legitimately loads the program's own sandbox data.
3. **③ encodes it.** The store address = that loaded value → it touches a value-dependent L1D set. Prime+Probe reads it out.

> CT+DEH predicts one set of fault-shadow observations; the silicon produces a **value-dependent** one it does not → **same ctrace, different htrace**.

---

## Verification — single-variable `asm` / config controls

| Experiment      | Change                        | Result       | Conclusion                         |
|-----------------|-------------------------------|--------------|------------------------------------|
| Stability       | `reproduce × 10`              | **10/10 YES**| Robust — not a noise artifact      |
| Remove fault    | `present:false → true`        | **NO**       | The **#PF is required**            |
| Insert LFENCE   | fence after ①                 | **NO**       | Genuinely **speculative**          |
| Remove read     | `xchg → plain store`          | **NO**       | Needs a **read / load component**  |
| Vary the op     | `load · lock add/xadd/inc`    | **all YES**  | Read suffices — **atomicity not required** |

> Net: **fault-triggered, speculative, load-gated**. Noise, architectural, and Spectre-v1/v4 are each ruled out by an experiment.

---

## Why it is consistent with "Not affected"

- **What the CVE verdicts certify.** *Meltdown / L1TF / MDS: Not affected* = the CPU won't forward a faulting load's data **across a security boundary** (user↔kernel, VM, SGX).
- **What they never cover.** The fault shadow *existing* — younger instructions executing transiently before a fault retires — is how OoO CPUs work, not a patchable flaw.
- **This gadget stays in the clear:** `sub rbx,rbx` **discards** ①'s value; the leak uses a **normal present load (②)** of the program's **own** data. Nothing crosses a boundary.

> **→ No contradiction. It is a contract violation, not a CVE.**

---

## Empirical proof — same contract, same fault, swap the gadget

Real Meltdown gadget (`fault_load.asm`: encodes the **faulting load's own value**) vs. this violation (`min.asm`: discards it), under **identical** `ct+deh` + `present:false`:

| # | Gadget | Config | Result |
|---|--------|--------|--------|
| A | real Meltdown | revizor's own `l1tf-p.yaml`      | **NO** |
| B | real Meltdown | this violation's `reproduce.yaml`| **NO** |
| C | **this gadget** | `reproduce.yaml`               | **YES** |

> On this CPU (Xeon w5-3423, model 143 ≥ 140), classic Meltdown **dies** (RDCL_NO) — Revizor's own suite even *skips* the `#PF-present` test for exactly this reason. This gadget **lives** because it never uses the patched faulting-load forwarding.

---

## Significance & next step

- **It is a `ct+deh` contract violation.** The CPU's fault-shadow behavior is **unsound** w.r.t. the delayed-exception contract.
- **Why it matters.** Contracts underpin reasoning about software safety (constant-time code, sandboxes). An unsound contract ⇒ code "verified safe" under it can still leak; the same gadget across a real boundary would be exploitable.
- **Next.** Dump the model's per-instruction contract trace → locate which fault-shadow access the DEH dependency heuristic omits ⇒ a **minimal counterexample to DEH soundness** on Sapphire Rapids.

**Artifacts:** `min.asm` (8 instr) · 6 minimized inputs · reproduce / control / model-trace scripts · 10× logs.
