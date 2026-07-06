---
marp: true
title: RL-Discovered Speculative Leaks — Root Cause
paginate: true
---

# RL-Discovered Speculative-Execution Leaks
### Root-causing violations beyond the `ct + cond-bpas` contract

- Tool: Revizor (sca-fuzzer) + RL test-case generator (SpecRL)
- Target: AMD (Zen), Prime+Probe, SSBD **off** (store bypass enabled)
- Contract under test: **CT observation + cond-bpas execution** (Spectre-v1 ∪ v4)
- Headline: agent finds **real leaks the contract cannot explain** → root cause = **partial store-to-load forwarding (STLF)**

---

# Background: contracts & violations (30s)

- A **leakage contract** = a model of *what a program is allowed to leak*.
- Revizor compares, per program:
  - **contract trace** (what the model says is observable), vs
  - **hardware trace** (Prime+Probe cache footprint).
- **Violation** = two inputs with the **same contract trace** but **different hardware traces**
  → hardware leaks *more* than the contract models.

> Contract lattice (weak→strong):
> `seq` ⊂ `cond` (v1) ⊂ `bpas` (v4) ⊂ `cond-bpas` (v1+v4)
> A violation under `cond-bpas` ⇒ leak is **beyond both v1 and v4**.

---

# Setup

- **Generator**: RL agent emits x86 gadgets (SpecRL); reward ∝ contract violation.
- **Contract (forced in the rollout worker)**: `ct` + `cond-bpas`.
- **Executor**: P+P, **SSBD off** (`spec_store_bypass: Vulnerable`) — matches training.
- Verification protocol (lessons-hardened):
  - replay under an **explicit** contract (never trust a mutable config file)
  - **SSBD state asserted** before every run
  - full FP pipeline (nesting / taint / **priming** / large-sample) → real vs flaky

---

# Results: violations after 6/19

| Metric | Value |
|---|---|
| Raw violations (6/19–6/20) | **127** |
| **Reproduce under `cond-bpas` + priming** | **89 real / 38 flaky (70%)** |
| Distinct gadgets among the 89 | **79** |
| With conditional branches | **0 / 89** → not Spectre-v1 |
| With `div`/`mul` (variable latency) | 53 / 89 |
| With `lock`/`xchg` atomics | **89 / 89** |

- 89 reproduce **under the same `cond-bpas` contract that found them** → genuinely beyond v1+v4.
- This slide deck root-causes a **representative no-`div` case**.

---

# Case study: `violation-260619-040753`

Minimal gadget (after `minimize`; committed addrs identical for both inputs):

```
L1  cmovz  rsi, [r14+rcx]          ; LOAD  8B @ 0x1788  (set30, off 8)
L2  movzx  rax, byte [r14+rsi]     ; LOAD  1B @ 0x1439  (set16, off 57)
L3  lock cmpxchg [r14+rdx], rdx    ; STORE 8B @ 0x1788  (atomic)
L4  add    [r14+rax], rax          ; STORE 8B @ 0x1439  (set16, off 57)  ← unaligned
L5  add    rcx, [r14+rsi]          ; LOAD  8B @ 0x1439  (set16, off 57)  ← unaligned
L6  mov    rsi, [r14+rcx]          ; LOAD  @ r14+rcx     ← TRANSMITTER
L7  mov    [r14+rsi], 0            ; STORE
```

- **No branches** ⇒ v1 ruled out. Committed traces identical ⇒ leak is **speculative**.

---

# The key address: 0x1439 is line-split

```
cache line = 64 B.   0x1439 → set 16, offset 57.

 offset:  ... 56 57 58 59 60 61 62 63 | 0 1 ...   (next line, set 17)
 8B access @57:        [== 7 bytes ==] | [1 byte]
                       └─ line 16 ──┘   └ line 17 ┘   ← CROSSES the boundary
```

- L4 **STORE 8B** then L5 **LOAD 8B**, both at the *same unaligned, line-crossing* 0x1439.
- A load that partially overlaps / line-splits an in-flight store → **store-to-load
  forwarding cannot complete cleanly** → data-dependent transient.

---

# Root-cause methodology (official + ours)

1. `minimize` → smallest gadget that still violates (above).
2. **comment pass** → per-input committed addresses (identical → speculative).
3. **de-align experiment** (decisive) → is the line-split causal?
4. **fence pass** → is it speculative? where is the window?
5. **code audit** → can the contract even represent this?

---

# Decisive experiment: de-align

Change *only* the alignment of the 0x1439 accesses (mask low-3 bits → 0):

```
- and rsi/rax, 0b1111111111111   ; any offset → 0x1439 (off 57, line-split)
+ and rsi/rax, 0b1111111111000   ; 8B aligned → 0x1438 (off 56, NO split)
```
Store **and** load both move to 0x1438 → **still alias each other; only the split is removed.**

| Program | Result |
|---|---|
| baseline (line-split allowed) | **YES** (violation) |
| de-aligned (8-byte aligned) | **NO** (gone) |

➡ **The line-split / partial overlap is the cause** — not aliasing, not data path.
➡ Also rules out classic v4: full same-address bypass survives alignment; this doesn't.

---

# Fence pass: it is speculative

`LFENCE`-after-every-instruction, keep only fences that preserve the violation:

```
lfence
lfence            ← only insertable OUTSIDE the chain
cmovz  rsi,[r14+rcx]
movzx  rax,[r14+rsi]
lock cmpxchg ...
add    [r14+rax],rax        ┐
add    rcx,[r14+rsi]        │  NO lfence can be placed inside this chain
mov    rsi,[r14+rcx]        │  (any fence here kills the violation)
mov    [r14+rsi],0          ┘
```

- The whole store→load→transmitter chain is **one speculation window** ⇒ **speculative**.
- Fences *do* kill it ⇒ **not** a model/executor bug.

---

# Why `cond-bpas` cannot explain it (code)

Revizor's only memory-speculation model = `StoreBpasSpeculator`:

```python
""" skips a store if followed by a load FROM THE SAME ADDRESS """      # full, same-addr only
self._emulator.mem_write(store_addr, old_value)                       # reverts WHOLE store
# partial / multi-store overlap:
raise NotImplementedError("Instructions with multiple stores ...")    # unsupported
```
- Models **whole-store, same-address** bypass (classic v4) only.
- **No partial / line-split / size-mismatch forwarding** anywhere in the model.
- ⇒ a partial-STLF transient is **absent from the contract trace** ⇒ inputs look
  equivalent to the model but differ on hardware ⇒ **violation**.

---

# Root cause (the mechanism)

```
        STORE 8B @0x1439 (unaligned, crosses cache line)
                 │  in-flight in the store queue
                 ▼
        LOAD  8B @0x1439  ── partial / split overlap ──► STLF cannot forward cleanly
                 │                                        → data-dependent transient value
                 ▼
        rcx (speculative, secret-dependent)
                 ▼
        L6: mov rsi,[r14+rcx]   ← transmitter: cache set depends on the secret
                 ▼
        Prime+Probe sees a data-dependent footprint  →  VIOLATION
```

**Partial store-to-load forwarding**: speculative, alignment/line-split-dependent,
SSBD-independent, and **outside** the v1/v4 (`cond-bpas`) model.

---

# Significance

- After fixing contract wiring, the RL fuzzer finds **genuine beyond-contract leaks**, not noise (89/127 reproduce + survive priming).
- Concrete, reproducible **gap in Revizor's `bpas` execution clause**: it models only whole-store same-address bypass, missing **partial/line-split STLF**.
- Characterizes a **data-dependent partial-STLF transient on AMD** with a minimal 6-instruction gadget.
- *Caveat*: partial STLF is a **known μ-arch effect**; novelty here is "a leak class the contract model cannot express", not (yet) a new CPU vulnerability — pending literature/cross-μarch comparison.

---

# Next steps

1. Classify the rest of the 89 (esp. the **53 with `div`/`mul`** — possibly a different sub-mechanism).
2. Extend the `bpas` model → **multi-store + partial/size-mismatch forwarding**; re-fuzz, see what still survives.
3. Converge `input-diff` to pin the exact **leaked byte** (`^`); build a clean stand-alone PoC.
4. Cross-check vs known AMD STLF / 4K-aliasing literature; test other microarchitectures.
5. Tighten generation-time priming to cut the 38 flaky false-positives.

---

# Summary

- **127** violations (6/19+); **89 real** under `ct + cond-bpas` (SSBD off).
- Representative root cause = **partial store-to-load forwarding**, proven by:
  - de-align kills it (split is causal) • fence localizes a speculation window •
    code shows the contract can't model partial forwarding • no branches / committed-identical (speculative, not v1).
- Take-away: the RL fuzzer surfaces **real leaks beyond v1+v4**, exposing a model gap to close next.
