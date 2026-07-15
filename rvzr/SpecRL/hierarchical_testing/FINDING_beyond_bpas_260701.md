# Finding: a genuine *beyond-`bpas`* store-bypass-family leak

**Case:** `violation-260701-234841` (found under `ct + cond-bpas`, AMD Zen, P+P, SSBD off)
**Status:** CONFIRMED candidate — robust, minimized, and *not* explained by a (correct) `bpas` contract.
**Date recorded:** 2026-07 (post contract-bug fixes)

---

## TL;DR

After fixing two contract bugs (below), the RL fuzzer found a **robust, minimal** x86 gadget whose
speculative leak is **NOT captured by Revizor's `bpas`/`cond-bpas` execution clause**, yet is
**mitigated by SSBD** → a **speculative store-bypass VARIANT (SSB / Spectre-v4 family) `bpas` fails to
represent**. The end-to-end chain is now traced by controlled asm edits (each verified to assemble):

> **secret `mem[0x219f+7]`** →[store-bypass transient window, SSBD-gated]→ **`mul rax×V`** → **high
> product `rdx`** → **encoder `L49: movsx rdx,[r14+rdx]`** → **P+P cache footprint**.

Evidence: value-vs-access resolved to the **VALUE**, top byte bits 56–63 (`exp_hibits`, access held
constant); propagation via the **high product rdx** (`exp_chain`: `imul` writing only the low product
dies); encoder is **L49** (zeroing rdx before it dies); the secret is **V itself, not the multiplicand**
(`exp_source`: constant rax still leaks, constant V does not). *(An earlier "partial-overlap STLF" claim
and its retraction were both unsafe — one conflated access-width with value-width, the other used a
control that didn't assemble; these constant-access edits are the clean version.)* **Open:** which store
opens the window; secret RECOVERY from the footprint; PSF vs classic SSB (PSFD not CPUID-enumerated).

This is the *real* version of a story we earlier got wrong: ~89 prior "violations" were **false
positives** caused by a broken `cond-bpas` implementation (Bug B). This one is different — with the
contract fixed, `bpas` genuinely cannot explain it.

---

## Why this is trustworthy: two contract bugs found & fixed FIRST

**Bug A — the rollout worker used the wrong contract.**
Ray workers re-import `CONF` with defaults, so `--config big_fuzz.yaml` (which sets `cond-bpas`)
reached only the driver; the model built in the worker silently used the default
`delayed-exception-handling`. Fixed by forcing the contract in `SpecEnv.__init__`
(`hi_SpecEnv.py`, before `factory.get_model()`).

**Bug B — `cond-bpas` silently degenerated to `seq`.**
`X86CondBpasSpeculator._speculate_mem_access` called `super(StoreBpasSpeculator, self)._speculate_mem_access(...)`.
By the MRO `[X86CondBpas, X86Cond, StoreBpas, UnicornSpeculator]`, `super(StoreBpasSpeculator, self)`
resolves to **`UnicornSpeculator` (a no-op)** — so the store-bypass logic in `StoreBpasSpeculator`
was **never called**. `cond-bpas` therefore did **no speculation at all (== `seq`)**, and every plain
Spectre-v4 leak showed up as a "violation."
Fixed: call `StoreBpasSpeculator._speculate_mem_access(self, ...)` directly, and run **both** cond and
bpas in `_speculate_instruction`. Verified: `cond-bpas` now `!= seq` and `== bpas` for branch-free code.

> The earlier ~89 "cond-bpas violations" were Bug-B false positives (plain V4 that a correct `bpas`
> explains). **This finding is not** — see the evidence below.

---

## Evidence (minimized gadget, violating inputs 46 vs 96, SSBD off unless noted)

| Test | Result | Meaning |
|---|---|---|
| Reproduce minimized gadget, full pipeline (priming) | **REAL 12 / 12** | robust, not flaky, not noise |
| Contract trace, **plain `bpas`** | `46 == 96` (hash `15167705989829076597`) | **`bpas` cannot distinguish the inputs → cannot explain the leak** |
| Contract trace, **`cond-bpas`** (fixed) | `46 == 96` (same `15167…`) | == `bpas` (cond is inert: no conditional branches) |
| Contract trace, **`seq`** | `46 == 96` (hash `6908240687030122763`) | `bpas != seq` → the contract IS speculating, just insufficiently (rules out Bug B) |
| **SSBD ON** | `Violations: 0` | **SSBD mitigates it → SSB / Spectre-v4 family** |
| **De-align** (force all masks 8-byte aligned) | `Violations: 0` | alignment / partial-overlap geometry is essential |

**Contrast with the earlier 89 (false positives):** for those, plain `bpas` *distinguished* the inputs
(different hashes) → `bpas` explained them → they were plain V4. Here plain `bpas` *cannot* distinguish
them → genuinely beyond `bpas`.

Also notable: **minimization strengthened the signal** — the original program survived priming only
~15% (3/20, ~2280–5400 s/run); the **minimized gadget survives 12/12 in ~37 s/run** (removing
noise-adding instructions un-buried the real signal).

---

## The minimized gadget (candidate store→load sites)

Committed memory addresses are identical for inputs 46 and 96 (comment-pass) → the divergence is
purely **speculative**. Candidate mechanisms in the minimized program:

```
mov      [r14+rbx], rax        ; STORE 8B @0x105a
cmovnbe  rax, [r14+rbx]        ; LOAD  8B @0x105a      ← same-addr store→load
...
mul      [r14+rsi]             ; mul computes downstream addresses (plumbing)
...
xchg     [r14+rdi], rax        ; STORE 8B @0x1000
movsx    rax, byte [r14+rdx]   ; LOAD  1B @0x1000      ┐ size-mismatch (byte vs qword)
cmovb    rbx, [r14+rdi]        ; LOAD  8B @0x1000      ┘ at the same address
lock cmpxchg [r14+rax], rbx    ; atomic store
lock inc     [r14+rax]         ; atomic store
```

Full minimized program: `violation-260701-234841/minimized.asm`.

### Geometry — MEASURED (`pin_geometry.sh`, REPS=3, single-access alignment sweep)

Tightening **exactly one** access's instrumentation mask to 8-byte alignment
(`0b1111111111000`) and reproducing:

| aligned access | present / 3 | |
|---|---|---|
| baseline (none) | 3/3 | — |
| S1 store `mov [r14+rbx]` @0x105a | 3/3 | not critical |
| S1 load `cmovnbe [r14+rbx]` @0x105a | 3/3 | not critical |
| S2 byte-load `movsx byte [r14+rdx]` @0x1000 | 3/3 | not critical |
| S2 qword-load `cmovb [r14+rdi]` @0x1000 | 3/3 | not critical |
| bg loads @0x1c77/0x105b/0x1fe1/0x17d8 | 3/3 | not critical |
| **`mul qword ptr [r14+rsi]` @0x219f** | **0/3** | **← the ONLY access whose alignment kills the leak** |

**Correction to the earlier guess:** the two store→load sites (`@0x105a`, `@0x1000`) are **NOT** the
geometry-critical accesses — aligning either does nothing. The single load-bearing access is the
**`mul` memory operand @0x219f**.

### Granularity — MEASURED (`pin_mul.sh`, REPS=3, sweep the `mul` mask bit-by-bit)

| `mul` operand mask | alignment | present / 3 |
|---|---|---|
| `0b1111111111111` | byte (bit0 free) | **3/3** |
| `0b1111111111110` | clear **bit0** → even | **0/3 — KILLED** |
| `…100` / `…000` / … (clear bit1..bit6) | 4B…64B | 0/3 |

**The threshold is bit 0 — the single-byte LSB.** Forcing the `mul`'s 8-byte load to an **even** address
kills the leak; the leaking case has an **ODD** address. *(Originally read as "misalignment triggers
STLF." But changing the mask also moves the address by a byte → loads a different 8-byte window →
**different value**. The controls below show it's the value that matters, so bit 0 most likely selects
**which value is loaded**, not alignment. See the RETRACTION.)*

**Mechanism (now well-supported): misaligned / partial store-to-load forwarding.** The transmitter
`mul qword ptr [r14+rsi]` issues an **8-byte load that must be misaligned (odd address)** so it
**partially overlaps** a preceding store; the partial STLF is resolved with speculative, secret-dependent
data → secret-dependent P+P footprint. Forcing the load even (full/no overlap) removes it. Consistent
with: **SSBD kills it** (SSB/STLF family), **`bpas` can't distinguish 46/96** (it models only aligned,
whole-store, same-address bypass — not misaligned partial forwarding) → genuine **beyond-`bpas`**.

*This "partial STLF" claim is specific to THIS violation and is now evidence-backed (bit-0 alignment
sensitivity) — unlike the earlier, retracted "partial STLF" story about the 89, which were plain V4 that
a correct `bpas` explained.*

**It's the LOAD, not the MUL — MEASURED (`exp_mul_fusion.sh`, REPS=5).** Replacing the fused
`mul qword ptr [r14+rsi]` with `mov r10, qword ptr [r14+rsi]; mul r10` (a plain unaligned 8-byte load +
a register multiply, semantically identical) **keeps the leak (5/5)**. So the memory-operand-MUL /
fused load-op path is **incidental**; the load-bearing element is the **misaligned 8-byte LOAD** itself.
Any 8-byte load at that odd address suffices.

**Load width — MEASURED (`exp_load_width.sh`).** Holding the (odd) address fixed and shrinking the
transmitter load: 8B → 3/3, 4B/2B/1B → 0/3. *(At first read this looked like "the 8-byte span is
essential → partial-overlap STLF." That reading was WRONG — see the controls below, which show the
narrowing also shrank the VALUE, and it's the value that matters.)*

**Controls to separate "8-byte ACCESS" from "64-bit VALUE" — `exp_controls.sh`:**

| variant | access | value | present / 3 | note |
|---|---|---|---|---|
| baseline `mov r10, qword…; mul` | 8B | full 64-bit | **3/3** | valid |
| **A** `… ; and r10, 0xffffffff; mul` | 8B | low 32 (intended) | ~~0/3~~ | **VOID — did not assemble** |
| **B** two 4B loads → `shl/or` → `mul` | 2×4B | full 64-bit | **3/3** | valid |

**⚠️ Control A is VOID (do not use its result).** `and r10, 0xffffffff` is an **assembler error**
(`operand type mismatch` — x86-64 `and r64, imm` takes only a sign-extended imm32; a low-32 mask needs
`mov r10d, r10d` or `movabs r11, mask; and r10, r11`). Its "0/3" was a **build failure**, not a killed
leak. **A previous retraction that leaned on Control A ("it's the value, not the access; partial STLF
refuted") is itself withdrawn.**

**What actually stands (valid experiments only):**
- `exp_load_width`: single 8B → 3/3; single 4B/2B/1B → 0/3. → the **high 4 bytes (rsi+4..rsi+7 / bits
  32–63)** of the region are needed.
- Control B: two 4B loads assembling the full 64-bit value → 3/3.
- **Open (NOT resolved):** whether the deciding factor is the **VALUE** (contents of bits 32–63) or the
  **ACCESS** to bytes rsi+4..rsi+7; and whether a partial-STLF wide-load story holds. Control B weakens
  "a *single monolithic* 8-byte load is required," but the second 4B load (rsi+4, odd) could itself
  partially forward — so partial STLF is **neither confirmed nor refuted**. `pin_mul`'s bit-0 result is
  likewise ambiguous (moving the mask moved the address → different bytes/value).
- **Redo needed:** the corrected value-mask sweep (`exp_hibits.sh`, using `movabs r11,MASK; and r10,r11`)
  to localize which high bits matter and to properly re-run the (void) Control A.

### It's the VALUE — top byte (bits 56–63) — MEASURED (`exp_hibits.sh`, all variants assemble; REPS=3)

Every variant uses the **identical 8-byte load** `mov r10, qword [r14+rsi]`; only a post-load value mask
`movabs r11,MASK; and r10,r11` differs (baseline = all-ones no-op). So the **access is held constant** and
any change in outcome is due to the **value**.

| kept bits (mask) | present / 3 |
|---|---|
| full `0xffff…ffff` (no-op) | **3/3** (sanity) |
| low 0–31 / 0–39 / 0–47 / 0–55 | 0/3 each |
| low32 + byte 40–47 / + byte 48–55 | 0/3 each |
| **low32 + byte 56–63** (`0xff000000ffffffff`) | **3/3** |

`keeplow56` (bits 0–55) **dies** but `full` (0–63) **lives** ⇒ **bits 56–63 are necessary**; and
`low32 + byte56–63` alone **reproduces** ⇒ the **top byte (MSB, bits 56–63)** is the discriminator.

**This cleanly resolves value-vs-access → it's the VALUE** (access held constant; only the value mask
changes the outcome). The transmitter is **value propagation** of the loaded top byte, **not** a load
access-geometry / partial-STLF effect. It also **reconciles every prior valid experiment**: narrow loads
(4/2/1B, `exp_load_width`) never load bits 56–63 → die; Control B's high 4-byte load (rsi+4..rsi+7)
includes bits 56–63 → reproduces. **Still store-bypass-gated (SSBD) and speculative (fence); the source
of the top byte's secret-dependence is upstream and undetermined.**

### Propagation + encoder — MEASURED (`exp_chain.sh`, all assemble; REPS=3)

Static data-flow: `mul` (L45) writes `rdx:rax = rax × V`; then `and rdx,…` (L48) and
**`movsx rdx, byte [r14+rdx]` (L49, @0x17d8)** uses **rdx as its address** (its loaded byte is dead —
overwritten by `xor rdx,rdx` at L60 — so L49 is a **pure address-type encoder**).

| variant | present / 3 | meaning |
|---|---|---|
| baseline `mul r10` | **3/3** | — |
| `imul rax, r10` (writes low product only; **rdx untouched**) | **0/3** | secret travels via the **HIGH product rdx** |
| baseline + `xor rdx,rdx` right before L49 | **0/3** | **L49 (address = rdx) IS the encoder** |

**Chain confirmed:** `… → mul (rax × V) → HIGH product rdx → L49 encoder (addr = rdx) → P+P footprint`.
This also *explains the top-byte result*: V's top byte (bits 56–63), multiplied by rax, lands in the
**high** 64 bits of the 128-bit product (rdx) — exactly the encoder's address — so masking V's top byte
removes the encoder's variation.

### Source — the secret is V, not rax — MEASURED (`exp_source.sh`, REPS=3)

| variant | present / 3 | meaning |
|---|---|---|
| baseline | 3/3 | — |
| **fix_rax** (`mov rax,const` before mul; V normal) | **3/3** | **rax is NOT the secret** — a constant multiplicand still leaks; the store-forward @0x105a is not the source |
| **fix_V** (`mov r10,const` after the load; 8B access kept) | **0/3** | **V's value IS the secret** (re-confirms `exp_hibits`) |

**The secret is the value loaded at 0x219f — concretely its most-significant byte, `mem[0x219f+7]`**
(input data that differs for inputs 46/96). This also resolves the SSBD puzzle — **secret and
speculation window are separate things:**

- **Secret:** `mem[0x219f]` top byte, loaded directly (not produced by a store-forward).
- **Speculation window:** a store-bypass (SSBD-gated) lets the `mul→L49` chain run transiently with the
  real V, hit a secret-dependent line, then squash — while the **committed** L49 access lands at 0x17d8
  for *both* inputs (so the `ct` trace matches → contract violation). *(Which store opens the window is
  still open; SSBD confirms it is store-bypass class.)*

**End-to-end chain (measured):**
`mem[0x219f+7] (secret)  →[store-bypass window, SSBD]→  mul rax×V → high product rdx → L49
movsx rdx,[r14+rdx] (encoder) → P+P cache footprint`.

### Recovery — the channel, read from the HTrace (reproduce log, `Violation Details`)

The two violating inputs' P+P HTraces (64 cache lines, `^`=hot) differ in **exactly one line**:

| | encoder line hit |
|---|---|
| architectural hits (both inputs) | 0,1,3,4,5,26,36,52 |
| **input 18 (secret A)** | **line 51** |
| **input 68 (secret B)** | **line 57** |

Everything else is identical → this single differing line **is** the leak channel: the secret
`mem[0x219f+7]`, via `mul→rdx→L49`, selects the encoder's cache line (51 vs 57), read out by Prime+Probe.
**Recoverable: yes** — an attacker observing line 51 vs 57 distinguishes the two secret values (a clean
single-line channel). Full 256-value byte recovery would need crafted inputs that step
`mem[0x219f+7]`; the channel itself is demonstrated. **The chain is now closed end-to-end.**

### Why the secret's top byte lands in the HIGH product rdx (the arithmetic)

The `mul` acts as a **spreader**: a high-position byte times a full-width number is a high-position
result, so the secret's top byte is relocated into rdx — exactly the register the gadget uses as the
encoder address. Decompose `V = V_hi·2^56 + V_lo` (V_hi = top byte, bits 56–63):

```
rax · V  =  rax·V_hi·2^56           +  rax·V_lo
            └─ ~2^71, shifted to      └─ bulk in the low 64 bits (rax_new)
               bits 56..127 → bulk
               in bits 64+ = rdx
```

The `rax·V_hi·2^56` term contributes `rax·V_hi >> 8` to rdx — **linear in V_hi with slope ≈ rax/256
≈ 2^55**, so each secret value throws rdx (and its low, line-selecting bits) somewhere new. Low bytes of
V land mostly in the low product and barely move rdx. Verified numerically (fixed rax):

| vary V's **top** byte 0x00→0x40→0x80→0xC0→0xFF | encoder line 107→103→99→95→**49** (swings wildly) |
|---|---|
| vary V's **low** byte 0x00→0x11→0x22→0xFF | encoder line 98→98→98→101 (barely moves) |

So it is specifically the **top byte** and specifically the **high product rdx**, which is exactly why
`exp_hibits` (mask V's top byte) and `exp_chain` (`imul`, no rdx) both kill it.

### Speculation confirmed — MEASURED (`fence_test.sh`, REPS=3)

| LFENCE position | present / 3 |
|---|---|
| baseline (no fence) | **3/3** |
| before `mul` @0x219f | 0/3 |
| after the @0x105a store→load pair | 0/3 |
| after the @0x105a store | 0/3 |
| before the @0x1000 byte-load | 0/3 |

Baseline reproduces; an LFENCE at **any** position kills it → the leak is **genuinely speculative /
transient** (confirmed, not a committed-path artifact). The fence is **non-discriminating for the
source**: `lfence` serializes *all* younger instructions, and the gadget is one dependent speculative
chain, so every position necessarily breaks it. (Independently, minimization already shows **all** the
remaining stores are essential.) Pinning the exact source store would need a non-serializing probe, not
fences. (This confirms the leak is speculative; it does **not** by itself pin the micro-mechanism — see
the controls and RETRACTION above.)

---

## Mechanism — candidate (NOT yet nailed)

**Two candidate mechanisms, both beyond `bpas`, distinguished by one experiment:**

1. **PSF — AMD Predictive Store Forwarding** (SPEC_CTRL bit 7, mitigated by **PSFD**). The CPU
   *predicts* a load aliases a recent store and forwards the store data **before addresses are
   verified**; a wrong prediction is the transient window. This is the *opposite direction* of `bpas`
   (which models a load **bypassing** an older store), so `bpas` structurally cannot represent it.
   The CPU is a **Ryzen 7 8700F (Zen 4, family 0x19) → PSF-capable**.
2. **Classic SSB variant** — an **unaligned / size-mismatch / partial-overlap store-to-load
   forwarding** bypass (e.g. the `0x1000` 8B-store → byte-load → qword-load size mismatch, or
   forwarding across atomics / intervening stores) that the `bpas` speculator misses because it only
   models a **single, whole-store, same-address, immediate-next** bypass.

**The discriminator (decisive):** SSBD mitigates *both* SSB and PSF, so "SSBD ON kills it" cannot tell
them apart. The clean test toggles **PSFD independently with SSBD OFF**:

| SSBD | PSFD | leak present? | conclusion |
|---|---|---|---|
| 0 | 0 | **yes, 5/5** (baseline) | — |
| 1 | 0 | **no, 0/5** | SSBD mitigates → **SSB family** (solid: SSBD *is* enumerated & works) |
| **0** | **1** | **yes, 5/5** | **INCONCLUSIVE** — see caveat: PSFD not enumerated on this CPU |

**RESULT (measured `psfd_test.sh`, N=5, 2026-07):** `A=5/5, B=0/5, C=5/5`.

- **Solid:** it is **SSB family** — SSBD (a genuinely enumerated, working control) turns it off.
- **NOT yet resolved — classic SSB vs PSF:** the C row (SSBD=0, PSFD=1) was meant to discriminate, but
  **CPUID Fn8000_0008 EBX[28] (PSFD) = 0** on this Ryzen 8700F → the CPU **does not enumerate the PSFD
  control**. Writing SPEC_CTRL bit 7 was *accepted and retained* (no `Not able to set MSR` in dmesg, and
  a rejected write would have aborted the run via `CHECK_ERR`→`-EIO`→`measurement.c:179`), but on a part
  that doesn't enumerate PSFD, a retained reserved bit **may be architecturally inert**. So "survives
  PSFD=1" does **not** prove "PSF was disabled and the leak persisted." **We cannot claim not-PSF.**

*Implemented for the test:* a new executor knob **`enable_psfd`** (`/sys/rvzr_executor/enable_psfd`,
SPEC_CTRL bit 7, default off) + `psfd_test.sh`. It is a valid discriminator only on a CPU that
enumerates PSFD, or once validated by a positive control.

*To actually resolve PSF vs classic-SSB:* (a) run on a CPU with `psfd` enumerated, **or** (b) add a
**known-PSF positive-control gadget** and confirm `enable_psfd=1` kills *it* on this box — if it does,
the knob works here despite the missing flag and our "survives" becomes meaningful; if it doesn't, bit 7
is inert here and PSFD cannot be tested on this machine.

**Calibration / caveat (lesson learned):** the de-align experiment changes *every* address, so
"violation disappears when aligned" is **suggestive, not conclusive** that partial overlap is the
mechanism — any address perturbation can break a store-bypass gadget. (We previously over-claimed a
"partial STLF" story on a case that turned out to be Bug-B-induced plain V4.) To nail it:

1. **Fence pass** — LFENCE between the store and the dependent load: if it kills the violation, the
   chain is a genuine speculation window.
2. **Targeted alignment** — align *only* the one suspect access (e.g. `0x1000`), not all, to isolate
   the partial-overlap variable.
3. **Input-diff on the clean minimized gadget** — pin the leaked byte (`^`). The earlier run reported
   "Leaked 0 bytes" because input-diff ran on the *noisy full* program; the minimized gadget is clean.
4. **Compare to known SSB variants** — partial STLF, store-buffer forwarding (MDS/Fallout), SPOILER —
   to state the relationship precisely.

---

## Reproduction (all SSBD off; system `spec_store_bypass = Vulnerable`; SMT off recommended)

```bash
cd /home/mluo/sca-fuzzer/rvzr/SpecRL/hierarchical_testing

# robustness + verdict (minimize against fast path, then verify with full priming)
VERIFY_N=12 RETRIES=12 ./minimize_and_verify.sh violation-260701-234841

# is it beyond bpas?  (ctrace hashes for 46 vs 96 under bpas / cond-bpas / seq)
#   -> bpas: 46==96 (can't explain);  cond-bpas==bpas != seq
# SSB family?  ./ssbd_test.sh-style toggle on minimized.asm  -> SSBD on = Violations 0
# alignment?   de-align minimized.asm (0b...111 -> 0b...000) -> Violations 0
```

Artifacts in `violation-260701-234841/`: `minimized.asm`, `minimized_aligned.asm`, `min_inputs/`,
`mv_minimize.log`, `mv_verify_*.log`.

---

## Status / next steps

- [~] **PSFD test (PSF vs classic SSB)** — ran (`A=5/5, B=0/5, C=5/5`) but **INCONCLUSIVE**: this CPU
      does not enumerate PSFD (CPUID Fn8000_0008 EBX[28]=0), so bit 7 may be inert. Confirms SSB family
      (B=0), does *not* settle PSF-vs-classic.
- [ ] Resolve PSF vs classic-SSB: known-PSF **positive control** on this box (does `enable_psfd=1` kill a
      known PSF gadget?), or re-run on a CPU that enumerates `psfd`.
- [x] Targeted single-access de-align (`pin_geometry.sh`) → transmitter = `mul [r14+rsi]` @0x219f (only
      access whose alignment matters; both store→load sites are NOT critical).
- [x] Granularity sweep (`pin_mul.sh`) → threshold is **bit 0** (odd address needed) → misaligned /
      partial store-to-load forwarding; plain cache-line leak ruled out.
- [x] LFENCE sweep (`fence_test.sh`) → baseline reproduces, any fence kills it → **speculative/transient
      CONFIRMED**. (Fences can't localize the source store — blunt serialization on a single dep chain.)
- [x] Load-op vs load (`exp_mul_fusion.sh`) → it's the LOAD, not the MUL (mul→mov+mul keeps it 5/5).
- [x] Load-width sweep (`exp_load_width.sh`) → 8B leaks, 4/2/1B die. *(Initially misread as partial STLF.)*
- [~] Controls A/B (`exp_controls.sh`): **A VOID** (assembler error — `and r64,imm` needs imm32/movabs),
      **B valid** (two 4B loads → 64-bit → 3/3). Net: high 4 bytes (bits 32–63) needed; value-vs-access
      and partial-STLF **unresolved**. Prior "partial STLF" claim AND its retraction both withdrawn.
- [x] Value-mask sweep (`exp_hibits.sh`, `movabs`): access held constant → **it's the VALUE**, and the
      discriminator is the **top byte (bits 56–63)** (keeplow56 dies, low32+byte56–63 lives). Reconciles
      `exp_load_width` + Control B. VALUE-vs-ACCESS resolved → value; partial-STLF-transmitter story out.
- [x] Propagation + encoder (`exp_chain.sh`): **high product rdx** (imul dies) → **encoder = L49**
      `movsx rdx,[r14+rdx]` (killrdx dies). Chain `mul→rdx→L49` confirmed; explains the top-byte result.
- [x] SOURCE (`exp_source.sh`): constant `rax` still leaks (3/3), constant `V` does not (0/3) → **the
      secret is `mem[0x219f]` (top byte), not the store-forwarded rax**. Store-bypass = the window, not
      the secret.
- [x] RECOVERY: HTrace shows the encoder hits **line 51 (input 18) vs line 57 (input 68)** — a clean
      single-line secret-dependent channel → **recoverable**. Chain closed end-to-end.
- [ ] (optional capstone) Craft inputs stepping `mem[0x219f+7]` to plot the full byte→line map.
- [ ] (optional) which store opens the transient window; narrow bits 56–63 to exact bit(s); PSF vs SSB.
- [ ] (optional) PSF vs classic-SSB on a CPU that enumerates PSFD, or via a known-PSF positive control.
- [ ] (optional) PSF vs classic-SSB on a CPU that enumerates PSFD, or via a known-PSF positive control.
- [ ] Input-diff on the minimized gadget (pin the leaked byte).
- [ ] Position vs known SSB variants (partial STLF / MDS-Fallout / SPOILER).
- [ ] **Enable priming inside training** (`hi_SpecEnv.py`) so flaky candidates are filtered at the
      source instead of re-checked for hours afterward.
- [ ] Re-fuzz with the fixed contract to see how many *robust* beyond-`bpas` candidates exist.

---

## Bottom line

A **real, robust, minimal** speculative store-bypass **variant** on AMD Zen that the Revizor `bpas`
execution clause **cannot model** — surfaced by the RL fuzzer only **after** fixing two contract bugs
that had been flooding the search with plain-V4 false positives. The exact micro-mechanism
(most likely unaligned/partial store-to-load forwarding) still needs the four confirmation steps above
before it can be claimed precisely.
