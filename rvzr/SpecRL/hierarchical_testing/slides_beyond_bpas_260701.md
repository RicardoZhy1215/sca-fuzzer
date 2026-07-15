---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- _paginate: false -->

# A *Beyond-`bpas`* Speculative Store-Bypass Leak on AMD Zen 4

### Found by RL-guided leakage-contract fuzzing (SpecRL / Revizor)

**`violation-260701-234841`** — mechanism pinned by single-variable `asm` controls

<br>

AMD Ryzen 7 8700F (Zen 4) · SSBD off · Prime+Probe

---

## Leakage-contract fuzzing in one slide

- **Contract** = (observation clause, execution clause). Here: **`ct` + `cond-bpas`**.
- **Violation** = two inputs with the **same contract trace** but a **different hardware trace** (Prime+Probe cache footprint).
- **`bpas`** = the execution clause that models **Spectre-v4** (speculative *store bypass*).
- **Target**: AMD Ryzen 7 8700F (Zen 4), **SSBD off**, SMT off.

> A *beyond-`bpas`* violation = hardware leaks something the strongest store-bypass contract does **not** predict.

---

## First: two contract bugs — both faked "violations"

- **Bug A — wrong contract in the worker.** Ray rollout workers re-import `CONF` with defaults → the model silently used the default contract.
- **Bug B — `cond-bpas` degenerated to `seq`.** An MRO off-by-one (`super(StoreBpasSpeculator, self)…`) meant the store-bypass logic was **never called** → the contract did **no speculation**.

**Effect:** ~89 earlier "violations" were **plain Spectre-v4** a *correct* `bpas` explains — **false positives**.

✅ Both fixed → *only then* did the fuzzer surface a genuine beyond-`bpas` case.

---

## The finding: `violation-260701-234841`

| property | evidence |
|---|---|
| **Robust** | minimized gadget survives priming **12/12** (~37 s) |
| **Beyond `bpas`** | correct `bpas`: the two inputs are **identical** |
| **SSB family** | **SSBD ON → 0** |
| **Speculative** | **any `LFENCE` → 0** |

The minimized gadget (key lines):

```asm
mul     qword ptr [r14+rsi]        ; V = mem[0x219f];  rdx:rax = rax * V
and     rdx, 0b1111111111110
movsx   rdx, byte ptr [r14+rdx]    ; ENCODER: load addr = rdx     @0x17d8
```

---

## How we pin the mechanism: single-variable `asm` controls

- Take the **minimized** gadget; change **one thing**; reproduce (full priming).
- **The decisive signal is "the violation VANISHES."** Removing feature *X* and losing **all** violations ⇒ *X* is **necessary**. (Robust — independent of which input pair.)
- A variant that **still leaks** shows *X* is *not required* (corroborating).
- **Rule learned the hard way:** *verify every edit assembles* — a variant that fails to build also reports "0", and can be mistaken for a real "kill."

> Below: three comparisons that isolate **access vs value**, **propagation path**, and **the secret**.

---

## Comparison 1 — is it the ACCESS or the VALUE? (a)

**Load-width sweep** — shrink the transmitter load, same address:

| load | 8B | 4B | 2B | 1B |
|---|---|---|---|---|
| leaks? | **yes** | no | no | no |

First read: *"needs the 8-byte span → partial-overlap STLF."*
⚠️ **Premature** — narrowing the load shrinks *both* the access **and** the value. The two are confounded.

**Control A** (`mov r10,[…]; and r10, 0xffffffff; mul`) — meant to keep the 8B access but low-32 value…
❌ **did not assemble** (`and r64, imm` takes only imm32). Its "0" was a *build failure*; a retraction that leaned on it was **withdrawn**.

---

## Comparison 1 — is it the ACCESS or the VALUE? (b)

**Clean version — hold the access constant, mask only the value** (`movabs r11,MASK; and r10,r11`):

| kept bits | full | 0–31 | 0–39 | 0–47 | 0–55 | low32 + **56–63** |
|---|---|---|---|---|---|---|
| leaks? | yes | no | no | no | **no** | **yes** |

- Same 8-byte load every time ⇒ outcome depends **only on the value**.
- `keeplow56` (0–55) dies, `full` lives ⇒ **bits 56–63 (top byte) are necessary**.

**⇒ It is the VALUE — specifically the top byte — not the load geometry.**
Reconciles the width sweep: 4/2/1-byte loads never load bits 56–63.

---

## Comparison 2 — how does it propagate?

`mul r10` writes the full 128-bit product `rdx:rax`. Two edits:

| variant | what changes | leaks? |
|---|---|---|
| `imul rax, r10` | writes **low product only**; `rdx` untouched | **no** |
| baseline + `xor rdx,rdx` before L49 | zero `rdx` at the encoder | **no** |

- `imul` dies ⇒ the secret rides the **HIGH product `rdx`**, not the low product.
- Zeroing `rdx` dies ⇒ the encoder is **`L49: movsx rdx,[r14+rdx]`** (address = `rdx`).

**⇒ propagation = high product `rdx` → encoder `L49`.** (Its loaded byte is later discarded ⇒ a *pure address-type* transmitter.)

---

## Comparison 3 — what is the secret, V or rax?

`rdx = high(rax × V)`. Fix each operand to a constant:

| variant | leaks? | meaning |
|---|---|---|
| `mov rax, C` before mul (V normal) | **yes** | multiplicand **rax is not the secret** |
| `mov r10, C` after the load (V replaced) | **no** | **V's value IS the secret** |

**⇒ the secret is `V = mem[0x219f]`** — concretely its **top byte `mem[0x219f+7]`** (input data differing across the two inputs).

**Resolves the SSBD puzzle:** *secret* and *speculation window* are separate — V is the secret; the **store-bypass** (SSBD-gated) merely provides the transient window in which `mul→L49` runs and is then squashed, while the **committed** encoder access is the same for both inputs (⇒ `ct` matches ⇒ violation).

---

## Why the secret's top byte → the **high** product `rdx`

`mul` is a **spreader**: a high-position byte × a full-width number = a high-position result.
`V = V_hi·2^56 + V_lo`:

```
rax · V = rax·V_hi·2^56   +   rax·V_lo
          ~2^71 → bits 64+     bulk in the low 64 bits
          = rdx               (rax_new)
```

Contribution to `rdx` = `rax·V_hi >> 8` — **linear in the secret byte**. Verified numerically (fixed rax):

| vary **top** byte `00→40→80→C0→FF` | encoder line `107→103→99→95→49` (swings) |
|---|---|
| vary **low** byte `00→11→22→FF` | encoder line `98→98→98→101` (flat) |

⇒ specifically the **top byte** and specifically **`rdx`** — exactly why Comparisons 1 & 2 kill it.

---

## End-to-end chain + the recovered channel

```
mem[0x219f+7] (SECRET) → [store-bypass window, SSBD] → mul rax×V
   → HIGH product rdx → encoder movsx rdx,[r14+rdx] → Prime+Probe
```

HTraces of the two violating inputs differ in **exactly one cache line**:

```
architectural hits (both):  0 1 3 4 5 26 36 52
input A:  + line 51      input B:  + line 57
```

**Recoverable:** observing line 51 vs 57 distinguishes the secret values — a clean single-line channel.

---

## Status & open questions

**Fully characterized (each link a controlled `asm` edit):**
secret → store-bypass window → `mul` high product → encoder → Prime+Probe channel.
A genuine **beyond-`bpas` SSB-family** leak on Zen 4.

**Open (do not affect the above):**
- *Which* store opens the window (SSBD confirms store-bypass class).
- **PSF vs classic SSB** — untestable here: CPU does not enumerate PSFD (`CPUID Fn8000_0008 EBX[28]=0`).
- Full 256-value byte→line recovery curve.

**Methodology:** validate the *contract* before the microarchitecture; isolate one variable per edit; **treat "kills it" as the decisive signal**; verify every edit assembles.
