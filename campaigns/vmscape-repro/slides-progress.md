---
marp: true
title: Cross-VM vBTI on Revizor — Progress
paginate: true
---

# Cross-VM Branch Target Injection (vBTI) on Revizor

**Extending Revizor with guest-user actors to fuzz cross-domain BTB leaks**

Progress, obstacles, and findings

<!-- Zen4 (Ryzen 7 8700F), microcode 0xa705206 (dis_ucode_ldr), BTB virtually-indexed & NOT domain-tagged -->

---

## Goal

- **VMSCAPE / Enter-Exit paper** already tested many host↔guest vBTI cases → reproducing them adds little.
- Our extension: **Revizor now supports guest-user actors**.
- **Target**: fuzz the configurations the paper could *not* cover —
  - guest-user → guest-user
  - guest-user → guest-kernel
- **Platform**: AMD Zen4, microcode reverted `0208→0206` (`dis_ucode_ldr`) so the BTB channel stays open; BTB is virtually-indexed and **not** tagged by domain/VMID.

---

## Taxonomy — two axes that matter

**Direction**
- **Residue**: victim trains → observer mispredicts & *self*-leaks (mispredictor = reader)
- **Injection**: attacker trains → *separate* victim mispredicts & leaks (mispredictor ≠ reader)

**Two halves of any injection**
- **Steering** — attacker's BTB training makes the victim's branch mispredict into a gadget
- **Readout** — someone recovers the secret the transient leaked

> The steering half is the novel primitive; the readout half is where it gets hard.

---

## What REPRODUCES — STEERING only (self-observe readout)

> All three keep the readout **inside the mispredictor** (self-observe, F+R): the attacker *steers*,
> the victim leaks & reads its **own** secret. This is **not** full injection.

| direction (trainer → mispredictor) | who reads | status |
|---|---|---|
| residue: vmvictim → vm | vm (itself) | ✅ TC 1113 |
| injection-dir u→u: vm → vmvictim | vmvictim (itself) | ✅ steering |
| injection-dir **u→k**: vm → vmvictim (**kernel**) | vmvictim (itself) | ✅ steering |

**Method — per-TC causal test** (`reproduce_trainer_test.sh`):
byte-exact strip of the trainer's `jmp` → `nop;nop`, then `revizor.py reproduce` A/B
→ **ORIGINAL 3/3 violate, NO-TRAINER 0/3** ⇒ the training branch *causes* the misprediction.
Programs had **0 conditional branches** (not Spectre-v1) and arch-target == SLS fall-through (rules out SLS).

**⇒ full injection — attacker *reading the victim's* secret — FAILS (see "The HARD problem").**

---

## Finding 1 — the "alignment wall" is about VA *base*, not privilege

BTB collision needs both branches at the **same virtual address**.

```c
// page_tables_guest.h
#define GUEST_V_MEMORY_START 0x0ULL   // every guest maps its code here
```

- All guests execute at **GVA 0x0 + offset** (separate address spaces) → same-offset branches **collide**.
- **u→k works** because privilege only flips the PTE U/S bit — it does **not** move the GVA.
- The wall blocks host↔guest / K2U because those live at **different VA bases**, not because of privilege.

---

## Finding 2 — IBPB-on-VM-switch is a working mitigation

Added `mov ecx,0x49; mov eax,1; wrmsr` (IBPB) right before `vmrun` in `body_macro_switch_h2g`:

| build | IBPB | strong TC-1113 residue |
|---|---|---|
| `1CD139EF` (clean) | no | reproduces 3/3 |
| `A07CAEF9` (IBPB) | before `vmrun` | **0 — gone** |

Same box, env, test case, inputs — only IBPB changed ⇒ **IBPB closes the cross-VM BTB channel**
(matches real VMSCAPE mitigation). Also closes u→k.

---

## The HARD problem — attacker-side EXFILTRATION does not reproduce

Steering works, but getting the **attacker** to *read* the victim's secret fails.

- Residue/self-observe reads trivially because **mispredictor = reader** (same round, F+R on its own footprint).
- Real injection needs a **different actor** to read → forces a `vmrun`/`vmresume` **actor switch** between "victim leaks" and "attacker probes".

**Attempt (P+P cross-actor):** the switch's memory traffic **obliterates the L1D prime**
→ `R:2731` rerun storm + "architectural mismatch" bugs → *unmeasurable*.

---

## Why the readout wall is fundamentally hard

Two candidate channels, both blocked:

- **P+P (cache contention)** — needs the L1D prime to survive from victim→attacker.
  The mandatory VM switch evicts L1D → prime destroyed → non-deterministic → arch mismatch.
- **F+R (flush+reload)** — needs the two guests to **share a physical frame**.
  `hpa_gpa` is only an L1TF-numeric GPA collision, *not* real sharing; a real shared frame (EPT alias)
  is **unrealistic** (guests don't share memory) — we reverted it.

⇒ On single-thread Revizor there is **no realistic cross-actor readout**; only self-observe works.

---

## Why host↔guest is barely testable here

- Guests run at **GVA 0x0**; the **host runs its code at its raw vmalloc'd kernel VA** (`0xffff…`).
  → host & guest branches never share a VA → **no BTB collision** → campaigns come back clean for
  *layout* reasons, not security reasons (a meaningless "pass").
- **K2U is impossible**: host-kernel & host-user are **one** address space — two branches can't share a VA.
- Would require executor surgery (remap host code to a guest GVA) **and** still hit the readout wall.

---

## Practical gotchas (cost real time)

- **Weak, near-threshold signal** — residue fires ~1/hundreds–2000 TCs; a marginal saved violation flips
  clean/violation run-to-run. Use a *strong* saved case or a full deterministic sweep before concluding.
- **`perf_env.sh` after every reboot** — a reboot resets governor/EPP/C-states; F+R then gets noisy
  (`R:` rerun explosion, ~3.5× slower). Microcode 0206 *does* survive via `dis_ucode_ldr`.
- **Reproduce exit code** = `violations>0`: `1`=violation, `0`=clean.

---

## Current work — fuzzing the guest-user configs

Built on Revizor's multi-actor fuzzing skeleton (`demo/tsa-l1d`), `ct-ni` contract, `random_instructions` bodies:

- `config-fuzz-u2u.yaml` — guest user → guest user
- `config-fuzz-u2k.yaml` — guest user (attacker) → guest kernel (victim)
- `fuzz-multiactor.asm`, `run_fuzz.sh [u2u|u2k]`

Goal: discover **any** cross-actor speculative leak (v1 or BTB) the Enter-Exit paper couldn't reach.

---

## Summary & next steps

**Established**
- Cross-guest **steering** reproduces in all directions incl. **user→kernel**, causally proven.
- Alignment wall = VA *base*, not privilege. **IBPB mitigation validated.**

**Open / hard**
- Attacker-side **exfiltration** blocked by the cross-actor readout wall (P+P dies at the switch; F+R needs shared memory).
- Host↔guest structurally untestable without executor changes.

**Next**
- Fuzz u2u / u2k for new leaks.
- Step 4: **L2/LLC Prime+Probe** in the executor (survives the VM switch) to attempt real exfiltration.
