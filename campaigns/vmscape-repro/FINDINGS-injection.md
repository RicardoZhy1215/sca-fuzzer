# vBTI injection in Revizor — REPRODUCED

_Companion to `FINDINGS-bti-alignment.md`. This one is about the **injection** primitive itself
(does an attacker's BTB training steer a victim's speculation, and can the leaked data reach the
attacker), separate from the branch-alignment question._

## Result: cross-domain vBTI injection reproduced (2026-08-11)

`campaigns/vmscape-repro/out-v2v-fuzz/violation-260811-214352` (also the smoke-run hit) is a
**confirmed cross-domain branch-target-injection leak**: a guest observer recovers a *different*
guest's secret purely through BTB misprediction. Verified on all three axes:

| Check | Evidence |
|---|---|
| **Speculative, not architectural** | the two counterexamples share the same contract hash (`17099477998017268125`) but differ in hardware trace ⇒ the leak is beyond the contract |
| **BTB (indirect jump), not Spectre-v1** | the violating program has exactly **2 indirect jmps and 0 conditional branches** |
| **Cross-domain, not self-leak** | the observer's (vm, actor 2) own input is **byte-identical** across the counterexamples; swapping **only the victim's** (vmvictim, actor 1) data flips the observer's trace; swapping `main` does nothing |

Mechanism, end to end: **vm (guest) trains the BTB → vmvictim (guest) mispredicts into its own
disclosure gadget → leaks its own secret transiently → vm reads it via the EPT-aliased shared frame.**

### The recipe (what it takes)

1. **Two guest actors** (same guest-virtual base ⇒ their indirect jmps share a VA "for free" — the
   alignment K2U/GS→HU can't get; see `FINDINGS-bti-alignment.md`).
2. **EPT physical alias** — `vm.data_alias = "vmvictim"` + the executor change in
   `set_extended_page_tables` (`page_tables_guest.c`) that backs the observer's MAIN page with the
   victim's physical frame. `hpa_gpa_collisions` alone is NOT enough (it's an L1TF-numeric GPA
   collision; the EPT still gives each guest its own frame — that's why the first attempt was flat).
3. **`random_instructions` in the gadget bodies** — the clean hand-written PoC was *frozen* (the
   vm→vmvictim misprediction never fired); the fuzzer-filled preambles/gadget supply the
   microarchitectural context that makes it fire. Let the fuzzer find the leaking instances.

Artifacts: `template-v2v-inject-fuzz.asm`, `config-v2v-inject.yaml`, `run_v2v_fuzz.sh`. Executor
build with the EPT alias: `srcversion 81507044…`. Channel sanity check: `verify_v2v_channel.sh`
(observer reads victim data, 8/8 model-vs-hardware divergence).

### What was ruled out on the road here (all with direct traces)
- Channel dead? No — poscontrol 8/8 divergence (EPT alias works).
- Cache lost across VMEXIT? No — cachexfer probe: victim's imprint reaches the observer, 11/12.
- Window too short? No — wide dependent-chain variant still frozen (so it wasn't the window).
- Cross-guest steering broken? The *clean* hand-written version is frozen in **both** directions,
  yet the fuzzer-generated 585 (and now the injection) leak — so the missing ingredient is the
  `random_instructions` context, not a broken predictor or an IBPB on VM entry/exit (there is none).

## The precise taxonomy (this is the whole point)

A full vBTI / VMSCAPE attack is two independent steps. Conflating them caused an earlier mislabel
("V2V is just residue, nothing like injection") — that was wrong. The steps are:

1. **Predictor steering** — attacker's indirect-branch training makes the *victim's* indirect branch
   mispredict into a chosen gadget. (Needs same-VA branches + no BTB flush across the domain switch.)
2. **Cross-domain data readout** — the victim's mispredicted gadget leaks the *victim's* data into a
   channel the *attacker* can read. (Needs attacker and victim to share physical memory.)

| | Step 1: steering | Step 2: data readout |
|---|---|---|
| Status in this harness | ✅ **works** (with `random_instructions`) | ✅ **works** (EPT physical alias) |
| Needs | same-VA branches (free between two guests) + fuzzed gadget context | shared physical backing (EPT alias) |

Both steps now close together — see the reproduction above. The sections below are the historical
diagnosis (kept for the reasoning trail); the two blockers they describe (dead channel, then a
frozen clean gadget) are the two items in the recipe above.

## Step 1 is already proven — by the "residue" violation, read carefully

The V2V `out/violation-*` (test case 585, from `template.asm` + `random_instructions`) was
re-examined instruction by instruction:

- The whole program has **exactly two branches, both hand-written indirect jmps, zero conditional
  branches** — so it is genuinely the BTB mechanism, not an incidental Spectre-v1.
- `vmvictim`: `… ; lea rax,[vmvictim_1]; jmp rax` → trains `BTB[jmp] = vmvictim_1`.
- `vm` (observer): loads its own reg-init (`mov rax,[r14+0x2000] …`), then `lea rbx,[vm_3]; jmp rbx`
  → architecturally `vm_3`, but **mispredicts to the target `vmvictim` trained** → transient
  `[r14+reg]` accesses.

So one guest's training steers another guest's speculation. Calling it "residue" (vm observes
vmvictim's leftover) or "injection" (vmvictim injects into vm) is two framings of the **same event** —
**Step 1 works**. The transient window and single-pass training are clearly adequate.

But note *whose data leaks*: the contract trace shows `vm`'s code (`0x60xx`) touching **`0x9xxx–0xbxxx`
= vm's own data**, while `vmvictim`'s data is at `0x5xxx`. **`vm` self-leaks its own reg-init secret**
along the mispredicted path; there is **no cross-guest data access**. That is Step 1 without Step 2.

## Step 2 is missing — and the V2V-injection run proved it

`template-v2v-inject.asm` was built to do the *full* thing: attacker (`vm`) trains, a *separate*
victim (`vmvictim`) mispredicts and leaks **the victim's** secret, and the observer (`vm`) reads it.
Alignment was byte-verified free (both jmps at `0x3f`, both disclosures at `0x4f`; same guest base).

Result: **clean, `Cls:0/0`** — the observer's F+R trace never varied with the victim's secret. Root
cause (confirmed from the executor source, not guessed):

`x86_enable_hpa_gpa_collisions` collides only the **numerical GPA** (an L1TF/Foreshadow trick); the
EPT still backs each guest with its **own** frame:

```c
// set_extended_page_tables — page_tables_guest.c:336-339 (the EPT: GPA -> real HPA)
for data pages:
    void *hva = &sandbox->data[actor_id] + i*PAGE_SIZE;   // actor's OWN backing, always
    set_ept_entry(ept_base, ..., hva);
```

```
victim   guest-VA → GPA(=main's HPA) → victim's EPT   → victim's backing HPA
observer guest-VA → GPA(=main's HPA) → observer's EPT → observer's backing HPA   (DIFFERENT frame)
```

L1 is tagged by the real HPA (each guest's own backing), so two guests **do not share cache lines**
for a normal Flush+Reload. The GPA collision only matters under a terminal fault (L1TF), not F+R.
That is why the residue test never needed Step 2 (it self-leaks and self-reads), and why the
full-injection template can't observe the victim's imprint.

## The fix that closed Step 2 (built + verified)

Give two guests a **real shared physical frame**: make the **EPT** map the observer guest's MAIN data
page to the victim's backing (`hva = &sandbox->data[alias_target]` for `MAIN_PAGE_ID`, gated on the
existing `alias_target`) in `set_extended_page_tables` (`page_tables_guest.c`). This is the guest/EPT
analog of the host↔host physical alias (`set_host_data_alias`, verified in K2U). Unlike the host alias,
**no per-input lifecycle is needed**: the EPT is rebuilt per test case and `load_sandbox_data` writes
each actor's input to its own host-side backing directly (not through the EPT), so there is no
load-time clobber. Only `MAIN_PAGE_ID` is aliased (reg-init/faulty stay the actor's own).

Verified by `verify_v2v_channel.sh`: the observer reads the victim's data, 8/8 model-vs-hardware
divergence. Then the *full* injection reproduced — see "Result" at the top.

Status: **DONE.** Both steps close; cross-domain injection reproduced (`out-v2v-fuzz/violation-260811-214352`).

## Artifacts
- `template-v2v-inject.asm` (+ `-no-jmp`), `config-v2v-inject.yaml`, `run_v2v_inject.sh` — the
  full-injection attempt; clean because Step 2's channel is dead.
- `out/violation-*` — the residue violation that proves Step 1 (re-read here).
