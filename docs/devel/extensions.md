# Revizor Extensions (guest-user + configurable aliasing)

This document records the extensions made to Revizor (the `side-channel-fuzzer` / `sca-fuzzer`
tree) on top of the multi-actor "Enter, Exit, Page Fault, Leak" release, to support testing
branch-target-injection (BTI) leaks across the host↔guest boundary.

Upstream supported only three actor kinds (per the paper's footnote):
**host-kernel, host-user, guest-kernel**. These extensions add:

1. **Guest-user actors** — `mode: guest` + `privilege_level: user` (the 4th quadrant).
2. **A model/executor consistency fix** for the faulty-page user bit (fixes false arch-mismatches).
3. **Configurable guest aliasing** (`data_alias`) — lets a guest actor share a physical F+R
   channel with an arbitrary *host* actor, not just `main` (actor 0).

All three are backward compatible: existing configs behave exactly as before (defaults preserve
the original behavior).

---

## Extension 1 — Guest-user actors (`mode: guest` + `privilege_level: user`)

Lets an actor run inside a VM **at CPL 3**. Enables V2VU / cross-VM tests where the attacker or
victim VM runs in user mode.

### Config

No new option — the previously-rejected `mode: guest` + `privilege_level: user` combination now
works:

```yaml
actors:
  - vm_user:
    - mode: "guest"
    - privilege_level: "user"
```

### Executor (kernel module) — put the guest into CPL 3

| File | Change |
|---|---|
| `rvzr/executor_km/x86/vmx.c` | `set_vmcs_guest_state()` takes `actor_id`; user actors get CPL-3 guest CS/SS selectors (`0x13`/AR `0xa0FB`, `0x23`/AR `0xc0F3`) instead of the DPL-0 ones. Intel takes the VM-entry CPL from CS.RPL. |
| `rvzr/executor_km/x86/svm.c` | `set_vmcb_guest_state()` takes `actor_id`; `save->cpl = is_user ? 3 : 0`, CS/SS get DPL=3. |
| `rvzr/executor_km/x86/page_tables_guest.c` | New `set_guest_user_pages()` sets the U/S bit on **all** present guest pages (code/data/util/GDT) — a CPL-3 guest must be able to reach its own memory. |
| `rvzr/executor_km/x86/macros.c` | New `body_macro_switch_gu2h` = a single `PAUSE`. `vmcall`/`vmmcall` #UD at CPL 3, so a user guest leaves the VM with PAUSE (unconditionally intercepted by VMX `PAUSE_EXITING` / SVM `INTERCEPT_PAUSE`; changes no regs/flags). Registered as `TYPE_SWITCH_GU2H`. |
| `rvzr/executor_km/include/macro_expansion.h` | New `TYPE_SWITCH_GU2H` subtype. |
| `rvzr/executor_km/macro_expansion.c` | `switch_g2h` dispatches to `GU2H` (PAUSE) vs `G2H` (vmcall) by the owner's PL; `get_macro_subtype_from_id()` takes `owner` + bounds-checks it. |
| `rvzr/executor_km/test_case_parser.c` | Fixed a `break` bug so `includes_vm_actors` and `includes_user_actors` can both be set; k2u/u2k macro targets additionally asserted `MODE_HOST`. |

### Model (Unicorn) — match the CPL-3 architectural semantics

`rvzr/model_unicorn/interpreter.py`:
- `interpret_instruction()` dispatches on `(mode, pl)`; adds a guest-user branch.
- New `_X86GuestUserspaceInterpreter`: its "must-exit" instruction set is the **union** of the VM
  set and the userspace set (a guest-user actor is constrained by both).
- New abstract `_emulate_guest_userspace_execution()` (ARM: no-op).

### Frontend / generator

| File | Change |
|---|---|
| `rvzr/elf_parser.py` | Fixed the macro-target validation (the original `and` logic wrongly rejected a guest-user `h2g` target). Now: k2u/u2k targets must be host + matching PL; **h2g target must be a guest (any PL)**; g2h target must be host. |
| `rvzr/arch/x86/generator.py` | `_X86U2KAccessPass` (Meltdown u→k redirect) skips guest actors — the kernel sandbox isn't in the guest address space. |

### Status
Validated: guest-user↔guest-user VMSCAPE reproduces (see `campaigns/vmscape-repro/out/`,
`violation-260807-055254`). Not supported: kernel→user transition *inside* a single guest (each
guest actor is its own VM; that would need a guest GDT/TSS/IDT + syscall MSRs).

---

## Extension 2 — Faulty-page user-bit model/executor consistency fix

**Symptom:** with two guest-user actors, essentially every detected violation was flagged as an
architectural mismatch (stored in `out/bugs/`), not a real violation.

**Root cause:** the executor maps *all* pages of a user actor (including the **faulty page**) as
user-accessible (`set_guest_user_pages` / `set_user_pages`), but the model's `induce_user_faults`
judged the faulty page's user-accessibility from `data_properties.user` alone (default False). So a
user actor touching its **own** faulty page → the model injected a #GP while the hardware did not →
the final register state diverged → arch mismatch on nearly every test case.

**Fix** — `rvzr/model_unicorn/interpreter.py` (`_FaultInterpreterCommon.load_test_case`):

```python
self._faulty_page_user_accessible[aid] = (
    self._page_is_user_accessible(pte) or actor.privilege_level == ActorPL.USER)
```

A user actor's own faulty page is now treated as user-accessible, matching the executor. Kernel
actors are unaffected, so Meltdown/K2U semantics (user touching **another** actor's kernel faulty
page → #GP) are preserved. Verified: 100/100 historical bug archives drop from 16 model faults → 0;
`unit_model` passes.

---

## Extension 3 — Configurable guest aliasing (`data_alias`)

`x86_enable_hpa_gpa_collisions` makes a guest actor's guest-physical addresses collide with a host
actor's host-physical addresses — the mechanism that builds a shared F+R channel. Upstream
hardcodes that host actor to **actor 0 (`main`)**. This extension makes the target configurable, so
a guest attacker can share a channel with a **specific host victim** (needed for vBTI_GS→HU, where
the victim is host-user, not `main`).

### Config

```yaml
actors:
  - victim:
    - mode: "host"
    - privilege_level: "user"
  - attacker:
    - mode: "guest"
    - privilege_level: "kernel"
    - observer: true
    - data_alias: "victim"     # guest attacker's GPA collides with victim's HPA (not main's)
```

`data_alias` is an actor **name**; default `"main"` = actor 0 (original behavior). Only meaningful
for guest actors under `x86_enable_hpa_gpa_collisions`.

### How it threads through the stack

| Layer | File | Change |
|---|---|---|
| Config schema | `rvzr/config.py` | `data_alias` added to `ActorConf` / `ActorConfKey`. |
| Config defaults/whitelist | `rvzr/arch/x86/config.py` | `data_alias` added to the `actor` option whitelist and to `_actor_default` (`"main"`). |
| Actor object | `rvzr/tc_components/actor.py` | `Actor.data_alias` field; `from_dict` reads it. |
| Binary (RCBF) | `rvzr/tc_components/test_case_binary.py` | Resolves `data_alias` name → target actor id and writes it into the **previously-unused trailing metadata word** (no binary-layout change). |
| Executor struct | `rvzr/executor_km/include/actor.h` | Renamed the unused `code_permissions` field to `alias_target` (same offset). |
| Executor aliasing | `rvzr/executor_km/x86/page_tables_guest.c` | `set_guest_page_tables()` aliases guest data/code to `sandbox->data[alias_target]` / `code[alias_target]` (bounds-checked, default 0) instead of `[0]`. |

### Mechanism note

The aliasing sets the guest **page-table paddr** (the GPA) to the alias target's HPA. The EPT still
maps that GPA to the guest's own backing HPA; the collision is on the *numerical GPA*, which is what
the microarchitectural leak (Foreshadow/VMSCAPE-style) keys on. This matches how V2V worked (both
guests collide onto `main`), just retargeted.

### Status
Implemented + parses; the executor module was rebuilt. GS→HU did **not** reproduce: the blocker is
cross-domain BTB source-PC alignment (host vs guest virtual addresses differ), confirmed as a general
wall in `campaigns/vmscape-repro/FINDINGS-bti-alignment.md`. Cross-actor BTB injection only fires when
the attacker's and victim's branches share a virtual address (why V2V works and H2V/GS→HU/K2U don't).

---

## Extension 4 — Host↔host data aliasing (`data_alias` for host actors)

The same `data_alias` config also works for **host** actors: it points one host actor's **main
(channel) page** at another host actor's physical frame, giving them a shared Flush+Reload channel
without any guest (used for the K2U experiment: host-user observer aliased to a host-kernel victim).

### Lifecycle — the important part

Unlike the guest alias (a static page-table setup), the host alias **must be applied per input**,
because `load_sandbox_data()` writes every actor's input **through its sandbox VA** each input. If
the alias were live during the load, the observer's input would be written through the alias into the
victim's frame and clobber it (both would then see the observer's data → nothing to leak). So:

```
for each input:
    restore_host_data_alias();   // revert → each actor's input lands in its OWN frame
    load_sandbox_data(i);
    set_host_data_alias();       // re-apply → observer's main page points at victim's frame
    ... execute ...
```

Only `MAIN_PAGE_ID` is aliased (not reg-init/faulty): the observer keeps its own register-init values
(no spurious model/hardware register mismatch), and the faulty-page permission machinery is untouched.
A victim's transient disclosure must therefore imprint within the main page (mask index `< 0x1000`).

| Layer | File | Change |
|---|---|---|
| Alias apply/revert | `rvzr/executor_km/page_tables_host.c` | `set_host_data_alias` (save own PTE + borrow target PFN, main page only), `restore_host_data_alias`, `reset_host_data_alias`. |
| Per-input hook | `rvzr/executor_km/measurement.c` | `restore_host_data_alias()` before `load_sandbox_data`, `set_host_data_alias()` after. |
| Per-test-case reset | `rvzr/executor_km/sandbox_manager.c` | `set_sandbox_page_tables()` calls `reset_host_data_alias()`; revert happens via existing `recover_orig_state`. |

### Verify
`bash campaigns/vmscape-repro/verify_alias.sh` → expect `>0/8 diverge → ALIAS WORKS`. The model is
alias-unaware (predicts the observer reads its own data), so hardware≠model iff the alias is live.

> Debugging trap this fixed: the first version applied the alias once, before the loop — it read as
> "no-op" (0/8 divergence) because `load_sandbox_data` overwrote the shared frame with the observer's
> own input every input. Always re-check per-input state, not just setup-time state.

---

## Build & load

The three extensions above change the kernel module, so it must be rebuilt and reloaded:

```bash
make -C rvzr/executor_km
sudo rmmod rvzr_executor 2>/dev/null
sudo insmod rvzr/executor_km/rvzr_executor.ko
cat /sys/module/rvzr_executor/srcversion    # note: changes every rebuild
```

The campaign scripts derive the wanted srcversion from the built `.ko` (`modinfo -F srcversion`) and
hot-swap the loaded module if it differs (needs sudo).

Python side needs no build; unit tests: `python3 -m unittest tests.x86_tests.unit_model
tests.unit_tc_components`.

---

## Campaign artifacts (`campaigns/vmscape-repro/`)

| Boundary | config | template | driver |
|---|---|---|---|
| V2V (guest↔guest), residue | `config.yaml` | `template.asm` (+ diff variants) | `reproduce_baseline.sh`, `reproduce_violations.sh` |
| H2V (host-victim → guest-observer), residue | `config-h2v.yaml` | `template-h2v.asm`, `template-h2v-no-jmp.asm` | `run_h2v.sh [baseline|no-jmp]` |
| vBTI_GS→HU (guest attacker → host-user victim), injection | `config-gs2hu.yaml` | `template-gs2hu.asm` | `run_gs2hu.sh` |

Notes on direction/modes:
- **residue** = victim trains BTB, observer observes the residue (what V2V/H2V templates do).
- **injection** = attacker trains, victim mispredicts & leaks (vBTI; the GS→HU template).
- The literature's vulnerable vBTI directions (HU↔GU, HS→GU, GS→HU) all have a **user-mode victim**;
  a kernel victim (e.g. H2V's guest-kernel observer) was not expected to — and did not — reproduce.

---

## Gotcha: split-round observers must reset registers (arch-mismatch filter)

The arch-mismatch filter compares the observer's final GPRs between model and hardware. The model
carries registers across actor transitions (it models them as plain jumps); hardware **clobbers**
guest GPRs across a VM-exit/host/VM-enter (AMD VMRUN only auto-saves RAX). So if an observer's
register-mutating work (its gadget) is in a **different round** from its `measurement_end` — as in
the GS→HU injection template, where the attacker trains in one round and measures in another with
the victim round in between — the final registers diverge (input-dependent in the model, constant
in hardware) and **every test case is filtered as an architectural mismatch** (0 real violations,
all bugs).

Fix: zero (or re-initialize from memory, like the V2V observer) the GPRs at the start of the
measurement round, before `measurement_end`. This made GS→HU drop from 10/10 to 0/10 arch-mismatch.
It is *not* an aliasing or model bug — verified by toggling `data_alias` (on/off both 10/10 before
the fix). Rule of thumb: any observer whose measurement is split from its computation across a VM
boundary must re-establish a deterministic register state before measuring.

## Open questions / future work

- **Cross-domain BTB alignment** (H2V, GS→HU): host and guest code live at different virtual
  addresses; source-PC collision relies on BTB low-bit indexing and may need preamble tuning.
- **GS→HU channel validation**: confirm the aliased F+R channel actually carries the victim's
  footprint (a positive control) before trusting a negative result.
- **Kernel↔user transition inside a single guest**: unsupported (needs guest GDT/TSS/IDT + syscall
  MSRs); would enable in-VM Meltdown-style tests.
