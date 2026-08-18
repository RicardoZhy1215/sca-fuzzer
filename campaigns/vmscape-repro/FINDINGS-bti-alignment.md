# vBTI reproduction — findings on the BTB-alignment wall

_Lab notebook, K2U / GS→HU / H2V branch-target-injection attempts. Zen4 box (see the
`dis_ucode_ldr` note that reverts 0208→0206 to reopen the cross-VM BTB channel)._

## TL;DR

Cross-actor BTB injection in Revizor only fires when the attacker's indirect branch and the
victim's indirect branch sit at the **same virtual address**. Revizor lays every actor's code
`sizeof(actor_code_t) = 0x3000` apart, so two different actors' branches never share a VA (or even
a nice power-of-two alignment). That single fact explains every result so far:

| Config | Attacker → Victim | Branches share a VA? | Result |
|---|---|---|---|
| **V2V** | guest-user → guest-user | **yes** (both guests based identically) | **reproduces**\* |
| H2V | host-kernel → guest-kernel | no (host base ≠ guest base) | clean |
| GS→HU | guest-kernel → host-user | no (guest base ≠ host base) | clean |
| **K2U** | host-user → host-kernel | **impossible** (one address space) | clean |

\* What the V2V **residue** template reproduces, precisely: **cross-guest predictor steering** — one
guest's indirect-branch training makes another guest's branch mispredict, and that guest **self-leaks
its own data** along the mispredicted path (not a separate victim's). The full cross-domain
**exfiltration** was reproduced separately with the EPT physical alias + a fuzzed injection template —
see `FINDINGS-injection.md` (`out-v2v-fuzz/violation-260811-214352`).

K2U is the worst case: host-user and host-kernel live in the **same address space**, so their two
branches physically cannot occupy the same VA. Same-VA collision is only available *across* address
spaces — which is exactly why every documented vBTI variant (HU↔GU, HS→GU, GS→HU) crosses the VM
boundary. **K2U is not a documented channel and is infeasible here without Zen4-specific BTB
index-aliasing.**

## The K2U diagnosis, step by step

Symptom: the K2U BTI campaign is clean even after the shared channel was fixed. We eliminated every
alternative until only alignment remained.

1. **Is the shared F+R channel even real?** It wasn't, at first — `set_host_data_alias` was a no-op
   in effect (see the fix below). After the fix, `verify_alias.sh` shows **7/8** inputs where the
   observer reads the *victim's* data (model, which is alias-unaware, disagrees). Channel confirmed.
2. **Does the model mask the leak?** No. `contract_execution_clause` defaults to
   `["delayed-exception-handling"]` — the model speculates only fault suppression, **not** branches.
   So the victim's mispredicted disclosure is *not* in the contract; a real hardware leak *would* be
   reported as a violation.
3. **Does the transition flush the BTB?** No. `switch_k2u` is `sysretq`, `switch_u2k` is `syscall`;
   neither flushes branch prediction (that is the basis of cross-syscall Spectre-v2), and there is no
   IBPB / `PRED_CMD` write anywhere in the transition macros. Training survives.
4. **So the hardware just isn't leaking → no BTB collision.** The attacker's `jmp` (user = actor 2)
   and victim's `jmp` (kvictim = actor 1) are placed at the same in-section offset, so they differ
   by exactly `sizeof(actor_code_t) = 0x2000 (section) + 0x1000 (macros) = 0x3000`.
   `0x3000` sets **bits 12 and 13** — identical low-12 (page offset), different bits 12–13. Zen4's
   BTB distinguishes those bits ⇒ different entries ⇒ no injection.

Note the misprediction/leak *mechanism* is fine: V2V (same fast-`lea`, single-training-pass style)
does produce a violation, so the transient window and training strength are adequate. What
K2U/H2V/GS→HU lack for their branches to even collide is the shared VA. (Caveat, added after reading
the V2V violation: what V2V demonstrates is cross-guest *predictor steering* + self-leak, not
cross-domain data exfiltration — see `FINDINGS-injection.md`. The alignment wall analysis here is
unaffected: no shared VA ⇒ no collision ⇒ nothing to steer, in any of the failing configs.)

## What we should have done differently

Nothing about the K2U template can overcome a same-address-space constraint. The productive unlock —
if/when this line resumes — is **VA co-location**: map the guest attacker's code page at the *same
virtual address* as the host victim's branch (a code analog of the `data_alias` we added for data).
That is the change that would let GS→HU / HU→GU branches collide the way V2V's already do.

## Byproduct: the host↔host data-alias fix (kept, verified)

While chasing K2U we found and fixed a real bug in the host↔host data aliasing that the earlier
guest work relied on conceptually. It is correct now and worth keeping regardless of the K2U outcome.

**Bug:** `set_host_data_alias` was applied **once**, before the per-input loop. But
`load_sandbox_data` writes each actor's input **through its sandbox VA** every input — so with the
alias live, the observer's input was written *through the alias into the victim's frame*, clobbering
the victim's data. Both actors then held the observer's data ⇒ nothing to leak (and the direct
model-vs-hardware probe showed 0/8 divergence).

**Fix** (`page_tables_host.c`, `measurement.c`, `sandbox_manager.c`):
- Apply the alias **per input, after `load_sandbox_data`**, and revert it **before the next load**
  (`restore_host_data_alias` → `load_sandbox_data` → `set_host_data_alias`), so each actor's input
  first lands in its own frame and only execution sees the alias.
- Alias **only the main (channel) page** (`MAIN_PAGE_ID`), not reg-init/faulty — the observer keeps
  its own register-init values (no spurious model/hardware register mismatch) and the faulty-page
  permission machinery is untouched. Victim disclosure must therefore imprint within the main page
  (mask index `< 0x1000`, e.g. `and rax, 0b111111000000`).
- `set_sandbox_page_tables` now only calls `reset_host_data_alias()`; the alias is reverted at
  test-case end by the existing `recover_orig_state → restore_orig_sandbox_page_tables`.

**Verify:** `bash campaigns/vmscape-repro/verify_alias.sh` → expect `>0/8 diverge → ALIAS WORKS`.
(Reads the poscontrol program, which reads victim data straight into registers; the alias-unaware
model diverges from hardware iff the alias is live.)

## Artifacts

- `template-k2u-bti.asm`, `template-k2u-bti-no-jmp.asm`, `template-k2u-poscontrol.asm`,
  `config-k2u-bti.yaml`, `run_k2u.sh`, `verify_alias.sh`.
- Executor build with the fix: `srcversion 80476EB4B47B9905E4C8FA0`.
