.intel_syntax noprefix

# ============================================================================
# vbti_gs->hu transient redesign -- no-jmp control (both indirect jmps removed).
# expected clean: with no misprediction the victim never reaches disclosure.
#
# vbti_gs->hu, transient-only redesign.
#
# goal: the shared (aliased) channel must expose only the victim's transient
# (mispredicted) footprint, never its architectural one -- otherwise the leak is
# just "shared memory reveals the victim's data" (see template-gs2hu.asm, whose
# no-jmp control also violates). mirrors why v2v's no-jmp is clean: the victim's
# architectural data and the observer's readout are only bridged by misprediction.
#
# design:
#   attacker (guest kernel, observer, data_alias=victim):
#       - warm: measurement_start
#       - train: one indirect jmp that trains the btb target = attacker_disclosure
#                offset. no data-reading gadget (so the attacker never encodes the
#                victim's data itself).
#       - measure: f+r only (zeroed gprs; see the register-reset note).
#   victim (host user):
#       - loads its secret input into gprs
#       - an indirect jmp whose architectural target is `victim_safe` (which does
#         nothing secret-dependent -> input-independent architectural footprint)
#       - `victim_disclosure` (secret-dependent access) is reached only when the
#         attacker-trained btb entry makes the jmp mispredict there.
#
# controls: template-gs2hu-transient-no-jmp.asm removes both indirect jmps.
#   expected: no-jmp clean, baseline violates -> genuine vbti_gs->hu.
#
# empirical crux: for the victim to mispredict into victim_disclosure, that gadget
# must sit at the same low address bits as attacker_disclosure (btb low-bit index
# + context high bits). host and guest live at different virtual bases, so this is
# the knob that decides whether it reproduces at all -- tune the preamble lengths
# / disclosure offset if baseline stays clean.
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.attacker.function_attacker_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
# attacker trains the btb (no ibpb afterwards)
.macro.set_h2g_target.attacker.function_attacker_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
# host-user victim runs; its indirect jmp may mispredict to the trained target
.macro.set_k2u_target.victim.function_victim_1: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.switch_k2u.victim.1: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_u2k.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
# attacker measures
.macro.set_h2g_target.attacker.function_attacker_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- attacker (guest kernel, observer) ------------
.section .data.attacker
.function_attacker_0:
.macro.landing_h2g.attacker_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.attacker_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.attacker_0: nop qword ptr [rax + 0xff]
lfence

# train only: an indirect jmp that pins the btb target to attacker_disclosure.
# no memory-reading gadget here -> the attacker never encodes the victim's data.
.function_attacker_train:
.macro.landing_h2g.attacker_1: nop qword ptr [rax + 0xff]
# control: no training jmp; fall through (btb never trained)
.macro.random_instructions.16.4.att_pre: nop qword ptr [rax + 0xff]
.function_attacker_disclosure:
lfence  # noremove   (train target; deliberately inert)
.macro.switch_g2h.main.attacker_1: nop qword ptr [rax + 0xff]
lfence

# measure only: zero gprs (deterministic final state), then f+r. no gadget.
.function_attacker_measure:
.macro.landing_h2g.attacker_2: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.measurement_end.attacker_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- victim (host user) ---------------------------
.section .data.victim
.function_victim_1:
.macro.landing_k2u.victim_1: nop qword ptr [rax + 0xff]

xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove   (secret input in rax)
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove

# mirror the attacker's preamble so the jmp collides in the btb.
.macro.random_instructions.16.4.vic_pre: nop qword ptr [rax + 0xff]

# control: no jmp; fall through to victim_safe (never reaches disclosure)

# architectural target: no secret-dependent memory access -> input-independent
# footprint. (also the fall-through target for the no-jmp control.)
.function_victim_safe:
lfence  # noremove
.macro.switch_u2k.main.1: nop qword ptr [rax + 0xff]
lfence

# disclosure gadget: reached only transiently (btb residue). secret-dependent
# access encodes rax (secret) into a cache line the attacker's f+r can read.
.function_victim_disclosure:
and rax, 0b1111111111111  # noremove
mov rcx, qword ptr [r14 + rax]   # noremove   (transient: leak secret index)
and rcx, 0b1111111111111  # noremove
mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint into cache)
lfence  # noremove
.macro.switch_u2k.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
