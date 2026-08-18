.intel_syntax noprefix

# ============================================================================
# vBTI_GS->HU, transient redesign + FIXED-OFFSET ALIGNMENT.
#
# Same transient-only principle as template-gs2hu-transient.asm (victim's arch
# path is inert; only a mispredict reaches the disclosure), PLUS deterministic
# BTB alignment so the injection can actually land:
#
#   - attacker `train` is the FIRST function in its section (offset 0), mirroring
#     the victim's `function_victim_1` (also offset 0), so their indirect jmps sit
#     at the SAME in-section offset (BTB source-index collision, low 12 bits; both
#     code bases are page-aligned).
#   - the attacker's train gadget copies the victim's prefix byte-for-byte (same
#     8-byte landing placeholder, same 6 loads, NO random preamble) so the jmp
#     offset matches exactly.
#   - the attacker pads a `safe`-sized block (14 B, = victim_safe) between its jmp
#     and its `disclosure`, so attacker_disclosure and victim_disclosure land at
#     the SAME offset (BTB target-index alignment).
#
# CAVEAT: this only controls the sub-page (low-12) bits. If the CPU's BTB indexes
# on bits above 12, the host vs guest code bases (which differ there) still won't
# collide and it will stay clean -- that would be a genuine negative for this setup.
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.attacker.function_attacker_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.attacker.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    .macro.set_h2g_target.attacker.function_attacker_train:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.attacker.1:

.function_main_2:
    .macro.landing_g2h.main_2:
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
    .macro.set_h2g_target.attacker.function_attacker_measure:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.attacker.2:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop


# ----------------------------- Attacker (guest kernel, observer) ------------
# NOTE: function_attacker_train is placed FIRST (offset 0) to mirror victim_1.
.section .data.attacker
.function_attacker_train:
    .macro.landing_h2g.attacker_1:
    # mirror the victim's prefix EXACTLY (same 6 loads at fixed addrs -> fixed,
    # input-independent footprint; values are not used as addresses, so no leak).
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    lea rax, qword ptr [rip + .function_attacker_disclosure]  # noremove
    jmp rax  # noremove                                       # trains BTB[this PC] = disclosure

# 14-byte pad matching victim_safe (lfence + switch(8) + lfence); skipped by the jmp.
.function_attacker_safe:
    lfence  # noremove
    .macro.switch_g2h.main.attacker_safe:
    lfence  # noremove

.function_attacker_disclosure:
    lfence  # noremove   (architectural jmp target; inert)
    .macro.switch_g2h.main.attacker_1:
    lfence

.function_attacker_0:
    .macro.landing_h2g.attacker_0:
    .macro.measurement_start.attacker_0:
    .macro.switch_g2h.main.attacker_0:
    lfence

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
    lea rbx, qword ptr [rip + .function_victim_safe]  # noremove
    jmp rbx  # noremove                     # ARCH -> safe;  mispredict -> disclosure

.function_victim_safe:
    lfence  # noremove
    .macro.switch_u2k.main.1:
    lfence

.function_victim_disclosure:
    and rax, 0b1111111111111  # noremove
    mov rcx, qword ptr [r14 + rax]   # noremove   (transient: secret-dependent load)
    and rcx, 0b1111111111111  # noremove
    mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint into cache)
    lfence  # noremove
    .macro.switch_u2k.main.2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
