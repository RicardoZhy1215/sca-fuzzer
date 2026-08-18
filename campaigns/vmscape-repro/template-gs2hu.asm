.intel_syntax noprefix

# ============================================================================
# vBTI_GS->HU  (injection): guest-kernel attacker trains the BTB;
#                           host-user victim mispredicts and leaks.
#
#   main     (host kernel)          : orchestrator only
#   attacker (guest kernel, observer): trains a BTB entry, then measures via F+R.
#                                      data_alias=victim -> its guest memory GPA-collides
#                                      with the host-user victim's sandbox (config-gs2hu.yaml).
#   victim   (host user)            : runs its own indirect jmp; if the attacker-trained BTB
#                                      entry survives into the host-user context, the victim
#                                      mispredicts to a disclosure gadget and leaves a footprint
#                                      the attacker then reads.
#
# Mirror layout: attacker and victim use identical in-function offsets (same preamble length,
# same jmp offset, same gadget offset) to maximize BTB source-PC collision on the low address
# bits. Whether the entry actually crosses the host<->guest boundary is the empirical question.
#
# Order (injection): attacker trains  ->  victim mispredicts/leaks  ->  attacker measures.
# No IBPB between training and the victim.
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    # Warm-up: attacker starts measurement, returns.
    .macro.set_h2g_target.attacker.function_attacker_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.attacker.0:


.function_main_1:
    .macro.landing_g2h.main_1:

    # ---- Attacker trains the BTB (guest kernel) ----
    .macro.set_h2g_target.attacker.function_attacker_train:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.attacker.1:


.function_main_2:
    .macro.landing_g2h.main_2:

    # ---- No IBPB. Run the host-user victim; it may mispredict to the trained target. ----
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

    # ---- Attacker measures ----
    .macro.set_h2g_target.attacker.function_attacker_measure:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.attacker.2:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop


# ----------------------------- Attacker (guest kernel, observer) ------------
.section .data.attacker

# Round 0: warm-up — start measurement, return.
.function_attacker_0:
    .macro.landing_h2g.attacker_0:
    .macro.measurement_start.attacker_0:
    .macro.switch_g2h.main.attacker_0:
    lfence


# Round 1: train a BTB entry with an indirect jmp at a fixed in-function offset.
.function_attacker_train:
    .macro.landing_h2g.attacker_1:

    .macro.random_instructions.16.4.att_pre:

    lea rax, qword ptr [rip + .function_attacker_gadget]  # noremove
    jmp rax  # noremove                                   # trains BTB at this PC/offset

.function_attacker_gadget:
    .macro.random_instructions.32.16.att_gadget:
    lfence
    .macro.switch_g2h.main.attacker_1:
    lfence


# Round 2: measurement end + F+R reload (reads the shared, aliased channel).
# Zero the GPRs before measuring so the final architectural register state is deterministic and
# matches the model (which carries registers across transitions as plain jumps). Without this, the
# train-round register values survive in the model but are clobbered by the VM-exit/host/VM-enter
# in hardware -> every test case is wrongly filtered as an architectural mismatch. (F+R uses r14/r15,
# not these GPRs, so zeroing them does not affect the measurement. Mirrors the V2V observer, which
# re-initializes its registers each round.)
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
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove

    # Mirror the attacker's preamble length so the jmp lands at the same in-function offset.
    .macro.random_instructions.16.4.vic_pre:

    lea rbx, qword ptr [rip + .function_victim_safe]  # noremove
    jmp rbx  # noremove                               # architecturally safe target;
                                                      # may mispredict to the attacker gadget offset

.function_victim_safe:
    # Disclosure gadget: secret-dependent traffic in host-user context (r14 = victim data).
    .macro.random_instructions.32.16.vic_gadget:
    lfence  # noremove
    .macro.switch_u2k.main.1:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
