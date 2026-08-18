.intel_syntax noprefix

# ============================================================================
# H2V — host-kernel victim  ->  guest-kernel observer   (cross-domain BTB / leak)
#
#   main (host kernel)            : orchestrator AND victim. Trains a BTB entry
#                                   with `lea r,[rip+L]; jmp r` in HOST context
#                                   (no VM enter).
#   vm   (guest kernel, observer) : enters a VM, does its own `lea r,[rip+L]; jmp r`
#                                   and measures via F+R.
#
# DIFFERENCE vs the V2V template: host and guest code live at DIFFERENT virtual
# addresses (host sandbox vs GUEST_V=0), so the two indirect jmps do NOT share a
# virtual PC -- BTB collision is NOT automatic here. Two channels are possible;
# use template-h2v-no-jmp.asm as the control to tell them apart:
#   (1) BTB residue    : observer mispredicts to the host-trained target.
#   (2) shared-mem F+R : with x86_enable_hpa_gpa_collisions the observer's data
#                        aliases the host victim's data, so the victim's cache
#                        footprint is visible even WITHOUT any misprediction.
# ============================================================================

# ----------------------------- Host (victim + orchestrator) -----------------
.section .data.main
.function_main_0:
    # Warm-up observer round: start measurement, return immediately.
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vm.0:


.function_main_1:
    .macro.landing_g2h.main_1:

    # ---- Victim round: HOST kernel trains the BTB (NO VM enter) ----
    # Pre-amble adds code distance; placed BEFORE the lea so it can't clobber rax.
    .macro.random_instructions.16.4.hv_pre:

    lea rax, qword ptr [rip + .function_main_gadget]  # noremove
    jmp rax  # noremove                               # trains BTB at this host PC

# Gadget in its own function so the indirect jmp is the BB terminator (parser req).
.function_main_gadget:
    # Secret-dependent traffic in HOST context (r14 = host main-actor data).
    .macro.random_instructions.32.16.hv_gadget:
    lfence

    # ---- No IBPB / no flush -- hand over to the observer VM ----
    .macro.set_h2g_target.vm.function_vm_1:
    .macro.set_g2h_target.main.function_main_2:

    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove

    .macro.switch_h2g.vm.1:


.function_main_2:
    .macro.landing_g2h.main_2:

.macro.fault_handler:
.patch_placeholder_fault_handler:

    .macro.set_h2g_target.vm.function_vm_2:
    .macro.set_g2h_target.main.function_main_3:
    .macro.switch_h2g.vm.2:

.function_main_3:
    .macro.landing_g2h.main_3:
    nop


# ----------------------------- VM - Observer --------------------------------
.section .data.vm

# Round 0: warm-up — start measurement, return.
.function_vm_0:
    .macro.landing_h2g.vm_0:
    .macro.measurement_start.vm_0:
    .macro.switch_g2h.main.vm_0:
    lfence


# Round 1: observation. Indirect jmp at a matching in-function offset; if BTB
# residue from the host victim survives the host->guest transition, this may
# transiently retarget to the victim gadget.
.function_vm_1:
    .macro.landing_h2g.vm_1:

    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove

    .macro.random_instructions.16.4.vm_pre:

    lea rbx, qword ptr [rip + .function_vm_3]  # noremove
    jmp rbx  # noremove

.function_vm_3:
    .macro.random_instructions.32.16.vm_post:

    lfence  # noremove
    .macro.measurement_end.vm_1:
    .macro.switch_g2h.main.1:
    lfence


# Round 2: cleanup.
.function_vm_2:
    .macro.landing_h2g.vm_2:
    .macro.measurement_end.vm_2:
    .macro.switch_g2h.main.2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
