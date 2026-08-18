.intel_syntax noprefix

# ============================================================================
# H2V CONTROL: same as template-h2v.asm but with BOTH indirect jmps removed
# (host victim + guest observer fall through into their gadget bodies).
#
# Purpose: isolate the channel.
#   - If this control STILL shows a violation  -> the leak is the shared-memory
#     F+R channel (aliasing), NOT branch-target injection.
#   - If this control is CLEAN while template-h2v.asm violates -> an indirect
#     branch is necessary, i.e. genuine cross-domain BTB/BHB residue.
# ============================================================================

# ----------------------------- Host (victim + orchestrator) -----------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vm.0:


.function_main_1:
    .macro.landing_g2h.main_1:

    # ---- Victim round: NO indirect jmp. Fall through into the gadget so the
    #      BTB/BHB is never trained by an indirect branch. ----
    .macro.random_instructions.16.4.hv_pre:

# Gadget reached by fall-through, no indirect branch involved.
.function_main_gadget:
    .macro.random_instructions.32.16.hv_gadget:
    lfence

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
.function_vm_0:
    .macro.landing_h2g.vm_0:
    .macro.measurement_start.vm_0:
    .macro.switch_g2h.main.vm_0:
    lfence


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

    # ---- CONTROL: no indirect jmp in the observer either; fall through. ----
    .macro.random_instructions.16.4.vm_pre:

# Reached by fall-through, no indirect branch.
.function_vm_3:
    .macro.random_instructions.32.16.vm_post:

    lfence  # noremove
    .macro.measurement_end.vm_1:
    .macro.switch_g2h.main.1:
    lfence


.function_vm_2:
    .macro.landing_h2g.vm_2:
    .macro.measurement_end.vm_2:
    .macro.switch_g2h.main.2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
