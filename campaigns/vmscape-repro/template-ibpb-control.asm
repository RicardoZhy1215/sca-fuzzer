.intel_syntax noprefix

# ============================================================================
# VMSCAPE-style cross-VM BTB / BTI residue test.
#
# Layout:
#   main      (host kernel)    : orchestrates VM-enter / VM-exit
#   vmvictim  (guest kernel)   : "trains" a BTB entry with an indirect call
#   vm        (guest kernel,   : observer; performs its own indirect call;
#              observer=true)    if BTB is not flushed across VM-exit / -enter,
#                                its indirect call may transiently jump to the
#                                victim-trained target, touching cache lines
#                                that leak through the F+R channel.
# ============================================================================

# ----------------------------- Hypervisor (Host) ----------------------------
.section .data.main
.function_main_0:
    # Warm-up observer round (collect baseline measurement state)
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vm.0:


.function_main_1:
    .macro.landing_g2h.main_1:

    # ---- Victim round: trains the BTB ----
    .macro.set_h2g_target.vmvictim.function_vmvictim_0:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.vmvictim.0:


.function_main_2:
    .macro.landing_g2h.main_2:

    # ============================================================
    #  CONTROL VARIANT: explicit IBPB between victim VM-exit and
    #  observer VM-entry. If the violation seen in the baseline
    #  template disappears with this in place, the leak is caused
    #  by cross-VM BTB residue (VMSCAPE-style).
    # ============================================================
    # WRMSR PRED_CMD (MSR 0x49), bit 0 = IBPB. Host is privileged
    # (privilege_level: kernel) so wrmsr is allowed.
    mov ecx, 0x49        # noremove
    mov eax, 1           # noremove
    xor edx, edx         # noremove
    wrmsr                # noremove
    lfence               # noremove

    .macro.set_h2g_target.vm.function_vm_1:
    .macro.set_g2h_target.main.function_main_3:

    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove

    .macro.switch_h2g.vm.1:


.function_main_3:
    .macro.landing_g2h.main_3:

.macro.fault_handler:
.patch_placeholder_fault_handler:

    # Tail observer round — closes out measurement cleanly
    .macro.set_h2g_target.vm.function_vm_2:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.vm.2:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop


# ----------------------------- VM - Victim ----------------------------------
.section .data.vmvictim
.function_vmvictim_0:
    .macro.landing_h2g.vmvictim_0:

    # Random pre-amble — adds code distance so the indirect jmp's PC
    # offset within this function matches the observer's (BTB-collision).
    # Placed BEFORE the lea so it cannot clobber the jump target.
    .macro.random_instructions.16.4.vv_pre:

    # Compute the victim's "secret" indirect-jump target.
    # Done immediately before `jmp` so random_instructions can't overwrite rax.
    lea rax, qword ptr [rip + .function_vmvictim_1]  # noremove
    jmp rax  # noremove                              # terminator: trains BTB at this PC offset

# Gadget lives in its own function so the indirect jmp above is the
# last instruction of its BB (required by Revizor's parser).
.function_vmvictim_1:
    # Secret-dependent memory traffic. r14 is the data-sandbox base in Revizor;
    # random_instructions sprinkles loads/stores that depend on input bytes,
    # leaving cache footprints the observer can detect.
    .macro.random_instructions.32.16.vv_gadget:

    lfence
    .macro.switch_g2h.main.vmvictim_0:
    lfence


# ----------------------------- VM - Observer --------------------------------
.section .data.vm

# Round 0: warm-up — start measurement, immediately return.
.function_vm_0:
    .macro.landing_h2g.vm_0:
    .macro.measurement_start.vm_0:
    .macro.switch_g2h.main.vm_0:
    lfence


# Round 1: the actual observation. We perform an indirect call at the
# SAME code offset within this actor as the victim used (both functions
# start with landing_h2g + random_instructions of equal length, so the
# `call` lands at a matching low-PC). If BTB residue exists, the call
# transiently retargets to vmvictim_gadget's address, touching its
# cache lines, which F+R then picks up.
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

    # Mirror the victim's pre-amble length so the indirect `jmp`
    # lands at an offset that collides in BTB with the victim's.
    # Placed BEFORE the lea so it cannot clobber the jump target.
    .macro.random_instructions.16.4.vm_pre:

    # Observer's "legitimate" indirect target. Set just before jmp.
    lea rbx, qword ptr [rip + .function_vm_3]  # noremove
    jmp rbx  # noremove                        # architecturally hits .function_vm_3,
                                               # transiently may hit function_vmvictim_1 via BTB residue.

# Legit target as its own function/BB (required by Revizor parser).
.function_vm_3:
    # Generic post-jmp workload; includes loads so the F+R sweep has
    # plenty of victim cache footprints to surface.
    .macro.random_instructions.32.16.vm_post:

    lfence  # noremove

    .macro.measurement_end.vm_1:
    .macro.switch_g2h.main.1:
    lfence


# Round 2: cleanup
.function_vm_2:
    .macro.landing_h2g.vm_2:
    .macro.measurement_end.vm_2:
    .macro.switch_g2h.main.2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
