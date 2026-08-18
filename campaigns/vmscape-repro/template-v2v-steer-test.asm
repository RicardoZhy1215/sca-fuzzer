.intel_syntax noprefix

# ============================================================================
# CROSS-GUEST STEERING TEST (residue direction), hand-written & self-contained.
#
# vmvictim TRAINS the BTB; vm MISPREDICTS and leaks ITS OWN secret (self-leak,
# like the residue violation) — so no cross-actor channel is involved, this tests
# ONLY whether one guest's indirect-branch training steers another guest's branch.
#
#   observer (vm) trace varies with vm's own secret -> vmvictim->vm steering WORKS
#       => my injection just has the roles backwards; flip them.
#   trace frozen -> cross-guest steering does NOT land even in the residue
#       direction with a clean template => the 585 violation leaked another way.
#
# Alignment: vmvictim_train and vm_1 are each their section's FIRST function with an
# identical prefix, so their jmps share a VA; each disclosure sits right after its
# jmp, so vm mispredicts into vm_disclosure (its own secret-reading gadget).
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vm.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    # trainer runs first
    .macro.set_h2g_target.vmvictim.function_vmvictim_train:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.vmvictim.1:

.function_main_2:
    .macro.landing_g2h.main_2:
    # then the observer mispredicts + self-leaks + measures
    .macro.set_h2g_target.vm.function_vm_1:
    .macro.set_g2h_target.main.function_main_3:
    .macro.switch_h2g.vm.2:

.function_main_3:
    .macro.landing_g2h.main_3:
.macro.fault_handler:
.patch_placeholder_fault_handler:
    nop


# ----------------------------- Observer/mispredictor (guest user) -----------
# function_vm_1 FIRST (offset 0) to mirror vmvictim_train.
.section .data.vm
.function_vm_1:
    .macro.landing_h2g.vm_1:
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove   (vm's OWN secret)
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    lea rbx, qword ptr [rip + .function_vm_safe]  # noremove   (arch target)
    jmp rbx  # noremove                     # ARCH -> safe;  mispredict -> vm_disclosure

.function_vm_disclosure:
    and rax, 0b111111000000  # noremove
    mov rcx, qword ptr [r14 + rax]   # noremove   (transient: vm's own secret-dependent load)
    and rcx, 0b111111000000  # noremove
    mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint)
    lfence  # noremove
    # (below is dead code: never reached architecturally)
    .macro.switch_g2h.main.vm_disc:
    lfence

.function_vm_safe:
    lfence  # noremove
    .macro.measurement_end.vm_1:
    .macro.switch_g2h.main.1:
    lfence

.function_vm_0:
    .macro.landing_h2g.vm_0:
    .macro.measurement_start.vm_0:
    .macro.switch_g2h.main.vm_0:
    lfence


# ----------------------------- Trainer (guest user) -------------------------
.section .data.vmvictim
.function_vmvictim_train:
    .macro.landing_h2g.vmvictim_1:
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    lea rbx, qword ptr [rip + .function_vmvictim_disclosure]  # noremove   (train target)
    jmp rbx  # noremove                     # trains BTB[this PC] = disclosure

.function_vmvictim_disclosure:
    lfence  # noremove   (inert; architectural jmp target for the trainer)
    .macro.switch_g2h.main.vmvictim_1:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
