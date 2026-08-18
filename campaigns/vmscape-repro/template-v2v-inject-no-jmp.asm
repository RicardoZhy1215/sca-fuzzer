.intel_syntax noprefix

# ============================================================================
# CONTROL for template-v2v-inject.asm — identical, but BOTH indirect jmps are
# removed. No BTB training, no misprediction => the disclosure is unreachable
# (architecturally or transiently). Expect CLEAN. A violation here would mean the
# "leak" in the main template is an artifact of the prefix/loads, not the injection.
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vm.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    .macro.set_h2g_target.vm.function_vm_train:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.vm.1:

.function_main_2:
    .macro.landing_g2h.main_2:
    .macro.set_h2g_target.vmvictim.function_vmvictim_1:
    .macro.set_g2h_target.main.function_main_3:
    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove
    .macro.switch_h2g.vmvictim.1:

.function_main_3:
    .macro.landing_g2h.main_3:
.macro.fault_handler:
.patch_placeholder_fault_handler:
    .macro.set_h2g_target.vm.function_vm_measure:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.vm.2:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop


# ----------------------------- Attacker (guest user, observer) --------------
.section .data.vm
.function_vm_train:
    .macro.landing_h2g.vm_1:
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    # CONTROL: no training jmp (fall through; BTB never trained)

.function_vm_safe:
    lfence  # noremove
    .macro.switch_g2h.main.vm_safe:
    lfence  # noremove

.function_vm_disclosure:
    lfence  # noremove
    .macro.switch_g2h.main.vm_1:
    lfence

.function_vm_0:
    .macro.landing_h2g.vm_0:
    .macro.measurement_start.vm_0:
    .macro.switch_g2h.main.vm_0:
    lfence

.function_vm_measure:
    .macro.landing_h2g.vm_2:
    xor rax, rax  # noremove
    xor rbx, rbx  # noremove
    xor rcx, rcx  # noremove
    xor rdx, rdx  # noremove
    xor rsi, rsi  # noremove
    xor rdi, rdi  # noremove
    .macro.measurement_end.vm_2:
    .macro.switch_g2h.main.2:
    lfence


# ----------------------------- Victim (guest user) --------------------------
.section .data.vmvictim
.function_vmvictim_1:
    .macro.landing_h2g.vmvictim_1:
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    # CONTROL: no victim jmp (fall through to safe; disclosure never reached)

.function_vmvictim_safe:
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_1:
    lfence

.function_vmvictim_disclosure:
    and rax, 0b111111000000  # noremove
    mov rcx, qword ptr [r14 + rax]   # noremove
    and rcx, 0b111111000000  # noremove
    mov rdx, qword ptr [r14 + rcx]   # noremove
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
