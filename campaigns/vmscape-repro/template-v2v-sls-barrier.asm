.intel_syntax noprefix

# ============================================================================
# V2V injection with an SLS BARRIER — the decisive injection-vs-SLS test.
#
# Same as template-v2v-inject-fuzz.asm, but a barrier BB (starting with lfence)
# sits between each jmp and its disclosure. The lfence is the jmp's fall-through,
# so straight-line speculation is blocked and can NOT reach the disclosure. The
# disclosure is a SEPARATE BB after the barrier, reachable ONLY via a BTB-predicted
# jump to its address. So:
#   - leak SURVIVES -> the victim really mispredicts into the disclosure via the
#     attacker-trained BTB target = genuine cross-guest INJECTION.
#   - leak DIES     -> it was the victim's own straight-line speculation (SLS) all
#     along, not injection.
# The barrier is identical on attacker and victim, so the disclosures stay aligned.
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
    .macro.random_instructions.16.4.vm_pre:
    lea rax, qword ptr [rip + .function_vm_disclosure]  # noremove
    jmp rax  # noremove                                 # trains BTB[this PC] = disclosure

.function_vm_sls:
    lfence  # noremove   (SLS barrier: the jmp's fall-through lands here and stops)
    .macro.switch_g2h.main.vm_sls:
    lfence

.function_vm_disclosure:
    lfence  # noremove   (architectural jmp target; inert in the attacker)
    .macro.switch_g2h.main.vm_1:
    lfence

.function_vm_safe:
    lfence  # noremove
    .macro.switch_g2h.main.vm_safe:
    lfence  # noremove

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
    mov rax, qword ptr [r14 + 0x2000] # noremove   (secret input in rax)
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    .macro.random_instructions.16.4.vmv_pre:
    lea rbx, qword ptr [rip + .function_vmvictim_safe]  # noremove   (arch target)
    jmp rbx  # noremove                     # ARCH -> safe; mispredict -> disclosure

.function_vmvictim_sls:
    lfence  # noremove   (SLS barrier: the jmp's fall-through lands here and stops)
    .macro.switch_g2h.main.vmvictim_sls:
    lfence

.function_vmvictim_disclosure:
    .macro.random_instructions.32.16.vmv_gadget:
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_2:
    lfence

.function_vmvictim_safe:
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_1:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
