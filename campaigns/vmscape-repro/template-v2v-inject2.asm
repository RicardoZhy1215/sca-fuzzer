.intel_syntax noprefix

# ============================================================================
# V2V INJECTION v2 — leak gadget at the MISPREDICT TARGET, not the fall-through.
#
# Fixes the flaw of template-v2v-inject-fuzz.asm (where the disclosure sat right
# after the jmp = the SLS fall-through, so it leaked without any injection). Here,
# mirroring the residue template that actually works:
#
#   victim jmp  --arch/SLS-->  vmvictim_safe   (right after jmp; lfence => benign, dead end)
#               --BTB mispred->  vmvictim_disclosure  (a LATER offset; ONLY reachable this way)
#
# The attacker trains BTB[jmp PC] = disclosure's VA (it jumps PAST its own safe to vm_disclosure).
# The victim's jmp inherits that target and mispredicts into vmvictim_disclosure, which reads the
# victim's secret; the observer reads the footprint via P+P (no shared memory). Because the fall-
# through is a benign lfence, straight-line speculation can NOT reach the disclosure — so a leak here
# REQUIRES the attacker's training. Verify with the bundled no-attacker control (must be clean).
#
# Run with config-v2v-pp.yaml (P+P, no data_alias).
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
# vm_train and vmvictim_1 share an identical prefix so their jmps collide.
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
    lea rax, qword ptr [rip + .function_vm_disclosure]  # noremove   (trained target = disclosure)
    jmp rax  # noremove                                 # trains BTB[this PC] = disclosure's VA

.function_vm_safe:
    lfence  # noremove   (jmp fall-through = benign dead end; blocks SLS from the disclosure)
    .macro.switch_g2h.main.vm_safe:
    lfence

.function_vm_disclosure:
    lfence  # noremove   (attacker's architectural target; inert)
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
    mov rax, qword ptr [r14 + 0x2000] # noremove   (secret input in rax)
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    .macro.random_instructions.16.4.vmv_pre:
    lea rbx, qword ptr [rip + .function_vmvictim_safe]  # noremove   (arch target = benign safe)
    jmp rbx  # noremove                     # ARCH/SLS -> safe;  BTB mispredict -> disclosure

.function_vmvictim_safe:
    lfence  # noremove   (jmp fall-through = benign dead end; blocks SLS from the disclosure)
    .macro.switch_g2h.main.vmvictim_1:
    lfence

.function_vmvictim_disclosure:
    .macro.random_instructions.32.16.vmv_gadget:
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
