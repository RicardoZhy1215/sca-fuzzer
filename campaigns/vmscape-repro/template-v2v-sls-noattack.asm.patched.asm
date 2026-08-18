.intel_syntax noprefix

# ============================================================================
# v2v injection with an sls barrier — the decisive injection-vs-sls test.
#
# same as template-v2v-inject-fuzz.asm, but a barrier bb (starting with lfence)
# sits between each jmp and its disclosure. the lfence is the jmp's fall-through,
# so straight-line speculation is blocked and can not reach the disclosure. the
# disclosure is a separate bb after the barrier, reachable only via a btb-predicted
# jump to its address. so:
#   - leak survives -> the victim really mispredicts into the disclosure via the
#     attacker-trained btb target = genuine cross-guest injection.
#   - leak dies     -> it was the victim's own straight-line speculation (sls) all
#     along, not injection.
# the barrier is identical on attacker and victim, so the disclosures stay aligned.
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.switch_h2g.vmvictim.1: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- attacker (guest user, observer) --------------
.section .data.vm
.function_vm_train:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
.macro.random_instructions.16.4.vm_pre: nop qword ptr [rax + 0xff]

.function_vm_sls:
lfence  # noremove   (sls barrier: the jmp's fall-through lands here and stops)
.macro.switch_g2h.main.vm_sls: nop qword ptr [rax + 0xff]
lfence

.function_vm_disclosure:
lfence  # noremove   (architectural jmp target; inert in the attacker)
.macro.switch_g2h.main.vm_1: nop qword ptr [rax + 0xff]
lfence

.function_vm_safe:
lfence  # noremove
.macro.switch_g2h.main.vm_safe: nop qword ptr [rax + 0xff]
lfence  # noremove

.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence

.function_vm_measure:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- victim (guest user) --------------------------
.section .data.vmvictim
.function_vmvictim_1:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove   (secret input in rax)
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
.macro.random_instructions.16.4.vmv_pre: nop qword ptr [rax + 0xff]
lea rbx, qword ptr [rip + .function_vmvictim_safe]  # noremove   (arch target)
jmp rbx  # noremove                     # arch -> safe; mispredict -> disclosure

.function_vmvictim_sls:
lfence  # noremove   (sls barrier: the jmp's fall-through lands here and stops)
.macro.switch_g2h.main.vmvictim_sls: nop qword ptr [rax + 0xff]
lfence

.function_vmvictim_disclosure:
.macro.random_instructions.32.16.vmv_gadget: nop qword ptr [rax + 0xff]
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence

.function_vmvictim_safe:
lfence  # noremove
.macro.switch_g2h.main.vmvictim_1: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
