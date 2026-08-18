.intel_syntax noprefix

# ============================================================================
# v2v injection v2 — leak gadget at the mispredict target, not the fall-through.
#
# fixes the flaw of template-v2v-inject-fuzz.asm (where the disclosure sat right
# after the jmp = the sls fall-through, so it leaked without any injection). here,
# mirroring the residue template that actually works:
#
#   victim jmp  --arch/sls-->  vmvictim_safe   (right after jmp; lfence => benign, dead end)
#               --btb mispred->  vmvictim_disclosure  (a later offset; only reachable this way)
#
# the attacker trains btb[jmp pc] = disclosure's va (it jumps past its own safe to vm_disclosure).
# the victim's jmp inherits that target and mispredicts into vmvictim_disclosure, which reads the
# victim's secret; the observer reads the footprint via p+p (no shared memory). because the fall-
# through is a benign lfence, straight-line speculation can not reach the disclosure — so a leak here
# requires the attacker's training. verify with the bundled no-attacker control (must be clean).
#
# run with config-v2v-pp.yaml (p+p, no data_alias).
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
# vm_train and vmvictim_1 share an identical prefix so their jmps collide.
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
lea rax, qword ptr [rip + .function_vm_disclosure]  # noremove   (trained target = disclosure)
jmp rax  # noremove                                 # trains btb[this pc] = disclosure's va

.function_vm_safe:
lfence  # noremove   (jmp fall-through = benign dead end; blocks sls from the disclosure)
.macro.switch_g2h.main.vm_safe: nop qword ptr [rax + 0xff]
lfence

.function_vm_disclosure:
lfence  # noremove   (attacker's architectural target; inert)
.macro.switch_g2h.main.vm_1: nop qword ptr [rax + 0xff]
lfence

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
lea rbx, qword ptr [rip + .function_vmvictim_safe]  # noremove   (arch target = benign safe)
jmp rbx  # noremove                     # arch/sls -> safe;  btb mispredict -> disclosure

.function_vmvictim_safe:
lfence  # noremove   (jmp fall-through = benign dead end; blocks sls from the disclosure)
.macro.switch_g2h.main.vmvictim_1: nop qword ptr [rax + 0xff]
lfence

.function_vmvictim_disclosure:
.macro.random_instructions.32.16.vmv_gadget: nop qword ptr [rax + 0xff]
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
