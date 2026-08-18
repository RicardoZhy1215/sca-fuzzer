.intel_syntax noprefix

# ============================================================================
# cross-guest steering test (residue direction), hand-written & self-contained.
#
# vmvictim trains the btb; vm mispredicts and leaks its own secret (self-leak,
# like the residue violation) — so no cross-actor channel is involved, this tests
# only whether one guest's indirect-branch training steers another guest's branch.
#
#   observer (vm) trace varies with vm's own secret -> vmvictim->vm steering works
#       => my injection just has the roles backwards; flip them.
#   trace frozen -> cross-guest steering does not land even in the residue
#       direction with a clean template => the 585 violation leaked another way.
#
# alignment: vmvictim_train and vm_1 are each their section's first function with an
# identical prefix, so their jmps share a va; each disclosure sits right after its
# jmp, so vm mispredicts into vm_disclosure (its own secret-reading gadget).
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
# trainer runs first
.macro.set_h2g_target.vmvictim.function_vmvictim_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
# then the observer mispredicts + self-leaks + measures
.macro.set_h2g_target.vm.function_vm_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
nop


# ----------------------------- observer/mispredictor (guest user) -----------
# function_vm_1 first (offset 0) to mirror vmvictim_train.
.section .data.vm
.function_vm_1:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove   (vm's own secret)
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
lea rbx, qword ptr [rip + .function_vm_safe]  # noremove   (arch target)
jmp rbx  # noremove                     # arch -> safe;  mispredict -> vm_disclosure

.function_vm_disclosure:
and rax, 0b111111000000  # noremove
mov rcx, qword ptr [r14 + rax]   # noremove   (transient: vm's own secret-dependent load)
and rcx, 0b111111000000  # noremove
mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint)
lfence  # noremove
# (below is dead code: never reached architecturally)
.macro.switch_g2h.main.vm_disc: nop qword ptr [rax + 0xff]
lfence

.function_vm_safe:
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence

.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- trainer (guest user) -------------------------
.section .data.vmvictim
.function_vmvictim_train:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
lea rbx, qword ptr [rip + .function_vmvictim_disclosure]  # noremove   (train target)
jmp rbx  # noremove                     # trains btb[this pc] = disclosure

.function_vmvictim_disclosure:
lfence  # noremove   (inert; architectural jmp target for the trainer)
.macro.switch_g2h.main.vmvictim_1: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
