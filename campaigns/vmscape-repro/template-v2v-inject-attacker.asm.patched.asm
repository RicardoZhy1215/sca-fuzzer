.intel_syntax noprefix

# ============================================================================
# injection: attacker (vm) trains the btb, victim (vmvictim) is steered.
#
# this is the residue template with the roles swapped and a cross-actor readout:
#   - vm       = attacker : primes the l1d, trains a btb entry, later probes.
#   - vmvictim = victim   : loads its own secret, its indirect jmp inherits the
#                           attacker's btb entry and mispredicts, transiently
#                           touching a secret-dependent line => evicts an l1d set.
# the attacker reads that eviction via p+p (prime+probe) — no shared memory.
#
# orchestration (host main):
#   round0  vm.vm_0        -> measurement_start   (prime: attacker fills l1d sets)
#   round1  vm.vm_train    -> attacker's indirect jmp trains btb[jmp pc]
#   round2  vmvictim.1     -> victim loads secret, jmp mispredicts -> transient leak
#   round3  vm.vm_measure  -> measurement_end     (probe: attacker reads evicted set)
#
# alignment mirrors the working residue template exactly: the trainer (attacker)
# has no data loads, the mispredictor (victim) has the 6 loads — same asymmetry that
# lands both indirect jmps in the same zen4 btb bucket via the matched .16.4 preambles.
# the victim leaks at the mispredict target (whatever sits at that offset in its space),
# driven by its secret registers — same self-leak mechanism as residue, but now a
# different actor reads it.
#
# run with config-v2v-pp.yaml (executor_mode: p+p — cross-actor cache contention).
#
# verify any violation is genuine injection (not a generic transient):
#   1) trainer-strip: replace the attacker's `jmp rax` (vm_train) with nops -> must go clean.
#   2) counterexample-swap: vary only vmvictim's input -> htrace changes; vary only vm's -> no change
#      (=> it is the victim's secret leaking, i.e. real cross-domain injection).
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
.macro.switch_h2g.vmvictim.2: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.3: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- attacker (vm): prime, train, probe -----------
.section .data.vm
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence

.function_vm_train:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
# preamble adds code distance so this jmp lands in the same btb bucket as the victim's
.macro.random_instructions.16.4.vm_pre: nop qword ptr [rax + 0xff]
lea rax, qword ptr [rip + .function_vm_3]  # noremove   attacker's benign target
jmp rax  # noremove                        # trains btb[this pc] = vm_3's offset

.function_vm_3:
.macro.random_instructions.16.8.vm_post: nop qword ptr [rax + 0xff]
lfence  # noremove
.macro.switch_g2h.main.vm_1: nop qword ptr [rax + 0xff]
lfence

.function_vm_measure:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- victim (vmvictim): mispredict + leak ---------
.section .data.vmvictim
.function_vmvictim_1:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
xor rax, rax  # noremove
mov rax, qword ptr [r14 + 0x2000] # noremove   victim's own secret in rax..rdi
mov rbx, qword ptr [r14 + 0x2008] # noremove
mov rcx, qword ptr [r14 + 0x2010] # noremove
mov rdx, qword ptr [r14 + 0x2018] # noremove
mov rsi, qword ptr [r14 + 0x2020] # noremove
mov rdi, qword ptr [r14 + 0x2028] # noremove
mfence # noremove
# same-length preamble as the attacker -> same btb bucket
.macro.random_instructions.16.4.vmv_pre: nop qword ptr [rax + 0xff]
lea rbx, qword ptr [rip + .function_vmvictim_3]  # noremove   victim's own arch target
jmp rbx  # noremove              # arch -> vmvictim_3;  btb mispredict -> vm_3's offset (transient leak)

.function_vmvictim_3:
.macro.random_instructions.32.16.vmv_gadget: nop qword ptr [rax + 0xff]
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
