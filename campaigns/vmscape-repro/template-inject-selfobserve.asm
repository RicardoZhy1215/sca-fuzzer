.intel_syntax noprefix

# ============================================================================
# INJECTION STEERING, self-observe. Roles are the WORKING residue template swapped:
#   vm       = ATTACKER : trains a BTB entry (no data loads, just preamble + indirect jmp).
#   vmvictim = VICTIM   : the OBSERVER. Loads its OWN secret, its indirect jmp inherits the
#                         attacker's BTB entry and MISPREDICTS, transiently touching a
#                         secret-dependent line; the victim reads its own footprint via F+R.
#
# This keeps the readout INSIDE the mispredictor (same round, no cross-actor switch in the
# measurement window) so it dodges the P+P wall. It proves the attacker STEERS the victim's
# branch into a disclosure — the real injection primitive — WITHOUT any EPT alias. It does NOT
# claim the attacker reads the secret (that needs L2/LLC P+P; see the notes in the chat).
#
# Structure/alignment mirror residue exactly: trainer (vm) has NO loads, mispredictor (vmvictim)
# has the 6 loads + matched .16.4 preambles -> both indirect jmps land in the same Zen4 BTB bucket.
#
# Run with config-inject-selfobserve.yaml (F+R + hpa_gpa; observer = vmvictim).
# Verify: trainer-strip the ATTACKER 'jmp rax' (vm_train) -> must go CLEAN (=> steering is causal);
#         counterexample -> only vmvictim's own input drives the htrace.
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.vmvictim.function_vmvictim_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vmvictim.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    .macro.set_h2g_target.vm.function_vm_train:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.vm.0:

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
    .macro.set_h2g_target.vmvictim.function_vmvictim_2:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.vmvictim.2:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop


# ----------------------------- Attacker (vm): trainer only ------------------
.section .data.vm
.function_vm_train:
    .macro.landing_h2g.vm_0:
    .macro.random_instructions.16.4.vm_pre:
    lea rax, qword ptr [rip + .function_vm_gadget]  # noremove
    jmp rax  # noremove

.function_vm_gadget:
    .macro.random_instructions.32.16.vm_gadget:
    lfence  # noremove
    .macro.switch_g2h.main.vm_0:
    lfence


# ----------------------------- Victim (vmvictim): mispredictor + observer ----
.section .data.vmvictim
.function_vmvictim_0:
    .macro.landing_h2g.vmvictim_0:
    .macro.measurement_start.vmvictim_0:
    .macro.switch_g2h.main.vmvictim_0:
    lfence

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
    .macro.random_instructions.16.4.vmv_pre:
    lea rbx, qword ptr [rip + .function_vmvictim_3]  # noremove
    jmp rbx  # noremove

.function_vmvictim_3:
    .macro.random_instructions.32.16.vmv_post:
    lfence  # noremove
    .macro.measurement_end.vmvictim_1:
    .macro.switch_g2h.main.1:
    lfence

.function_vmvictim_2:
    .macro.landing_h2g.vmvictim_2:
    .macro.measurement_end.vmvictim_2:
    .macro.switch_g2h.main.2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
