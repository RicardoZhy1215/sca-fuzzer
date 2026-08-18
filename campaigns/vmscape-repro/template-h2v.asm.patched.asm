.intel_syntax noprefix

# ============================================================================
# h2v — host-kernel victim  ->  guest-kernel observer   (cross-domain btb / leak)
#
#   main (host kernel)            : orchestrator and victim. trains a btb entry
#                                   with `lea r,[rip+l]; jmp r` in host context
#                                   (no vm enter).
#   vm   (guest kernel, observer) : enters a vm, does its own `lea r,[rip+l]; jmp r`
#                                   and measures via f+r.
#
# difference vs the v2v template: host and guest code live at different virtual
# addresses (host sandbox vs guest_v=0), so the two indirect jmps do not share a
# virtual pc -- btb collision is not automatic here. two channels are possible;
# use template-h2v-no-jmp.asm as the control to tell them apart:
#   (1) btb residue    : observer mispredicts to the host-trained target.
#   (2) shared-mem f+r : with x86_enable_hpa_gpa_collisions the observer's data
#                        aliases the host victim's data, so the victim's cache
#                        footprint is visible even without any misprediction.
# ============================================================================

# ----------------------------- host (victim + orchestrator) -----------------
.section .data.main
.function_main_0:
# warm-up observer round: start measurement, return immediately.
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]


.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]

# ---- victim round: host kernel trains the btb (no vm enter) ----
# pre-amble adds code distance; placed before the lea so it can't clobber rax.
.macro.random_instructions.16.4.hv_pre: nop qword ptr [rax + 0xff]

lea rax, qword ptr [rip + .function_main_gadget]  # noremove
jmp rax  # noremove                               # trains btb at this host pc

# gadget in its own function so the indirect jmp is the bb terminator (parser req).
.function_main_gadget:
# secret-dependent traffic in host context (r14 = host main-actor data).
.macro.random_instructions.32.16.hv_gadget: nop qword ptr [rax + 0xff]
lfence

# ---- no ibpb / no flush -- hand over to the observer vm ----
.macro.set_h2g_target.vm.function_vm_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]

xor rax, rax  # noremove
xor rbx, rbx  # noremove
xor rcx, rcx  # noremove
xor rdx, rdx  # noremove
xor rsi, rsi  # noremove
xor rdi, rdi  # noremove

.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]


.function_main_2:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]

.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:

.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]

.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
nop


# ----------------------------- vm - observer --------------------------------
.section .data.vm

# round 0: warm-up — start measurement, return.
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence


# round 1: observation. indirect jmp at a matching in-function offset; if btb
# residue from the host victim survives the host->guest transition, this may
# transiently retarget to the victim gadget.
.function_vm_1:
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

lea rbx, qword ptr [rip + .function_vm_3]  # noremove
jmp rbx  # noremove

.function_vm_3:
.macro.random_instructions.32.16.vm_post: nop qword ptr [rax + 0xff]

lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence


# round 2: cleanup.
.function_vm_2:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
