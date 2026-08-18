.intel_syntax noprefix

# ============================================================================
# h2v control: same as template-h2v.asm but with both indirect jmps removed
# (host victim + guest observer fall through into their gadget bodies).
#
# purpose: isolate the channel.
#   - if this control still shows a violation  -> the leak is the shared-memory
#     f+r channel (aliasing), not branch-target injection.
#   - if this control is clean while template-h2v.asm violates -> an indirect
#     branch is necessary, i.e. genuine cross-domain btb/bhb residue.
# ============================================================================

# ----------------------------- host (victim + orchestrator) -----------------
.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]


.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]

# ---- victim round: no indirect jmp. fall through into the gadget so the
#      btb/bhb is never trained by an indirect branch. ----
.macro.random_instructions.16.4.hv_pre: nop qword ptr [rax + 0xff]

# gadget reached by fall-through, no indirect branch involved.
.function_main_gadget:
.macro.random_instructions.32.16.hv_gadget: nop qword ptr [rax + 0xff]
lfence

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
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence


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

# ---- control: no indirect jmp in the observer either; fall through. ----
.macro.random_instructions.16.4.vm_pre: nop qword ptr [rax + 0xff]

# reached by fall-through, no indirect branch.
.function_vm_3:
.macro.random_instructions.32.16.vm_post: nop qword ptr [rax + 0xff]

lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence


.function_vm_2:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence


# ----------------------------- exit -----------------------------------------
.section .data.main
.test_case_exit:nop
