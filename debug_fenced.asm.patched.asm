.intel_syntax noprefix
.section .data.main
.function_main_0:
.bb_main_0.entry:
lfence
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]
.exit_main_0:
lfence
.function_main_1:
.bb_main_1.entry:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]
.exit_main_1:
lfence
.function_main_2:
.bb_main_2.entry:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
lfence
xor rbx, rbx # noremove
lfence
xor rcx, rcx # noremove
lfence
xor rdx, rdx # noremove
lfence
xor rsi, rsi # noremove
lfence
xor rdi, rdi # noremove
lfence
.patch_placeholder:
lfence
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]
.exit_main_2:
lfence
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
lfence
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]
.exit_main_3:
lfence
.function_main_4:
.bb_main_4.entry:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop
lfence
.exit_main_4:
lfence
.section .data.vmvictim
.function_vmvictim_0:
.bb_vmvictim_0.entry:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
mov rax, qword ptr [r14 +0x2000]
lfence
mov byte ptr [r14 +0x140], al
lfence
movzx rbx, byte ptr [r14 +0x140]
lfence
mfence
lfence
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence
lfence
.exit_vmvictim_0:
lfence
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence
lfence
.exit_vm_0:
lfence
.function_vm_1:
.bb_vm_1.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
lfence
mov rax, qword ptr [r14 +0x2000] # noremove
lfence
mov rbx, qword ptr [r14 +0x2008] # noremove
lfence
mfence  # noremove
lfence
mov qword ptr [r14 +0x1000], rax
lfence
movzx rcx, byte ptr [r14 +0x140]
lfence
and rcx, 0x3f
lfence
shl rcx, 6
lfence
mov byte ptr [r14 +rcx], al
lfence
lfence  # noremove
lfence
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence
lfence
.exit_vm_1:
lfence
.function_vm_2:
.bb_vm_2.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence
lfence
.exit_vm_2:
lfence
.section .data.main
.test_case_exit:nop
