.intel_syntax noprefix
.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]
.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]
.function_main_2:
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
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]
.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]
.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
.section .data.vmvictim
.function_vmvictim_0:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
add rbx, 0b1111111111111 #
lfence
mov rcx, 0b1111111111111 #
lfence
add ebx, 0b1000000000000000000000000000000 #
lfence
add rbx, 0b1111111111000 #
lfence
mov rcx, 0b1111111111111 #
lfence
add rbx, 0b1111111111000 #
lfence
mov rdx, 0b1111111111000 #
lfence
mov rcx, 0b1111111111111 #
lfence
mov rcx, 0b1111111111000 #
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
or word ptr [r14 + rbx], 1 # instrumentation
lfence
# mem access: [34] 0x5668-0x5668 cl 25:40 | [84] 0x5668-0x5668 cl 25:40
and dx, word ptr [r14 + rbx] # instrumentation
lfence
# mem access: [34] 0x5668 cl 25:40 | [84] 0x5668 cl 25:40
shr dx, 1 #
lfence
mov al, -23 #
lfence
add cl, -46 # instrumentation
lfence
mov rax, 0b1111111111111 #
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov dword ptr [r14 + rdi], 0b1000000000000000000000000000000 #
lfence
# mem access: [34] 0x65f5 cl 23:53 | [84] 0x65f5 cl 23:53
mov rcx, 0b1111111111111 #
lfence
mov rdx, 0b1111111111111 #
lfence
add cl, 1 #
lfence
mov al, 117 #
lfence
add rcx, 0b1111111111000 #
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
add dword ptr [r14 + rsi], 0b1000 #
lfence
# mem access: [34] 0x697c-0x697c cl 37:60 | [84] 0x61eb-0x61eb cl 7:43
mov al, -121 #
lfence
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence
lfence
.section .data.vm
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence
lfence
.function_vm_1:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
lfence
mov rax, qword ptr [r14 +0x2000] # noremove
lfence
# mem access: [34] 0xb000 cl 0:0 | [84] 0xb000 cl 0:0
mov rbx, qword ptr [r14 +0x2008] # noremove
lfence
# mem access: [34] 0xb008 cl 0:8 | [84] 0xb008 cl 0:8
mov rcx, qword ptr [r14 +0x2010] # noremove
lfence
# mem access: [34] 0xb010 cl 0:16 | [84] 0xb010 cl 0:16
mov rdx, qword ptr [r14 +0x2018] # noremove
lfence
# mem access: [34] 0xb018 cl 0:24 | [84] 0xb018 cl 0:24
mov rsi, qword ptr [r14 +0x2020] # noremove
lfence
# mem access: [34] 0xb020 cl 0:32 | [84] 0xb020 cl 0:32
mov rdi, qword ptr [r14 +0x2028] # noremove
lfence
# mem access: [34] 0xb028 cl 0:40 | [84] 0xb028 cl 0:40
mfence  # noremove
lfence
lfence  # speculation barrier (lfence-test)
lfence
mov rdx, 0b1111111111111 #
lfence
lea esi, qword ptr [rcx + rcx + 61885]
lfence
and rcx, 0b1111111111111 # instrumentation
lfence
mov ecx, dword ptr [r14 + rcx]
lfence
# mem access: [34] 0x9646 cl 25:6 | [84] 0x9646 cl 25:6
and rcx, 0b1111111111111 # instrumentation
lfence
bts dword ptr [r14 + rcx], 0
lfence
# mem access: [34] 0x909a-0x909a cl 2:26 | [84] 0x909a-0x909a cl 2:26
and rcx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rcx], rcx
lfence
# mem access: [34] 0x909a cl 2:26 | [84] 0x909a cl 2:26
mov rcx, 0b1111111111111 #
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov byte ptr [r14 + rdi], al
lfence
# mem access: [34] 0xab82-0x1ff8 cl 46:2 | [84] 0xab82-0x1ff8 cl 46:2
and rsi, 0b1111111111111 # instrumentation
lfence
mov ebx, dword ptr [r14 + rsi]
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
setns byte ptr [r14 + rbx]
lfence
mov rcx, 0b1111111111111 #
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mov dx, word ptr [r14 + rax]
lfence
mov rdx, 0b1111111111111 #
lfence
mov rcx, 0b1111111111111 #
lfence
mov rsi, 0b1111111111000 #
lfence
mov rsi, 0b1111111111111 #
lfence
add rbx, 0b1111111111111 #
lfence
add rbx, 0b1111111111000 #
lfence
add rbx, 0b1111111111111 #
lfence
lfence  # noremove
lfence
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence
lfence
.function_vm_2:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence
lfence
.section .data.main
.test_case_exit:nop
