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
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
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
add cl, -51 # instrumentation
sub dl, cl
mov cl, bl
and rdx, 0b1111111111000 # instrumentation
mov dword ptr [r14 + rdx], esi
# mem access: [30] 0x53c8 cl 15:8 | [80] 0x53c8 cl 15:8
lea bx, qword ptr [rsi]
mov dil, sil
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], 70
# mem access: [30] 0x5000 cl 0:0 | [80] 0x5000 cl 0:0
mov rsi, 0b1111111111111 #
and rbx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rbx], bl
# mem access: [30] 0x5128 cl 4:40 | [80] 0x5128 cl 4:40
or rbx, 0b1000000000000000000000000000000 # instrumentation
mov rax, rbx
and rbx, 0b1111111111111 # instrumentation
add dl, byte ptr [r14 + rbx]
# mem access: [30] 0x5128 cl 4:40 | [80] 0x5128 cl 4:40
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax
# mem access: [30] 0x5d28 cl 52:40 | [80] 0x5d28 cl 52:40
and rdx, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rdx]
# mem access: [30] 0x53f0 cl 15:48 | [80] 0x53f0 cl 15:48
mov bx, 139
mov cl, dl
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], dl
# mem access: [30] 0x5128 cl 4:40 | [80] 0x5128 cl 4:40
and rdi, 0b1111111111000 # instrumentation
mov dword ptr [r14 + rdi], eax
# mem access: [30] 0x5d28 cl 52:40 | [80] 0x5d28 cl 52:40
inc edi
and rcx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rcx]
# mem access: [30] 0x53f0 cl 15:48 | [80] 0x53f0 cl 15:48
add cl, 66 # instrumentation
and rdi, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rdi]
# mem access: [30] 0x5d29 cl 52:41 | [80] 0x5d29 cl 52:41
inc bx
and rcx, 0b1111111111111 # instrumentation
mov bx, word ptr [r14 + rcx]
# mem access: [30] 0x5332 cl 12:50 | [80] 0x5332 cl 12:50
and rsi, 0b1111111111000 # instrumentation
add qword ptr [r14 + rsi], rdx
# mem access: [30] 0x6ff8-0x6ff8 cl 63:56 | [80] 0x6ff8-0x6ff8 cl 63:56
and rdx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdx]
# mem access: [30] 0x6ffa cl 63:58 | [80] 0x5041 cl 1:1
mov dx, di
and rcx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rcx], eax
# mem access: [30] 0x5332 cl 12:50 | [80] 0x5332 cl 12:50
neg dx
mov rsi, 0b1111111111000 #
mov rdi, 0b1111111111111 #
mov rcx, 0b1111111111111 #
mov rbx, 0b1111111111111 #
add rsi, 0b1111111111000 #
mov dl, -91 #
mov al, 55 #
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], 1 #
# mem access: [30] 0x6fff-0x6fff cl 63:63 | [80] 0x6fff-0x6fff cl 63:63
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence
.section .data.vm
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence
.function_vm_1:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
add cl, 72 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
# mem access: [30] 0xb000 cl 0:0 | [80] 0xb000 cl 0:0
mov rbx, qword ptr [r14 +0x2008] # noremove
# mem access: [30] 0xb008 cl 0:8 | [80] 0xb008 cl 0:8
mov rcx, qword ptr [r14 +0x2010] # noremove
# mem access: [30] 0xb010 cl 0:16 | [80] 0xb010 cl 0:16
mov rdx, qword ptr [r14 +0x2018] # noremove
# mem access: [30] 0xb018 cl 0:24 | [80] 0xb018 cl 0:24
mov rsi, qword ptr [r14 +0x2020] # noremove
# mem access: [30] 0xb020 cl 0:32 | [80] 0xb020 cl 0:32
mov rdi, qword ptr [r14 +0x2028] # noremove
# mem access: [30] 0xb028 cl 0:40 | [80] 0xb028 cl 0:40
mfence  # noremove
and rax, 0b1111111111111 # instrumentation
cmovb rdx, qword ptr [r14 + rax]
# mem access: [30] 0x9a21 cl 40:33 | [80] 0x9a21 cl 40:33
and rdi, 0b1111111111000 # instrumentation
inc qword ptr [r14 + rdi]
# mem access: [30] 0x9e90-0x9e90 cl 58:16 | [80] 0x9e90-0x9e90 cl 58:16
mov rcx, 0b1111111111111 #
and rsi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rsi]
# mem access: [30] 0xa4de cl 19:30 | [80] 0xa4de cl 19:30
xor al, dl
and rbx, 0b1111111111000 # instrumentation
btr qword ptr [r14 + rbx], 1
# mem access: [30] 0xa9f8-0xa9f8 cl 39:56 | [80] 0xa9f8-0xa9f8 cl 39:56
and rbx, 0b1111111111000 # instrumentation
add dword ptr [r14 + rbx], -92
and rdi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdi]
setns dl
mul dil
dec eax
and rdx, 0b1111111111111 # instrumentation
movsx ecx, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mov ebx, dword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 1 # instrumentation
and dx, word ptr [r14 + rdi] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rdi]
mul si
and rbx, 0b1111111111000 # instrumentation
mov qword ptr [r14 + rbx], rcx
or esi, edx
add rbx, 0b1111111111111 #
xadd cl, dl
and rdi, 0b1111111111000 # instrumentation
mov byte ptr [r14 + rdi], bl
mov rbx, 0b1111111111111 #
and rsi, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rsi]
add rdx, 0b1111111111111 #
add rdx, 0b1111111111111 #
and rsi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rsi]
mov rax, 0b1111111111111 #
and rcx, 0b1111111111000 # instrumentation
dec dword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rdx], 3
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence
.function_vm_2:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence
.section .data.main
.test_case_exit:nop
