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
add cl, -33 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdx]
# mem access: [45] 0x5624 cl 24:36 | [95] 0x5624 cl 24:36
and rbx, 0b1111111111000 # instrumentation
add dword ptr [r14 + rbx], edi
# mem access: [45] 0x6248-0x6248 cl 9:8 | [95] 0x6248-0x6248 cl 9:8
and rax, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rax]
# mem access: [45] 0x577f cl 29:63 | [95] 0x577f cl 29:63
setnz al
or edi, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
inc edi
and rax, 0b1111111111000 # instrumentation
add word ptr [r14 + rax], -33
# mem access: [45] 0x6200-0x6200 cl 8:0 | [95] 0x5800-0x5800 cl 32:0
and rdx, 0b1111111111000 # instrumentation
inc word ptr [r14 + rdx]
# mem access: [45] 0x5000-0x5000 cl 0:0 | [95] 0x5000-0x5000 cl 0:0
mov dl, 120
and rdx, 0b1111111111111 # instrumentation
mov ebx, dword ptr [r14 + rdx]
# mem access: [45] 0x5078 cl 1:56 | [95] 0x5078 cl 1:56
lea dx, qword ptr [rcx + rcx + 2092]
and rdx, 0b1111111111000 # instrumentation
add byte ptr [r14 + rdx], -27
# mem access: [45] 0x6158-0x6158 cl 5:24 | [95] 0x6158-0x6158 cl 5:24
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx]
# mem access: [45] 0x5497 cl 18:23 | [95] 0x5497 cl 18:23
mov eax, esi
mov al, bl
imul bx, di, -80
mov rax, 0b1111111111111 #
mov rsi, 0b1111111111111 #
mov rdi, 0b1111111111000 #
and rsi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rsi], 0b1000000000000000 #
# mem access: [45] 0x6fff cl 63:63 | [95] 0x6fff cl 63:63
and rcx, 0b1111111111000 # instrumentation
mov dword ptr [r14 + rcx], edi
# mem access: [45] 0x5068 cl 1:40 | [95] 0x5068 cl 1:40
mov rax, 0b1111111111111 #
mov rsi, 0b1111111111111 #
mov rdi, 0b1111111111000 #
mov rbx, 0b1111111111111 #
mov rsi, 0b1111111111000 #
mov rcx, 0b1111111111111 #
mov rdx, 0b1111111111111 #
mov rsi, 0b1111111111111 #
or si, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
mov dx, 0b11 #
mov rax, 0b1111111111111 #
mov rax, 1330433608
mov rdx, 0b1111111111111 #
mov rax, 0b1111111111000 #
mov rax, 0b1111111111000 #
mov rcx, 0b1111111111111 #
mov rsi, 0b1111111111000 #
mov rdx, 0b1111111111111 #
mov rsi, 0b1111111111111 #
mov rdx, 0b1111111111000 #
mov rax, 0b1111111111111 #
mov rax, 0b1111111111000 #
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
add cl, -111 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
# mem access: [45] 0xb000 cl 0:0 | [95] 0xb000 cl 0:0
mov rbx, qword ptr [r14 +0x2008] # noremove
# mem access: [45] 0xb008 cl 0:8 | [95] 0xb008 cl 0:8
mov rcx, qword ptr [r14 +0x2010] # noremove
# mem access: [45] 0xb010 cl 0:16 | [95] 0xb010 cl 0:16
mov rdx, qword ptr [r14 +0x2018] # noremove
# mem access: [45] 0xb018 cl 0:24 | [95] 0xb018 cl 0:24
mov rsi, qword ptr [r14 +0x2020] # noremove
# mem access: [45] 0xb020 cl 0:32 | [95] 0xb020 cl 0:32
mov rdi, qword ptr [r14 +0x2028] # noremove
# mem access: [45] 0xb028 cl 0:40 | [95] 0xb028 cl 0:40
mfence  # noremove
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax]
# mem access: [45] 0x9485 cl 18:5 | [95] 0x9485 cl 18:5
lea rdx, qword ptr [rdi]
add cl, -115
and rax, 0b1111111111111 # instrumentation
movsx si, byte ptr [r14 + rax]
# mem access: [45] 0xab07 cl 44:7 | [95] 0xab07 cl 44:7
adc dl, cl
and rbx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rbx], 67
# mem access: [45] 0x9940-0x9940 cl 37:0 | [95] 0x9940-0x9940 cl 37:0
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], bl
# mem access: [45] 0x939a-0x939a cl 14:26 | [95] 0x939a-0x939a cl 14:26
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], dl
# mem access: [45] 0x939a cl 14:26 | [95] 0x939a cl 14:26
and rax, 0b1111111111111 # instrumentation
imul di, word ptr [r14 + rax], 84
# mem access: [45] 0xab07 cl 44:7 | [95] 0xab07 cl 44:7
add cl, -119 # instrumentation
and rax, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rax]
# mem access: [45] 0xab07 cl 44:7 | [95] 0xab07 cl 44:7
and rsi, 0b1111111111000 # instrumentation
add dword ptr [r14 + rsi], edi
# mem access: [45] 0x9048-0x9048 cl 1:8 | [95] 0x9048-0x9048 cl 1:8
add dl, cl
mov al, -97
and rdi, 0b1111111111111 # instrumentation
add bl, byte ptr [r14 + rdi]
# mem access: [45] 0xa89c cl 34:28 | [95] 0xa89c cl 34:28
xchg cx, ax
and rdx, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rdx]
# mem access: [45] 0x9365-0x9365 cl 13:37 | [95] 0x9365-0x9365 cl 13:37
and rax, -76
lea rsi, qword ptr [rsi + rdi + 36558]
add al, -39
and rbx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rbx]
# mem access: [45] 0x998b-0x998b cl 38:11 | [95] 0x998b-0x998b cl 38:11
imul edi, edx
and rax, 0b1111111111000 # instrumentation
bts qword ptr [r14 + rax], 2
# mem access: [45] 0xa778-0xa778 cl 29:56 | [95] 0xa778-0xa778 cl 29:56
or sil, -98
add rcx, rdi
add ecx, -105
btr edx, esi
add cl, -101 # instrumentation
cmovo eax, ecx
bts di, 69
cmp esi, eax
cmp sil, 82
mov cx, cx
bswap edi
bt esi, 81
add al, 27
add edx, edi
add ax, -18386
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], 100
and rbx, 0b1111111111111 # instrumentation
movsx rbx, word ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rsi], rax
cmovns ecx, ebx
and rcx, 0b1111111111111 # instrumentation
movzx ebx, byte ptr [r14 + rcx]
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
