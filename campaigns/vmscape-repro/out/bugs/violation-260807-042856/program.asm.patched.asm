.intel_syntax noprefix
.section .data.main
.function_main_0:
.bb_main_0.entry:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]
.exit_main_0:
.function_main_1:
.bb_main_1.entry:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]
.exit_main_1:
.function_main_2:
.bb_main_2.entry:
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
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]
.exit_main_3:
.function_main_4:
.bb_main_4.entry:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop
.exit_main_4:
.section .data.vmvictim
.function_vmvictim_0:
.bb_vmvictim_0.entry:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000000000000000 # instrumentation
bsf di, word ptr [r14 + rcx]
add al, 100
setnle dl
mov bl, cl
lea rax, qword ptr [rdi + rsi + 21751]
and rbx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rbx], al
cmovnle esi, esi
and esi, 30
and rbx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rbx], -15
lea rsi, qword ptr [rcx + rdx]
and rdx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdx]
cmp eax, edi
xor al, 26
mul si
and rdx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdx], rdx
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], cl
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add al, -18 # instrumentation
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 0
and rdi, 0b1111111111111 # instrumentation
cmovnz rdi, qword ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], -79
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx]
add dl, -120 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovo rax, qword ptr [r14 + rbx]
sets al
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf cx, word ptr [r14 + rsi]
add bl, dl
cmovl eax, eax
sbb ax, -32021
and rsi, 0b1111111111111 # instrumentation
cmovp edi, dword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rcx], 81
cmp ecx, ebx
or dl, 1 # instrumentation
add bl, 31 # instrumentation
adc rax, 52
and rax, 0b1111111111111 # instrumentation
movsx rdi, word ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdi], 31
and rsi, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rsi], 74
movzx ecx, al
and rsi, 0b1111111111111 # instrumentation
and eax, 0b111 # instrumentation
bts dword ptr [r14 + rsi], eax
and rbx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rbx], -671874045
and rax, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rcx], 51
add al, cl
mov bx, -2355
or bx, 1 # instrumentation
and dx, bx # instrumentation
shr dx, 1 # instrumentation
div bx
xadd bx, bx
and rbx, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rbx]
add ax, -25869
movzx rax, dx
add rax, rcx
lfence
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence
.exit_vmvictim_1:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence
.exit_vm_0:
.function_vm_1:
.bb_vm_1.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
add cl, -62 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setnp cl
sub al, -6
and rsi, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rsi]
add dl, 62 # instrumentation
setle cl
cmp al, bl
xor al, -115
and rax, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rax], -75
btc edi, eax
add esi, -68
and rbx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rbx], 2
add cl, 70 # instrumentation
cmovns eax, edx
btr rbx, 208
test eax, -656872384
cmp al, 84
cmovnle bx, di
and rdi, 0b1111111111111 # instrumentation
cmovns cx, word ptr [r14 + rdi]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
btr word ptr [r14 + rdx], 7
clc
imul edi
and rsi, 0b1111111111111 # instrumentation
movzx di, byte ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], bl
lea cx, qword ptr [rax + rax]
xor eax, 2093341123
and rbx, 0b1111111111111 # instrumentation
and word ptr [r14 + rbx], -54
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rsi], esi
sub eax, -1158635459
cmpxchg dl, bl
bt rdx, 19
add cl, 73 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovp rcx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr rsi, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdx], di
cmpxchg si, cx
and rax, 0b1111111111111 # instrumentation
and di, word ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
and rax, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rcx], 1
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 1 # instrumentation
and dx, word ptr [r14 + rax] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rax]
cld
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rbx
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 1 # instrumentation
add dl, 55 # instrumentation
cmovl cx, ax
and rdx, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rdx], eax
btr esi, 6
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], dl
and rax, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rax], bl
and rcx, 0b1111111111111 # instrumentation
movzx rsi, word ptr [r14 + rcx]
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence
.exit_vm_3:
.function_vm_2:
.bb_vm_2.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence
.exit_vm_2:
.section .data.main
.test_case_exit:nop
