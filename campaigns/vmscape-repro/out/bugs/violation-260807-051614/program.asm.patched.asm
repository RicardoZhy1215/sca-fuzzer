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
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], 54
and rdx, 0b1111111111111 # instrumentation
add cl, byte ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
adc ax, word ptr [r14 + rdi]
bswap esi
cmp rbx, 3
sub cl, 122
movzx esi, ax
cmovle si, ax
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], 100
movsx esi, bx
and rsi, 0b1111111111111 # instrumentation
cmp bl, byte ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rsi], 27
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], cl
and rdi, 0b1111111111111 # instrumentation
mov ebx, dword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rdx]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, 84 # instrumentation
sbb ebx, edx
and rax, 2085879958
and rdi, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rdi], 38
adc ecx, edx
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], ecx
lea rdi, qword ptr [rbx + rax]
and rsi, 0b1111111111111 # instrumentation
btc word ptr [r14 + rsi], 1
test al, 35
not edi
and rcx, 0b1111111111111 # instrumentation
xor cx, word ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovnb rcx, qword ptr [r14 + rbx]
cmovnp ecx, edx
and rdi, 0b1111111111111 # instrumentation
mov dx, word ptr [r14 + rdi]
xor al, -19
sbb al, -124
and rax, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rax], bl
bt dx, 244
add al, -19 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovs dx, word ptr [r14 + rdi]
cmovnle edi, edx
and rcx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rcx], cl
sets bl
lea esi, qword ptr [rcx + rcx + 43984]
or edx, ebx
and rdi, 0b1111111111111 # instrumentation
not word ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rcx
neg dl
and rcx, 0b1111111111111 # instrumentation
cmovp bx, word ptr [r14 + rcx]
adc di, ax
and rcx, 0b1111111111111 # instrumentation
movzx bx, byte ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rsi], cx
cmovnp edx, esi
and rbx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rbx], 98
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
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
btr bx, si
xchg bl, dl
and rax, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rax], bl
add esi, esi
or esi, 1 # instrumentation
and edx, esi # instrumentation
shr edx, 1 # instrumentation
div esi
lea edi, qword ptr [rbx + rbx + 35064]
and rcx, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rcx], 108
add dl, 88 # instrumentation
movzx eax, cl
setbe cl
lea ecx, qword ptr [rax + rcx]
and rsi, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsr ecx, dword ptr [r14 + rdx]
or dx, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
add bl, 32 # instrumentation
sbb al, cl
and rsi, 0b1111111111111 # instrumentation
cmovbe si, word ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], dl
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add al, 6 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdx], -61
and rax, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rax], dx
and rdi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdi]
sbb bl, 54
and rcx, 0b1111111111111 # instrumentation
mov dx, word ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rdx]
lea rsi, qword ptr [rax + rdx]
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], ax
and rcx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rcx], 1
and rcx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdx], 84
and rdi, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rdi], ecx
test dl, 86
and rdx, 0b1111111111111 # instrumentation
sub bx, word ptr [r14 + rdx]
or cx, si
cmp cl, bl
and rdi, 0b1111111111111 # instrumentation
cmovle bx, word ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock bts word ptr [r14 + rbx], 2
xchg rdx, rbx
cwde
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
xadd ebx, esi
and rcx, 0b1111111111111 # instrumentation
cmovno ebx, dword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rcx], 13
sub al, 84
and rdx, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rdx]
movzx ebx, al
mov dl, -81
and rax, 0b1111111111000 # instrumentation
and ecx, 0b111 # instrumentation
lock bts dword ptr [r14 + rax], ecx
and rsi, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rsi], bx
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
