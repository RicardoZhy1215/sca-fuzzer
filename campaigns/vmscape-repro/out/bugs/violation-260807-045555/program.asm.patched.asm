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
xor bl, al
setl cl
setnb bl
cmovbe bx, di
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], 84
lea rsi, qword ptr [rax]
cmovle eax, ecx
and rbx, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rbx]
adc ax, -7179
cmovs rdx, rbx
and rsi, 0b1111111111111 # instrumentation
and cl, byte ptr [r14 + rsi]
or edx, 1 # instrumentation
add bl, -103 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rcx], -16
xadd dil, cl
sub al, cl
lea di, qword ptr [rbx]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, -91 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovp rdx, qword ptr [r14 + rdi]
sub edi, 89
and rax, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 71
setbe al
and rdx, 0b1111111111111 # instrumentation
cmovnz rsi, qword ptr [r14 + rdx]
add dl, -15
and rdx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rdx], 0
add ebx, esi
nop
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 1 # instrumentation
and dx, word ptr [r14 + rbx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rbx]
cmp sil, 47
and rdx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdx], -2
and rax, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rax], dl
imul ax, cx, 68
add dl, 98 # instrumentation
and rbx, 0b1111111111111 # instrumentation
movzx ecx, byte ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rdx]
cmovnp dx, cx
and rdx, 0b1111111111111 # instrumentation
cmovnl cx, word ptr [r14 + rdx]
adc cl, cl
and rdi, 0b1111111111111 # instrumentation
cmp cl, byte ptr [r14 + rdi]
not cl
and rsi, 0b1111111111111 # instrumentation
or rdx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
dec word ptr [r14 + rax]
test al, -109
add dl, cl
and rbx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rbx], 110
and rbx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rbx], eax
sbb al, 53
lea edx, qword ptr [rdx + rdx + 58256]
sets dil
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
add bl, -75 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rdi, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rdi]
sub cl, al
and rsi, 0b1111111111111 # instrumentation
cmp ecx, dword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rax], 1
sub cl, bl
mov rsi, rbx
lea bx, qword ptr [rcx + rcx]
movsx bx, bl
lea ebx, qword ptr [rax + rsi]
lea rbx, qword ptr [rcx]
cmp dl, cl
test esi, edx
or bl, 19
lea rdx, qword ptr [rdi + rdx + 24662]
add ax, -20122
and dl, bl
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
sub dl, sil
lea rsi, qword ptr [rsi + rdi + 16441]
mov ebx, -1998934388
lea esi, qword ptr [rsi + rcx + 29217]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdx], -62
lea edi, qword ptr [rdx]
and rbx, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rbx], al
lea cx, qword ptr [rcx]
cmpxchg bl, bl
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 1 # instrumentation
and edx, dword ptr [r14 + rcx] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
dec word ptr [r14 + rsi]
test dl, 7
xchg ecx, esi
lea dx, qword ptr [rax + rsi]
and rdi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdi]
test dl, 119
and rdi, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdi], -106
and rsi, 0b1111111111111 # instrumentation
movsx esi, byte ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rbx], esi
or rdi, 84
and rcx, 0b1111111111111 # instrumentation
cmovnbe dx, word ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsr edi, dword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rdi], rcx
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
movzx cx, byte ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax]
adc ecx, -43
and rax, 0b1111111111111 # instrumentation
cmovp edx, dword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rax
or ax, 16184
and rbx, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rbx], 27
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
