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
lea rbx, qword ptr [rbx + rsi]
cmp bl, 125
and rax, 0b1111111111111 # instrumentation
cmovo rcx, qword ptr [r14 + rax]
lea eax, qword ptr [rdx + rbx]
and rbx, 0b1111111111111 # instrumentation
test word ptr [r14 + rbx], 9491
cmovz rdx, rbx
xadd eax, ecx
bt edi, 102
btr eax, ecx
sub rbx, rsi
xor al, 18
test rsi, rsi
and ax, 16844
and rbx, 0b1111111111111 # instrumentation
sub bx, word ptr [r14 + rbx]
add eax, eax
add bl, cl
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add al, 55 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], edi
and rax, 0b1111111111111 # instrumentation
cmovo edx, dword ptr [r14 + rax]
cmovno rbx, rax
and rsi, 0b1111111111111 # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi]
btr rcx, 3
and rbx, 0b1111111111111 # instrumentation
imul rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rax]
sub al, -43
and rsi, 0b1111111111111 # instrumentation
and ebx, dword ptr [r14 + rsi]
dec rdx
and bx, 123
and rdx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdx], ax
and rbx, 0b1111111111111 # instrumentation
and word ptr [r14 + rbx], bx
movzx edx, ax
and rax, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rax]
adc rax, rdi
and rsi, 0b1111111111111 # instrumentation
movsx rbx, word ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], al
setz dl
or rax, -64
and rax, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rsi]
xadd dil, bl
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax
and rdi, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], -464109460
inc rdx
and rsi, 0b1111111111111 # instrumentation
cmovnl ebx, dword ptr [r14 + rsi]
lea ax, qword ptr [rdx]
and rax, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rax]
add cl, 23 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rax], 86
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
add al, 104
or esi, 111
and rbx, 0b1111111111111 # instrumentation
sub esi, dword ptr [r14 + rbx]
lea di, qword ptr [rcx + rax + 30333]
and rax, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rax], eax
lea di, qword ptr [rdi + rbx + 37997]
cmovo ebx, eax
lea ecx, qword ptr [rdi]
movsx edx, si
xor di, di
cmovp edi, edi
imul ebx, esi
adc cl, cl
and rbx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rbx]
or al, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv al
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, 66 # instrumentation
and rax, 0b1111111111111 # instrumentation
movzx rdx, word ptr [r14 + rax]
setz dl
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], 68
mov rax, 7476923485458857813
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], cl
lea esi, qword ptr [rax]
lea rax, qword ptr [rsi + rax + 4398]
and rcx, 0b1111111111111 # instrumentation
cmovnz rdx, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rdx], edx
lea rdx, qword ptr [rcx + rsi + 63273]
and rcx, 0b1111111111111 # instrumentation
cmovnbe ax, word ptr [r14 + rcx]
bts rax, 58
and rsi, 0b1111111111111 # instrumentation
sub rsi, qword ptr [r14 + rsi]
sbb rbx, 18
and rcx, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rcx], 3
adc dx, -77
sbb al, 27
add bl, 92
mov rsi, -2082384419298467724
and rbx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rbx], bl
bt di, 135
sbb dil, -79
and rdx, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdx]
lea rax, qword ptr [rsi + rax + 21634]
and rdx, 0b1111111111111 # instrumentation
cmovnz bx, word ptr [r14 + rdx]
xchg ecx, eax
xor rbx, rsi
and rcx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rcx]
clc
add al, 6
lea rcx, qword ptr [rbx + rdi + 33740]
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
