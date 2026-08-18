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
xchg bl, al
not ax
mul rsi
lea rdi, qword ptr [rcx]
cmp eax, -964629095
add ebx, eax
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rdx], bx
mov bl, dl
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx]
sahf
movzx ecx, dl
lea rax, qword ptr [rax + rsi]
neg cl
lea cx, qword ptr [rdi]
and rdx, 0b1111111111111 # instrumentation
add bx, word ptr [r14 + rdx]
lea eax, qword ptr [rdi + rdx + 17981]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add al, 61 # instrumentation
adc dil, -91
and rdi, 0b1111111111111 # instrumentation
sbb cx, word ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
setl byte ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rcx], ebx
cmovns rcx, rdx
sub al, 5
imul edx, esi
and rdx, 0b1111111111111 # instrumentation
and di, word ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], 121
cmovo ebx, ecx
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], 44
and rax, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rax], si
cmovs di, dx
and rbx, 0b1111111111111 # instrumentation
cmovb edi, dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
lea rdi, qword ptr [rbx + rdi]
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
adc sil, sil
and rcx, 0b1111111111111 # instrumentation
cmp rax, qword ptr [r14 + rcx]
sub al, dl
lea ebx, qword ptr [rsi]
lea rax, qword ptr [rbx]
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
add al, -44 # instrumentation
lea si, qword ptr [rcx + rdx + 54394]
and rdx, 0b1111111111111 # instrumentation
cmovnz rax, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
adc word ptr [r14 + rsi], dx
adc al, al
lea di, qword ptr [rsi + rdi]
sbb bl, 54
and rcx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rcx], dx
and rbx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rbx], esi
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
add dl, -78 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
btr si, 56
and rsi, 0b1111111111111 # instrumentation
cmovz rsi, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
btr word ptr [r14 + rax], dx
inc ecx
btc eax, esi
setnbe dl
and rdx, 0b1111111111111 # instrumentation
add dl, byte ptr [r14 + rdx]
cmp cx, 70
lea dx, qword ptr [rbx]
and rdx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdx], eax
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr rdx, qword ptr [r14 + rbx]
cmp bl, 88
add bl, cl
and rax, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rax], dl
lea esi, qword ptr [rcx + rbx + 56241]
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], cl
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdx, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rdx]
test rsi, rdi
sbb dl, -50
cmovnp rdi, rbx
sub rax, rdx
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdx
and rdx, 0b1111111111111 # instrumentation
and si, word ptr [r14 + rdx]
or edi, 1 # instrumentation
and edx, edi # instrumentation
shr edx, 1 # instrumentation
div edi
and rax, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rax], bl
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rcx], di
not sil
test rbx, 443386747
sub ax, 122
clc
and rdi, 0b1111111111111 # instrumentation
sub ecx, dword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rbx], ax
and rdi, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rdi], 1
btc rdx, rcx
or dil, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv dil
add al, 74 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnb dx, word ptr [r14 + rax]
xor bl, cl
and rax, 0b1111111111111 # instrumentation
add byte ptr [r14 + rax], 30
lea ecx, qword ptr [rax + rcx + 35875]
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl
cmp eax, 65872612
lea eax, qword ptr [rcx + rdx]
and rax, 0b1111111111111 # instrumentation
xor word ptr [r14 + rax], -49
cmpxchg rbx, rcx
sub ecx, 118
and rbx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rbx], 53
movzx edi, bx
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
