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
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf si, word ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], -93
movzx edi, bl
and rdx, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdx]
adc rbx, rsi
sbb dil, 101
cmovnl rbx, rcx
neg bx
test rdx, rdi
xchg di, di
and rax, 0b1111111111111 # instrumentation
mov word ptr [r14 + rax], di
lea esi, qword ptr [rsi + rdi + 55036]
xor al, 96
and rdx, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rdx], 23
add cl, 32 # instrumentation
setle al
cmp di, -95
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, 15 # instrumentation
cmovs ax, bx
cmc
and rdi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdi], al
and rdx, 0b1111111111111 # instrumentation
cmovp esi, dword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
movzx esi, word ptr [r14 + rdx]
lea bx, qword ptr [rdi + rax + 26915]
and rdx, 0b1111111111111 # instrumentation
cmovb dx, word ptr [r14 + rdx]
cmpxchg bl, cl
setp bl
adc si, bx
and rcx, 0b1111111111111 # instrumentation
cmovz si, word ptr [r14 + rcx]
sub esi, ebx
and rdi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rdi], 88
xor al, -55
and rbx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rbx], 36
and rdx, 0b1111111111111 # instrumentation
imul ax, word ptr [r14 + rdx], -79
lea bx, qword ptr [rbx]
and rax, 0b1111111111111 # instrumentation
sbb cx, word ptr [r14 + rax]
sub al, al
and rbx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rbx], di
or ebx, 0b1000000000000000000000000000000 # instrumentation
bsf esi, ebx
cmp rbx, rdx
btr bx, 102
add cl, -96 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovl di, word ptr [r14 + rdi]
sub ecx, edi
lea cx, qword ptr [rdi + rax + 80]
and rdx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdx], al
mul bl
add al, -116 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovnl dx, word ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rdx]
setnbe dil
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
lea rdx, qword ptr [rcx + rbx]
lea rdx, qword ptr [rax + rdi + 23270]
btc edi, 135
add cl, 91 # instrumentation
nop di
lea dx, qword ptr [rax + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
and si, 0b111 # instrumentation
lock bts word ptr [r14 + rdx], si
and rsi, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rsi], 2
or dil, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv dil
and rsi, 0b1111111111111 # instrumentation
movzx rcx, word ptr [r14 + rsi]
or rax, -1888907768
lea bx, qword ptr [rbx]
and rcx, 0b1111111111111 # instrumentation
add esi, dword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
imul ebx, dword ptr [r14 + rcx], 27
or cl, dl
lea cx, qword ptr [rdi + rdi + 39724]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rcx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rcx]
adc al, -14
sbb dl, -26
and rdi, 0b1111111111111 # instrumentation
xor bx, word ptr [r14 + rdi]
lea dx, qword ptr [rsi]
and rdi, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rax]
cmovz di, si
lea bx, qword ptr [rdx + rsi]
and rdx, 0b1111111111111 # instrumentation
cmovno si, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovnb dx, word ptr [r14 + rax]
setle dl
and rdx, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rdx]
lea cx, qword ptr [rsi]
and rsi, 0b1111111111111 # instrumentation
not byte ptr [r14 + rsi]
bts esi, ecx
add cl, 35 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rdx]
imul ecx
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl
and rdx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdx], -118
and rax, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
btr word ptr [r14 + rax], bx
and rdi, 0b1111111111111 # instrumentation
btc word ptr [r14 + rdi], 0
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rbx], di
and rdi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdi], cl
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 1 # instrumentation
and edx, dword ptr [r14 + rsi] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rsi]
test dl, 125
and ax, -2057
and rsi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rsi]
dec dil
inc bl
xchg dx, si
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rdx
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
