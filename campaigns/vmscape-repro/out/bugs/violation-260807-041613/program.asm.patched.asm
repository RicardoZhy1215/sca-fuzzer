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
and rdi, 0b1111111111111 # instrumentation
and word ptr [r14 + rdi], dx
setns bl
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], rcx
cmovno rsi, rcx
cmovnp rax, rdi
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], ax
lea di, qword ptr [rax]
and rsi, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rsi]
lea rax, qword ptr [rdi + rdx + 47026]
nop ecx
add dl, -55
lea rbx, qword ptr [rbx]
xor al, cl
and rsi, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rcx], cx
lea rdx, qword ptr [rbx + rcx + 65337]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
inc rdx
and rax, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rax], al
and rbx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rbx], cl
and rbx, 0b1111111111111 # instrumentation
movzx edi, byte ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdx]
sub ecx, eax
and rdi, 0b1111111111111 # instrumentation
add esi, dword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rcx]
lea eax, qword ptr [rax + rcx]
and rdx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rdx], sil
and rcx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], 86
lea dx, qword ptr [rdx + rcx + 61253]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdi], al
and rdx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdx], 53
lea rax, qword ptr [rdx]
and rbx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rbx]
cmp rcx, rcx
cmovle esi, edx
lea dx, qword ptr [rsi + rcx + 29213]
cmovp bx, di
sub al, 4
and rbx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rbx], dl
btc rbx, rax
sahf
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], 88
and rdx, 0b1111111111111 # instrumentation
movzx rdi, word ptr [r14 + rdx]
xadd cl, al
and rcx, 0b1111111111111 # instrumentation
cmovnl cx, word ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rcx], al
and rax, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rax], 44
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
or eax, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv eax
and rbx, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rbx]
and ax, 32
cmovb dx, dx
or cl, 102
sbb dl, bl
lea si, qword ptr [rdi]
mov edi, ecx
lea ecx, qword ptr [rax + rax + 48055]
lea ebx, qword ptr [rdi]
lea cx, qword ptr [rbx + rsi + 32722]
bt rdi, rdx
sbb al, cl
inc rbx
cld
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
neg dx
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], cl
or di, si
sbb rax, rsi
sbb al, -78
test cl, bl
sub al, al
test bl, al
mov al, 105
or al, al
and rsi, 0b1111111111000 # instrumentation
lock bts word ptr [r14 + rsi], 4
add dl, -84 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovns ebx, dword ptr [r14 + rcx]
mul cl
sbb eax, -1032590309
xor bl, al
cmpxchg bl, cl
lea rdi, qword ptr [rcx + rax + 41842]
and rcx, 0b1111111111111 # instrumentation
or edi, dword ptr [r14 + rcx]
cmovnle di, cx
xor eax, -196879124
and rbx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rbx], 60
and rbx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rbx], eax
lea bx, qword ptr [rdi]
xadd cl, bl
lea edx, qword ptr [rax]
sbb rax, -848642282
add sil, 72
and rbx, 0b1111111111111 # instrumentation
adc cl, byte ptr [r14 + rbx]
mov dl, cl
lea si, qword ptr [rbx + rdx]
sub eax, 1653185635
cmovl eax, ebx
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
