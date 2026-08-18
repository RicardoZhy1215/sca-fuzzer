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
add bl, 40 # instrumentation
cmovnz rdx, rsi
dec cl
cmpxchg bl, bl
and rdi, 0b1111111111111 # instrumentation
sub di, word ptr [r14 + rdi]
add al, al
cmovnle ax, cx
lea rcx, qword ptr [rsi]
and rax, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rax], 101
and rcx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rcx], rcx
or cl, bl
lea si, qword ptr [rax + rbx]
cmp cl, dl
test rsi, rdx
and rsi, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
btr word ptr [r14 + rsi], cx
btc rsi, 99
cmp edi, -1
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, 116 # instrumentation
adc al, dl
setnb cl
and rdi, 0b1111111111111 # instrumentation
movsx ecx, byte ptr [r14 + rdi]
sbb al, cl
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], 116
and rcx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rcx], 1
and rdi, 0b1111111111111 # instrumentation
cmovnb bx, word ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
add word ptr [r14 + rax], -49
and rdx, 0b1111111111111 # instrumentation
cmovz rcx, qword ptr [r14 + rdx]
and bl, -121
and rcx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], 114
bts cx, 0
add al, -85 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovo rcx, qword ptr [r14 + rcx]
add al, bl
setnl dl
neg rbx
add al, dil
adc bl, 12
not cx
cmpxchg ebx, ecx
and rbx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], -10
and rdi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdi]
cmovb eax, esi
adc dl, al
adc ax, 10122
and rsi, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rsi], 4
add cl, -117 # instrumentation
setnl cl
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
add dl, 78 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
adc rbx, rsi
and rax, 89
cmp dl, dl
nop esi
sbb si, dx
and rdi, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rdi]
sbb cl, cl
imul ebx, ebx, -37
add dl, 45 # instrumentation
cmovnl rbx, rdi
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], -31
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], al
add bl, 78
and rcx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rcx]
xadd edi, esi
test di, cx
sbb eax, 1651022204
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 0b1000000000000000000000000000000 # instrumentation
bsr edi, dword ptr [r14 + rcx]
add dl, -38 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnbe cx, word ptr [r14 + rax]
lea ebx, qword ptr [rdi]
sbb cl, al
and rcx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rcx]
or rsi, -62
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rax
and rdx, 0b1111111111111 # instrumentation
and bl, byte ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
movsx di, byte ptr [r14 + rcx]
or sil, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv sil
add dl, -93 # instrumentation
mov al, bl
and rax, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rax], si
and rdx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdx], 42
or cx, 1 # instrumentation
and dx, cx # instrumentation
shr dx, 1 # instrumentation
div cx
add bl, -128 # instrumentation
setnl dil
and rdx, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rdx]
lea rsi, qword ptr [rsi]
lea di, qword ptr [rdx + rcx]
and rcx, 0b1111111111111 # instrumentation
cmovnp rdi, qword ptr [r14 + rcx]
mul al
dec rbx
lea esi, qword ptr [rdx + rdx + 15510]
and rdi, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
movzx rdx, word ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdi], eax
and rbx, 0b1111111111111 # instrumentation
cmovno rax, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovnz rsi, qword ptr [r14 + rax]
add cl, cl
sbb eax, -512742190
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], -28
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
