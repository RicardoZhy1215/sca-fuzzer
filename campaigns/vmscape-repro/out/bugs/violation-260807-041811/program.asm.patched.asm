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
test dl, 42
and bl, 53
and rsi, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rsi], di
and rdi, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], -678168178
adc dil, 110
add di, -67
sub dl, bl
or ecx, 1 # instrumentation
and edx, ecx # instrumentation
shr edx, 1 # instrumentation
div ecx
and rbx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rbx], rbx
sub al, -5
cmovo edi, eax
xadd al, al
sub eax, eax
cmovnbe rax, rdx
sub al, -64
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, 49 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovle si, word ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], -90
and dl, -19
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
and dil, byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rcx], al
sub rsi, 11
and rsi, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rsi]
cmpxchg rcx, rdi
inc rsi
and rdx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdx], edi
sub eax, 1599099195
nop cx
and rcx, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
test word ptr [r14 + rdi], cx
setnl dil
add cl, bl
movzx rcx, sil
and rdi, 0b1111111111111 # instrumentation
cmovb rdi, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovnz dx, word ptr [r14 + rbx]
xor cl, -115
or sil, -35
imul cl
test dil, -115
and rdx, 0b1111111111111 # instrumentation
cmovnbe si, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rax], -58
and rdx, 0b1111111111111 # instrumentation
cmovl cx, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
bts word ptr [r14 + rax], dx
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
xor ebx, dword ptr [r14 + rdx]
add dx, di
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], dl
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
add al, 64 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
sbb rax, 1043092019
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], rax
and rcx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rcx], ax
setz bl
and rbx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rbx]
lea rcx, qword ptr [rcx + rsi]
neg bx
cmovnb ecx, esi
and rdx, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rdx], 4
inc cl
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl
add bl, 96 # instrumentation
setle dl
dec al
cmp edi, edi
and rax, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add dl, 31 # instrumentation
and rbx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovnbe esi, dword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], sil
and rax, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rax], al
and rdi, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rdx]
sub bx, si
and rbx, 0b1111111111111 # instrumentation
cmovnb di, word ptr [r14 + rbx]
sbb bl, -84
lea dx, qword ptr [rbx + rax]
and rdi, 0b1111111111111 # instrumentation
sbb ax, word ptr [r14 + rdi]
dec cx
and rbx, 0b1111111111000 # instrumentation
xchg word ptr [r14 + rbx], dx
sbb bl, bl
and rbx, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rbx], al
setb bl
cmovs rcx, rbx
bt ecx, esi
btr si, dx
mul di
lea rsi, qword ptr [rax + rdi]
sbb edi, esi
imul di, si
and rdi, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
bts dword ptr [r14 + rdi], edx
sbb dl, 7
cmovnb rax, rsi
cmp bl, dl
and rcx, 0b1111111111111 # instrumentation
cmovno si, word ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
xchg word ptr [r14 + rbx], bx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi
or bx, 0b1000000000000000 # instrumentation
bsf cx, bx
add bl, 120 # instrumentation
and rax, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rax], -91
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
