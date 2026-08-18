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
xor bl, dl
xor ebx, 34
movsx esi, cl
mov rcx, rdx
cmovns rcx, rax
bts rbx, 115
and rsi, 0b1111111111111 # instrumentation
adc word ptr [r14 + rsi], ax
btr rcx, rcx
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rcx
movsx di, al
lea esi, qword ptr [rsi]
add eax, esi
sbb sil, sil
cmp ax, -14799
and rbx, 0b1111111111111 # instrumentation
cmovle ebx, dword ptr [r14 + rbx]
and cl, bl
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111111 # instrumentation
cmpxchg word ptr [r14 + rax], bx
cmovnz ax, di
xchg ax, ax
setns dil
test bx, 4218
and rbx, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rcx], bl
and rcx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rcx], 109
lea cx, qword ptr [rsi]
add edx, ebx
test al, al
test dx, 12149
adc cx, 85
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 1 # instrumentation
and dx, word ptr [r14 + rcx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rcx]
add al, -60 # instrumentation
lea rax, qword ptr [rbx + rax + 49897]
adc cl, 34
and rbx, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rbx]
sub esi, -109
imul rcx, rax
sub al, -114
and rsi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rsi], cl
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], 11
and rbx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rbx]
xor dl, al
and rax, 0b1111111111111 # instrumentation
cmovns bx, word ptr [r14 + rax]
xadd ebx, ebx
and rbx, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rbx]
xor bl, al
and eax, -1335026406
lea rdi, qword ptr [rcx + rdi]
and rsi, 0b1111111111111 # instrumentation
sub si, word ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
btr word ptr [r14 + rdx], di
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
xor edx, 110
or al, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv al
bt rdx, 151
and rax, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rax], al
cmpxchg sil, dl
btc edx, 63
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdi
and dl, bl
and rcx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rcx], 79
and rsi, 0b1111111111111 # instrumentation
cmovle ax, word ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mov esi, dword ptr [r14 + rax]
setnl dl
and rdi, 0b1111111111111 # instrumentation
adc cl, byte ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovl ax, word ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovns di, word ptr [r14 + rbx]
btc ax, 235
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, 98 # instrumentation
setbe dl
cmp al, -3
cmpxchg bl, cl
test al, 92
xor eax, -1990464281
test bl, dl
and rdi, 0b1111111111111 # instrumentation
movsx edx, word ptr [r14 + rdi]
cmovns cx, di
sub cl, 127
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
and rcx, 0b1111111111111 # instrumentation
add word ptr [r14 + rcx], si
bswap esi
sub ax, -32650
cmovnbe ecx, ecx
btr esi, ecx
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], eax
and rdx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rdx], dl
and rdx, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rdx]
bts dx, 234
or dl, 112
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rax
cmovs di, di
and rbx, 0b1111111111111 # instrumentation
cmovp ecx, dword ptr [r14 + rbx]
bt si, bx
and rax, 0b1111111111111 # instrumentation
cmovnz ax, word ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rax], cl
cmovnp rcx, rax
cmovl ebx, esi
xor cx, -128
and rax, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rax], esi
and rdx, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rdx]
lea rdi, qword ptr [rdx]
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
