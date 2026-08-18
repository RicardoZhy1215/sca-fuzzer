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
add bl, -91 # instrumentation
lea ax, qword ptr [rax + rdi + 57169]
and rbx, 0b1111111111111 # instrumentation
not word ptr [r14 + rbx]
inc di
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmovnl ecx, dword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rcx], -35
and rax, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rax], ebx
not si
mov dl, bl
sub al, al
lea ax, qword ptr [rsi + rbx + 51020]
imul rsi, rsi
xor ax, -6485
sbb rsi, rcx
lea rax, qword ptr [rcx]
sub di, 74
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, -123 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovnb dx, word ptr [r14 + rsi]
xchg ebx, eax
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx]
btr eax, 211
btc eax, 127
and rax, 0b1111111111111 # instrumentation
test word ptr [r14 + rax], 24699
or bl, 0b1000 # instrumentation
and bl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv bl
add al, -97 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovb rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rax]
cmovno di, dx
and rdi, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdi], 45
and rdx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdx], cl
lea rdx, qword ptr [rbx + rdi + 30584]
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], al
bswap rsi
and rcx, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
add dl, -75
add rdi, 110
lea dx, qword ptr [rsi]
setnbe bl
xadd dl, dl
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], bl
and rax, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rax], rdx
cmovns bx, cx
and rbx, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rbx]
add dl, -107
cmpxchg al, cl
and rcx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rbx], edx
dec rdx
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
sub si, cx
stc
add al, -121
movsx edx, dil
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
sbb al, bl
cmovp ebx, eax
sbb cl, al
adc cx, 115
lea ax, qword ptr [rax]
cmovnz edx, ecx
bts ecx, eax
add cl, -84 # instrumentation
cmovnb rbx, rdi
cmovle edx, ebx
mov esi, esi
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add al, -41 # instrumentation
and rdi, 0b1111111111111 # instrumentation
sbb bl, byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
and ecx, 0b111 # instrumentation
bt dword ptr [r14 + rcx], ecx
and rdx, 0b1111111111111 # instrumentation
test word ptr [r14 + rdx], di
movsx ax, sil
add cl, dl
sbb bl, dl
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi
and rax, 0b1111111111111 # instrumentation
sbb cl, byte ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
mov edx, dword ptr [r14 + rax]
and bl, dl
and ebx, ecx
and rdi, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], sil
and rsi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rsi], dl
dec al
cmpxchg rax, rbx
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000000000000000 # instrumentation
bsf cx, word ptr [r14 + rcx]
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsr eax, ecx
add cl, -45 # instrumentation
mov dl, dl
and rcx, 0b1111111111111 # instrumentation
cmovp bx, word ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdi], dil
and bl, al
sbb cl, 97
and rdx, 0b1111111111111 # instrumentation
sub word ptr [r14 + rdx], di
btr rcx, rsi
and rcx, 0b1111111111111 # instrumentation
or rdx, qword ptr [r14 + rcx]
btc eax, 151
xadd dl, dil
and rcx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rcx], -44
and rax, 0b1111111111111 # instrumentation
and rdx, 0b111 # instrumentation
bts qword ptr [r14 + rax], rdx
and rbx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rbx]
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
