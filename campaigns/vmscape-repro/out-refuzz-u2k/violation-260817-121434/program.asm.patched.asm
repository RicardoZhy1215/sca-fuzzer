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
.patch_placeholder:
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
add dl, -2 # instrumentation
lea ecx, qword ptr [rdi + rcx + 23504]
adc dil, -15
setns cl
and rcx, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rcx], 5
dec cx
imul esi
and rbx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
and rax, 0b111 # instrumentation
lock btr qword ptr [r14 + rbx], rax
and rbx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movsx edi, byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rax]
add eax, 743535278
cmpxchg al, cl
and rcx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rcx], 0
and rsi, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rsi], -97
imul rcx
add al, 52 # instrumentation
cmovnbe eax, edi
lea dx, qword ptr [rcx]
and rbx, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rbx]
lea rbx, qword ptr [rdi + rdi + 32801]
and rbx, 0b1111111111111 # instrumentation
and rdx, 0b111 # instrumentation
btc qword ptr [r14 + rbx], rdx
and rax, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rax], dl
and rcx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
and edx, dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
movzx rsi, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rax]
neg rbx
cmpxchg dl, cl
and rdx, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rbx], 35
and rcx, 0b1111111111111 # instrumentation
cmovnle esi, dword ptr [r14 + rcx]
sahf
and rdx, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bts word ptr [r14 + rdx], si
lea edx, qword ptr [rdx]
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr rdi, qword ptr [r14 + rbx]
add al, -66 # instrumentation
sbb bx, si
add rax, -554362114
cmp bl, -41
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], cl
and rdx, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rdx], -75
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rcx]
lea rsi, qword ptr [rsi]
xor dl, -58
sbb ecx, 42
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], dl
bts ebx, ebx
cmovnb rcx, rdx
and rdx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rdx]
add al, 13 # instrumentation
cmovnl di, bx
cmpxchg si, cx
and rax, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovnbe rdi, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], 102
lea rbx, qword ptr [rdi]
add rax, 19
adc edx, -110
and rbx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rsi]
and rax, 996348955
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
or ebx, ebx
and cl, cl
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence
.exit_vmvictim_0:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
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
or ax, dx
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rbx], edi
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rsi], cl
and rax, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rax], -60
sub dl, dl
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rbx
and rsi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rsi], sil
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], 112
cmp cl, 69
or al, 47
add dx, bx
and rax, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rax], 1
and rax, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rax], dil
and rdx, 0b1111111111000 # instrumentation
and ax, 0b111 # instrumentation
lock btr word ptr [r14 + rdx], ax
and rdx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rdx], 116
and rax, 0b1111111111111 # instrumentation
sbb dl, byte ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], 18
xor rsi, -63
and rdi, 0b1111111111111 # instrumentation
cmovo ecx, dword ptr [r14 + rdi]
xadd bl, al
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], rcx
and rdx, 0b1111111111111 # instrumentation
and rsi, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmovo cx, word ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
adc word ptr [r14 + rdi], si
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rbx], al
and rbx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rbx], bl
and rdx, 0b1111111111111 # instrumentation
or ebx, dword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rsi], 480105509
and rax, 0b1111111111111 # instrumentation
add byte ptr [r14 + rax], dl
and rdx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdx], dil
and rbx, 0b1111111111111 # instrumentation
btc word ptr [r14 + rbx], 7
add cl, -37 # instrumentation
and rsi, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmovnle edx, dword ptr [r14 + rdx]
stc
lea ebx, qword ptr [rdi]
neg bl
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], -116
xchg di, ax
cbw
and rsi, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rcx], -97
and rsi, 0b1111111111111 # instrumentation
add bl, byte ptr [r14 + rsi]
adc eax, 1711446141
add dil, bl
sub bl, bl
and rax, 0b1111111111111 # instrumentation
cmovnp di, word ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
sbb ax, word ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rcx], rdx
and rax, 0b1111111111111 # instrumentation
sub cx, word ptr [r14 + rax]
movzx rsi, dl
and rax, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
and edi, 0b111 # instrumentation
lock bts dword ptr [r14 + rcx], edi
and rsi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rsi], 1920984951
add dl, 27
and rsi, 0b1111111111111 # instrumentation
movzx bx, byte ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rax], eax
and rcx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rcx], 55
and rsi, 0b1111111111111 # instrumentation
cmovp edi, dword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rax]
test eax, -994864226
cmovnb edi, esi
lea rcx, qword ptr [rdx]
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence
.exit_vm_1:
.function_vm_2:
.bb_vm_2.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence
.exit_vm_2:
.section .data.main
.test_case_exit:nop
