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
add al, 80 # instrumentation
cmovnle rax, rsi
and rdx, 0b1111111111111 # instrumentation
imul rdx, qword ptr [r14 + rdx], -26
add al, 17 # instrumentation
setbe al
and eax, 198862465
dec bl
or cl, -96
mov cl, al
bts di, si
add cl, bl
imul si
cmp bl, sil
lea rax, qword ptr [rax + rcx + 54532]
adc rax, 813120407
test cl, -83
and rcx, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
mov ecx, dword ptr [r14 + rbx]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
mov edx, ebx
and rcx, 0b1111111111111 # instrumentation
movsx si, byte ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rcx]
cmovnl di, cx
and rax, 0b1111111111111 # instrumentation
cmovnl ecx, dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
movzx rbx, word ptr [r14 + rbx]
movsx edi, sil
and rcx, 0b1111111111111 # instrumentation
movsx rbx, word ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rax], ecx
adc ax, bx
setl dil
cmovle rsi, rbx
and rax, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rax], edi
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], ecx
xchg dl, dl
and rdx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdx], -105
and rdi, 0b1111111111111 # instrumentation
cmovs dx, word ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rbx], 1
lea dx, qword ptr [rcx + rcx]
setnb sil
xor bl, cl
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
add cl, -109 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rdi], edi
and rdi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
and cx, 0b111 # instrumentation
lock btr word ptr [r14 + rbx], cx
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], al
seto dl
lea ebx, qword ptr [rdx + rdi + 52586]
and rdi, 0b1111111111111 # instrumentation
or di, word ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rdi], 7
and rcx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rcx], ebx
and rbx, 0b1111111111111 # instrumentation
adc dl, byte ptr [r14 + rbx]
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
test al, -115
sbb ecx, edx
sub al, -34
sbb dl, cl
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdi], dil
lea rsi, qword ptr [rcx + rbx]
and rsi, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rsi]
xadd si, bx
sub bl, cl
movsx ecx, bl
add si, 22
movzx di, dl
test dil, -74
lea ebx, qword ptr [rax + rcx + 57211]
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], 110
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add bl, 79 # instrumentation
mov eax, 1671810429
and rdi, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rbx]
cmovnb cx, bx
cmovb di, cx
bswap ecx
and rbx, 0b1111111111000 # instrumentation
and di, 0b111 # instrumentation
lock btr word ptr [r14 + rbx], di
and rax, 0b1111111111111 # instrumentation
movzx eax, byte ptr [r14 + rax]
and cl, dl
movsx ax, bl
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], eax
and rsi, 0b1111111111111 # instrumentation
cmovnb rdi, qword ptr [r14 + rsi]
bt rsi, rdi
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rsi], di
and rdx, 0b1111111111111 # instrumentation
sbb dl, byte ptr [r14 + rdx]
setnl dl
and rdx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdx], cl
and rax, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rax], 0
add bl, -88 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovo si, word ptr [r14 + rsi]
imul rdi, rbx, 96
imul rbx, rdx, 16
xor al, -40
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 0b1000000000000000 # instrumentation
bsf di, word ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
and rdi, qword ptr [r14 + rsi]
cmovs di, ax
and rdx, 0b1111111111000 # instrumentation
and ecx, 0b111 # instrumentation
lock bts dword ptr [r14 + rdx], ecx
xchg rax, rax
nop ebx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rax
and rcx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rcx], -104
xadd rsi, rdx
stc
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
