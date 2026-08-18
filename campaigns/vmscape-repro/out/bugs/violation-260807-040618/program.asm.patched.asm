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
or dx, 0b1000000000000000 # instrumentation
bsr di, dx
inc eax
cmovns bx, si
sub bl, bl
or cl, bl
lea rdx, qword ptr [rcx]
lea si, qword ptr [rbx + rdi + 18420]
seto cl
cmovnb cx, si
cmovnp dx, si
and rsi, 0b1111111111111 # instrumentation
and word ptr [r14 + rsi], -79
neg rdx
test bl, bl
lea rbx, qword ptr [rax]
and rdx, 0b1111111111111 # instrumentation
cmovnz ecx, dword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rax], -42
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, 22 # instrumentation
adc al, cl
and rdi, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rdi], ax
setnl dl
setnle cl
and rbx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rbx], edx
and rcx, 0b1111111111111 # instrumentation
sbb edx, dword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rax], 3
and rbx, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rcx
and rax, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
adc di, word ptr [r14 + rsi]
movzx si, al
mov bx, 8822
and rax, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rax]
cmp bl, bl
and rcx, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rcx], -124
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], -70
and rsi, 0b1111111111111 # instrumentation
mov si, word ptr [r14 + rsi]
not dl
mov dl, bl
and rbx, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
btr dword ptr [r14 + rbx], edi
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi]
xchg edx, edx
sbb bl, dl
cmovnl edx, eax
and rbx, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rbx]
inc ecx
and rcx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rcx], -85
btc edx, esi
and rdi, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rdi], edi
and rbx, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rbx], 0
cmp ax, 6258
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
add dl, -119 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
not edi
and rax, 0b1111111111111 # instrumentation
movzx ecx, byte ptr [r14 + rax]
adc si, di
lea rcx, qword ptr [rbx]
and rsi, 0b1111111111111 # instrumentation
add dx, word ptr [r14 + rsi]
add bx, 3
and rdi, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rdi], 7
cmovnb rdi, rdx
imul cx
and al, -2
mov cx, dx
and rdx, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rdx], 3
and rsi, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rsi], -70
and rbx, 0b1111111111000 # instrumentation
and eax, 0b111 # instrumentation
lock bts dword ptr [r14 + rbx], eax
and rbx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rbx]
sbb ax, 1641
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add al, 13 # instrumentation
setnl bl
test bl, dl
and rcx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rcx], 6
and rbx, 0b1111111111111 # instrumentation
or sil, byte ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rbx], 46
mov bl, bl
and rcx, 0b1111111111111 # instrumentation
cmovnb si, word ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rsi]
mov cl, 95
xadd bl, cl
adc sil, dil
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi]
lea ecx, qword ptr [rdi + rdi]
and rdi, 0b1111111111111 # instrumentation
cmovb rax, qword ptr [r14 + rdi]
xchg ax, ax
and rdi, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rdi], edx
movsx rcx, al
or esi, 0b1000000000000000000000000000000 # instrumentation
bsr edx, esi
add al, -50 # instrumentation
and rdx, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rcx], rsi
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000000000000000 # instrumentation
bsr dx, word ptr [r14 + rax]
add bl, 2 # instrumentation
lea rsi, qword ptr [rdx + rcx]
adc eax, -2030291863
and rax, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmovbe esi, dword ptr [r14 + rsi]
setnz cl
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], al
and rsi, 0b1111111111111 # instrumentation
sub esi, dword ptr [r14 + rsi]
cmovle ax, ax
cmpxchg dl, cl
and rcx, 0b1111111111111 # instrumentation
cmovbe eax, dword ptr [r14 + rcx]
neg rcx
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
