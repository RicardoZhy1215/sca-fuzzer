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
add cl, -51 # instrumentation
sbb dl, cl
add cl, bl
and rdx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rdx], esi
lea bx, qword ptr [rsi]
or dil, sil
and rax, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rax], 70
and rsi, 0b1111111111111 # instrumentation
movzx esi, byte ptr [r14 + rsi]
lea dx, qword ptr [rdx + rbx + 29790]
and rbx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rbx], bl
or rbx, 0b1000000000000000000000000000000 # instrumentation
bsr rax, rbx
and rbx, 0b1111111111111 # instrumentation
add dl, byte ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdi], rax
and rdx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rdx]
bt bx, 139
xor cl, dl
and rax, 0b1111111111111 # instrumentation
add byte ptr [r14 + rax], dl
and rdi, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rdi], eax
mul edi
and rcx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rcx]
add cl, 66 # instrumentation
and rdi, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rdi]
inc bx
and rcx, 0b1111111111111 # instrumentation
cmovz bx, word ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rsi], rdx
and rdx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdx]
sub dx, di
and rcx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rcx], eax
neg dx
xor al, dl
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], -2
sub dl, 2
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rcx], 5
test di, 28824
and rbx, 0b1111111111111 # instrumentation
sbb ebx, dword ptr [r14 + rbx]
lea edi, qword ptr [rax + rax + 58042]
setz bl
and rsi, 0b1111111111000 # instrumentation
and ecx, 0b111 # instrumentation
lock btc dword ptr [r14 + rsi], ecx
and rdi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdi]
setbe cl
lea esi, qword ptr [rsi + rcx + 60421]
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsf edi, ecx
test sil, 67
bt cx, bx
add dl, -91 # instrumentation
sets cl
xor dl, al
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], dl
and rcx, 0b1111111111111 # instrumentation
adc cl, byte ptr [r14 + rcx]
or ebx, 0b1000000000000000000000000000000 # instrumentation
bsr edx, ebx
add al, 55 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovnle bx, word ptr [r14 + rdi]
neg esi
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], cl
cbw
and rcx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rcx]
add dl, 8 # instrumentation
mov edi, 690299991
not cl
cmovz di, di
add sil, -32
movzx bx, dl
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rbx], rbx
adc rax, -1280989400
and rbx, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rbx]
mov ax, 9471
and rcx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rcx]
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
add cl, 72 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rax, 0b1111111111111 # instrumentation
cmovb rdx, qword ptr [r14 + rax]
add cl, -39
bts bx, 66
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rax], -68
and rcx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
movzx cx, byte ptr [r14 + rsi]
cmovo cx, si
and rsi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rsi]
bts rsi, rdx
xor al, dl
and rbx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rbx], 1
add ax, si
and rbx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rbx], -92
and rdi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rdi]
cmovnl rbx, rcx
setns dl
mul dil
add bl, 20 # instrumentation
setz dl
and rbx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rbx], -3
dec eax
mov rbx, rsi
and rsi, 0b1111111111111 # instrumentation
cmp bx, word ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
movsx ecx, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovnz ebx, dword ptr [r14 + rax]
test cl, dl
and rsi, 0b1111111111111 # instrumentation
cmovnb rdx, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 1 # instrumentation
and dx, word ptr [r14 + rdi] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rdi]
mul si
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
or esi, edx
setns dl
and rbx, 0b1111111111111 # instrumentation
movzx bx, byte ptr [r14 + rbx]
test rax, -800745221
xadd cl, dl
bts rcx, rsi
and rdi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdi], bl
and rbx, 0b1111111111111 # instrumentation
cmovnp ax, word ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
adc rdx, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 108
cmovbe ax, di
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], sil
movzx cx, cl
and rsi, 0b1111111111111 # instrumentation
cmovp ax, word ptr [r14 + rsi]
xor bl, -45
add al, bl
and rax, 0b1111111111111 # instrumentation
and word ptr [r14 + rax], si
and rsi, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rsi], 5
cmc
and rcx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rdi], -86
and ax, 12219
movsx rax, cx
cmovnz ax, si
cmovle rcx, rbx
and rdx, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rdx], 3
or rcx, rax
and rdi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdi], cl
and rbx, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rbx], bl
and rdx, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rdx], rdx
and rbx, 0b1111111111111 # instrumentation
cmovnb bx, word ptr [r14 + rbx]
and ecx, edx
cmovle eax, ebx
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
