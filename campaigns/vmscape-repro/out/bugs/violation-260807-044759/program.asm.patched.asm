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
movzx ax, dl
and rbx, 0b1111111111111 # instrumentation
movzx ecx, word ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rbx], dl
sub rcx, 51
or di, 0b1000000000000000 # instrumentation
bsr ax, di
btc ecx, 53
add al, -72 # instrumentation
cmovns rdx, rdx
cmovnle esi, ecx
cmovle ecx, eax
lea rbx, qword ptr [rbx + rdi]
and rdx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rdx], 2
add dl, -113 # instrumentation
cmovp eax, esi
seto dl
and rdi, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
movzx esi, word ptr [r14 + rax]
test dl, bl
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rsi, 0b1111111111111 # instrumentation
and bl, byte ptr [r14 + rsi]
xchg bl, bl
and rcx, 0b1111111111111 # instrumentation
cmovnb si, word ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], bl
setnle cl
cmovo eax, esi
sbb cl, -120
add edx, eax
and rdx, 0b1111111111111 # instrumentation
cmovns bx, word ptr [r14 + rdx]
mov cl, dl
lea esi, qword ptr [rdx + rsi]
test al, -50
and rdi, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rcx], -1428065643
cmp ax, 19644
cmp rax, rcx
imul bl
btr rsi, 244
sbb bl, -121
inc dil
add cx, -27
lea di, qword ptr [rdx + rax]
and rax, 0b1111111111111 # instrumentation
add edi, dword ptr [r14 + rax]
cmovz esi, edi
and rax, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rax], cl
and rdi, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
cmovb ecx, dword ptr [r14 + rdi]
std
cmovno rcx, rdi
and rbx, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rbx], 112
and rsi, 0b1111111111111 # instrumentation
and ecx, 0b111 # instrumentation
bts dword ptr [r14 + rsi], ecx
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rax
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
add cl, 111 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
adc cl, bl
sub dil, 35
setp dl
and rdi, 0b1111111111111 # instrumentation
add bl, 116 # instrumentation
cmovz esi, edi
setnbe dil
and rax, 0b1111111111111 # instrumentation
adc word ptr [r14 + rax], -25
setz al
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], -96
and rax, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rax], rdx
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], si
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsr rdx, qword ptr [r14 + rax]
add cl, -41 # instrumentation
not cl
setp bl
cmovz rdi, rcx
cmovbe rsi, rsi
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
or dl, -128
and rdx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdx], dil
cmovnbe rsi, rbx
and rsi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rsi], sil
adc eax, -1652436500
and rsi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rsi]
sbb al, 79
and rcx, 0b1111111111111 # instrumentation
cmovp rdi, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rcx], 17
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rcx
lea rcx, qword ptr [rax]
lea edi, qword ptr [rdx + rax + 51293]
cbw
and rsi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rcx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx
inc ebx
and rbx, 0b1111111111111 # instrumentation
cmovl esi, dword ptr [r14 + rbx]
lea rbx, qword ptr [rcx + rdx + 32468]
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], -60
and rax, 0b1111111111111 # instrumentation
cmovp rbx, qword ptr [r14 + rax]
mov cl, al
mov dl, -110
cmovbe ecx, edx
setnz dl
and rdx, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rcx], bl
and rdx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdx], sil
sub rsi, -76
and rsi, 0b1111111111111 # instrumentation
and rax, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rax]
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
