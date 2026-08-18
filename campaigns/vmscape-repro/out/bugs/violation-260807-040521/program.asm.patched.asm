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
lea rcx, qword ptr [rbx]
and rsi, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rsi], -56
xadd al, al
mov rax, rdi
mov esi, -1125408173
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi]
or cx, 1 # instrumentation
and dx, cx # instrumentation
shr dx, 1 # instrumentation
div cx
or ax, si
and rsi, 0b1111111111111 # instrumentation
cmovle cx, word ptr [r14 + rsi]
cmpxchg dil, al
adc rax, -1441564894
inc bl
lea edx, qword ptr [rbx + rcx]
xor eax, 56326389
and edx, edx
xor dl, bl
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, -59 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovns ax, word ptr [r14 + rcx]
xadd al, bl
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rcx
and rbx, 0b1111111111111 # instrumentation
and cx, word ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
test dword ptr [r14 + rax], edi
and rbx, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rsi], cl
mov dx, dx
movzx rsi, cx
and rdi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdi], dl
xadd sil, bl
and rdi, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rbx], 5
add rax, -2013532056
cmp al, dl
and rcx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdx], ebx
cmovnle ebx, ecx
movzx esi, ax
and rax, 0b1111111111111 # instrumentation
imul di, word ptr [r14 + rax], 25
and rsi, 0b1111111111111 # instrumentation
sbb bl, byte ptr [r14 + rsi]
cdq
btr rcx, rdx
and al, -38
and rdi, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rdi], 27
or ebx, edi
and rbx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rbx], dl
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], al
and rbx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rbx], esi
bts ecx, edi
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
add bl, 13 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovz si, dx
cmovns rsi, rcx
cmpxchg esi, edx
lea rax, qword ptr [rsi + rsi + 23886]
cmovz edx, eax
mov esi, edx
setp dl
mov dl, dl
and rbx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rbx], 84
cmovnp rbx, rbx
lea rcx, qword ptr [rdx + rax]
inc dil
and rbx, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovl ebx, dword ptr [r14 + rsi]
sbb ax, -1333
and rax, 0b1111111111111 # instrumentation
xor edi, dword ptr [r14 + rax]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
sub sil, -17
and rdi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rcx], -127
and rcx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rcx], dx
or rax, rdi
or cl, al
and rdx, 0b1111111111111 # instrumentation
cmovnl di, word ptr [r14 + rdx]
btr rdi, rdi
and rbx, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mov bx, word ptr [r14 + rsi]
adc dl, -120
and rax, 0b1111111111111 # instrumentation
cmovbe edi, dword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rsi]
mov cl, dil
and rbx, 0b1111111111111 # instrumentation
or cl, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovle rdi, qword ptr [r14 + rbx]
cmovo dx, ax
add al, -102
mov cl, bl
and rdx, 0b1111111111111 # instrumentation
cmovo ax, word ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rdi], edi
and rcx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rcx]
cmovo bx, cx
add dl, 107 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovns edi, dword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
and rbx, 0b111 # instrumentation
lock bts qword ptr [r14 + rsi], rbx
or ebx, 0b1000000000000000000000000000000 # instrumentation
bsf eax, ebx
btr ecx, edi
add bl, -127 # instrumentation
mov cl, bl
and rdi, 0b1111111111111 # instrumentation
cmovo esi, dword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
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
