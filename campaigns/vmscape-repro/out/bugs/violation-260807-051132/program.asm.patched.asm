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
sub al, cl
lea dx, qword ptr [rax + rdi]
sub al, dl
and rdx, 0b1111111111111 # instrumentation
cmovs dx, word ptr [r14 + rdx]
cmovp rbx, rax
add cl, 112
or si, 1 # instrumentation
and dx, si # instrumentation
shr dx, 1 # instrumentation
div si
dec sil
lea bx, qword ptr [rcx]
cmovns bx, cx
add dl, dl
bts bx, 145
setnbe al
setnz cl
mov rdx, 582421462413393999
sub bx, ax
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, 119 # instrumentation
and rsi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rsi], bl
cmovbe cx, si
cmovz edx, edi
and rbx, 0b1111111111111 # instrumentation
or dx, word ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rsi], ebx
and rdx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdx], rcx
or bl, al
cmovns eax, ebx
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], cl
and rbx, 0b1111111111111 # instrumentation
cmovnp rbx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rax], 3
or bl, bl
btc ebx, ecx
and rsi, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rsi]
sbb cl, dl
cmp ax, -6812
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], -49
adc di, di
xadd cl, dil
cmp sil, sil
add rcx, rax
and rax, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rax]
add cl, -36
adc cx, 82
lea rbx, qword ptr [rdx + rsi + 2480]
and rax, 0b1111111111111 # instrumentation
cmovnle dx, word ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rbx]
add ax, 23437
cmovp rdx, rax
lea ecx, qword ptr [rsi + rcx]
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rdi]
and bl, bl
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
add dl, 65 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rbx, 0b1111111111111 # instrumentation
cmovs cx, word ptr [r14 + rbx]
cmovnbe ebx, edx
and rcx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rcx]
movsx rbx, dx
sbb sil, -76
and rax, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rax]
adc cl, al
imul rax, rax, 66
xor sil, 126
neg rdi
bts rdx, 1
and rax, 0b1111111111000 # instrumentation
and esi, 0b111 # instrumentation
lock btc dword ptr [r14 + rax], esi
lea rsi, qword ptr [rcx + rbx + 10806]
not rax
and rbx, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rbx]
lea cx, qword ptr [rbx]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, 80 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovo rdx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx
xor cl, dl
and rsi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rsi], -104
and rsi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rsi], cl
lea esi, qword ptr [rax]
or bx, 0b1000000000000000 # instrumentation
bsr bx, bx
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rdx
and rax, 0b1111111111000 # instrumentation
xchg word ptr [r14 + rax], ax
and rbx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rbx], -87525778
bt rax, rcx
and rax, 0b1111111111111 # instrumentation
imul rsi, qword ptr [r14 + rax], 56
lea dx, qword ptr [rcx + rcx]
and rax, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rax]
sbb di, si
xadd sil, dl
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], -50
and rax, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rax], 72
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rax], dil
and rdi, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdi], 3
and rcx, 0b1111111111111 # instrumentation
imul ebx, dword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
btc word ptr [r14 + rsi], bx
and rsi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rsi], 54
cmp eax, 1924061944
and rdx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdx], rdi
mov cl, -123
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], -35
movzx edx, dx
std
and rdi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rdi], bl
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
