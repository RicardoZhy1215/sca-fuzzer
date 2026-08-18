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
or sil, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv sil
add al, -104 # instrumentation
not dil
and rdi, 0b1111111111111 # instrumentation
cmovz edi, dword ptr [r14 + rdi]
lea ax, qword ptr [rax + rcx + 18336]
or al, bl
cmp dl, 42
and rax, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rax], al
cmovb ebx, edi
imul rdi, rax, -20
sbb al, al
and rdx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdx], -71
and rbx, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdi], cl
lea rcx, qword ptr [rdi]
and rbx, 0b1111111111111 # instrumentation
sub ecx, dword ptr [r14 + rbx]
or esi, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv esi
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add al, -26 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovb rsi, qword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rdi], cx
lea esi, qword ptr [rdi + rdi + 13115]
and rdx, 0b1111111111111 # instrumentation
cmovno edx, dword ptr [r14 + rdx]
btc rdi, rdx
lea edi, qword ptr [rcx]
and ebx, edx
dec bl
and rdi, 0b1111111111111 # instrumentation
and rbx, 0b1111111111111 # instrumentation
and rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdi], dl
and esi, esi
and rdi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdi], dl
btc ecx, edx
add al, 85 # instrumentation
setp bl
test dl, dl
and rcx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rcx], cl
and rdi, 0b1111111111111 # instrumentation
movsx rax, word ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rbx], 3
cmp eax, 1441844743
test cl, 118
cmovns edi, ecx
or al, bl
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
cmpxchg al, dil
cmovnp esi, edx
and rdx, 0b1111111111111 # instrumentation
mov dx, word ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rdi]
movzx cx, cl
and rcx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rcx], -83
cmovnp rdi, rdi
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
add cl, 36 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
lea ax, qword ptr [rdi + rdx + 44229]
cmovnb ax, dx
sbb bl, 63
movzx rsi, si
xadd bl, cl
bts di, 174
xor rax, -1118836607
mov rcx, -4856313997268850596
and rbx, 0b1111111111111 # instrumentation
cmovnb rcx, qword ptr [r14 + rbx]
lea ax, qword ptr [rdi + rdi + 42074]
and rbx, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx]
sets al
imul esi, ebx
and rax, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rcx]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
mov cl, dl
and rax, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bt word ptr [r14 + rax], si
or di, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv di
add dl, -45 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovp cx, word ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 1 # instrumentation
and dx, word ptr [r14 + rbx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rbx]
nop cx
and rbx, 0b1111111111111 # instrumentation
movsx rdi, word ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rsi], -116
and rax, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
movzx bx, byte ptr [r14 + rbx]
sub sil, 54
and rcx, 0b1111111111111 # instrumentation
or dl, byte ptr [r14 + rcx]
xchg ecx, eax
and rdx, 0b1111111111111 # instrumentation
sub word ptr [r14 + rdx], cx
test al, 104
lea rbx, qword ptr [rbx]
cmovz ebx, esi
test al, 22
cmovnl rbx, rax
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rsi], bl
sub bx, -65
and rdi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdi], 13
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], edi
sbb eax, -1027686649
and sil, al
cmovb cx, bx
or rax, rsi
and rdx, 0b1111111111111 # instrumentation
not word ptr [r14 + rdx]
adc ax, dx
and rsi, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rsi]
lea ebx, qword ptr [rax + rdi + 38914]
and rax, 0b1111111111111 # instrumentation
mov word ptr [r14 + rax], dx
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
