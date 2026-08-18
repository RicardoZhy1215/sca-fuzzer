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
lea rax, qword ptr [rdx] 
and rdi, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
btr word ptr [r14 + rdi], di 
btc ecx, 182 
sbb ax, 11369 
lea rsi, qword ptr [rdx] 
and rsi, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rsi], esi 
lea esi, qword ptr [rdi + rdx] 
lahf  
and rdx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdx], bl 
movsx ebx, al 
and rcx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
not byte ptr [r14 + rdi] 
mov edx, edx 
and rsi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rsi], 2 
adc al, -70 
and rbx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rbx], ax 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdx], 105 
lea ax, qword ptr [rcx] 
sub eax, -971085516 
lea rdi, qword ptr [rax + rdx + 51927] 
or esi, 1 # instrumentation
and edx, esi # instrumentation
shr edx, 1 # instrumentation
div esi 
and rsi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rsi], si 
and rax, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovb esi, dword ptr [r14 + rbx] 
neg eax 
lea ecx, qword ptr [rax + rax + 24096] 
neg rdx 
and rax, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], -83 
xor ebx, 77 
and rsi, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rsi] 
lea edx, qword ptr [rbx] 
xadd al, cl 
test al, 14 
and rcx, 0b1111111111111 # instrumentation
and cl, byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rsi] 
xor rsi, -121 
and rdx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rdx] 
add dl, -126 # instrumentation
cmovnbe ax, si 
bswap rsi 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rax], ax 
and sil, -125 
cmp rsi, -91 
mov al, dl 
and rax, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rax] 
inc dl 
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
and rdi, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdi], dil 
movzx edx, ax 
cmovz edi, ebx 
or cl, dl 
and rsi, 0b1111111111111 # instrumentation
cmovz rdi, qword ptr [r14 + rsi] 
cmovnp si, bx 
and rdx, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rdx], si 
cmp al, al 
movzx rax, dil 
cmovns si, bx 
and rsi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rsi], al 
and rdi, 0b1111111111111 # instrumentation
xor rsi, qword ptr [r14 + rdi] 
or edi, ecx 
or sil, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv sil 
add cl, -14 # instrumentation
lea si, qword ptr [rcx + rax + 1303] 
setnbe bl 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], -13 
btr rax, rbx 
and rsi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
imul rcx, qword ptr [r14 + rcx], 33 
and rcx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rsi], bl 
and rdi, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
add dx, word ptr [r14 + rcx] 
cmp rax, -448854046 
or edx, 0b1000000000000000000000000000000 # instrumentation
bsf ebx, edx 
xchg edi, eax 
and rax, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rax], ebx 
and rax, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rax], dl 
add rdi, rdi 
cdq  
and rax, 0b1111111111111 # instrumentation
cmovo rcx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
and eax, 0b111 # instrumentation
bts dword ptr [r14 + rdx], eax 
and rcx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rcx], al 
bts cx, 189 
add al, -118 # instrumentation
setl bl 
cmovnb ecx, edx 
adc dl, -108 
and rdx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdx], -97 
and bx, -4 
cmovnl dx, di 
adc cl, bl 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rsi] 
add al, 37 # instrumentation
sbb cl, bl 
and rax, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rax], 2 
dec bl 
and rdi, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
and cl, byte ptr [r14 + rsi] 
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
