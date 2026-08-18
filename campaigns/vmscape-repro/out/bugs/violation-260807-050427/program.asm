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
lea rdx, qword ptr [rcx] 
and rax, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rax], ebx 
movzx rbx, di 
or sil, 1 # instrumentation
mov ax, 1 # instrumentation
div sil 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rax 
test edx, -1063600109 
and rdx, 0b1111111111111 # instrumentation
cmovle dx, word ptr [r14 + rdx] 
lea rcx, qword ptr [rbx] 
nop edi 
cdq  
and rax, 0b1111111111111 # instrumentation
sbb rcx, qword ptr [r14 + rax] 
cmp cl, cl 
lea ax, qword ptr [rbx] 
imul cx, ax 
btc rdx, 39 
bt rbx, 148 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rcx, 0b1111111111111 # instrumentation
mov edi, dword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
movzx dx, byte ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rbx], al 
and rax, 0b1111111111111 # instrumentation
sub bx, word ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rbx], 106 
and rcx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rcx], 50 
neg di 
and rsi, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rsi], si 
lea eax, qword ptr [rax] 
and rdi, 0b1111111111111 # instrumentation
cmovs bx, word ptr [r14 + rdi] 
and rbx, rdi 
add rax, 121877073 
test ecx, 813938196 
and rdi, 0b1111111111111 # instrumentation
xor edi, dword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rsi], rdx 
and rdx, 0b1111111111111 # instrumentation
add esi, dword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rdx] 
or dil, 1 # instrumentation
mov ax, 1 # instrumentation
div dil 
add bl, 96 
setnle cl 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000000000000000 # instrumentation
bsr si, word ptr [r14 + rax] 
add cl, 98 # instrumentation
adc edx, 6 
and rsi, 0b1111111111111 # instrumentation
cmovs di, word ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
bt word ptr [r14 + rbx], di 
neg cl 
adc dl, bl 
and rcx, 0b1111111111111 # instrumentation
test word ptr [r14 + rcx], -17024 
and rsi, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rsi], -19 
and rbx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rbx], si 
sub eax, 1815547082 
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx] 
cdq  
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
sub dil, -18 
or cx, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv cx 
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 106 
mov al, -87 
and rbx, 0b1111111111111 # instrumentation
bts word ptr [r14 + rbx], 1 
bt rdi, rdx 
and rsi, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rsi], esi 
and rsi, 0b1111111111111 # instrumentation
and dword ptr [r14 + rsi], ecx 
sub dl, bl 
and dil, bl 
lea edi, qword ptr [rbx] 
imul rdi, rax 
and rcx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rcx], -109 
mov dx, dx 
cmovp rdi, rax 
adc dx, 110 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add bl, -122 # instrumentation
lea rdi, qword ptr [rbx + rsi] 
and rdi, 0b1111111111111 # instrumentation
cmovno dx, word ptr [r14 + rdi] 
cmp cl, -4 
and rax, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rax], rbx 
and rsi, 0b1111111111111 # instrumentation
cmovle ax, word ptr [r14 + rsi] 
sbb bl, dl 
and rsi, 0b1111111111000 # instrumentation
and si, 0b111 # instrumentation
lock btr word ptr [r14 + rsi], si 
and rsi, 0b1111111111111 # instrumentation
cmovnb si, word ptr [r14 + rsi] 
cmovnb rbx, rcx 
and rcx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rcx], 119 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rax 
and rcx, 0b1111111111111 # instrumentation
dec word ptr [r14 + rcx] 
lea edi, qword ptr [rsi + rcx + 42909] 
cmovz rsi, rbx 
bt di, 93 
adc rax, 1054527873 
mul dl 
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], cl 
lea rdx, qword ptr [rdx] 
xor dl, cl 
and dl, -12 
and rax, 0b1111111111111 # instrumentation
add word ptr [r14 + rax], -92 
and rsi, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rcx], -40 
and rax, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rax] 
not al 
and rdi, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rdi] 
lea bx, qword ptr [rcx + rdi + 12322] 
cmp dl, cl 
sub eax, 1798446948 
and rax, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rax], 23 
test rcx, 1351349075 
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
