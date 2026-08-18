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
add dl, -67 # instrumentation
sbb si, 12 
cmovz ecx, edi 
and rsi, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rsi] 
xadd al, bl 
neg eax 
seto bl 
and rcx, 0b1111111111111 # instrumentation
or edi, dword ptr [r14 + rcx] 
and rdx, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rdx], 4 
sbb rcx, rcx 
add dl, -119 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rsi] 
add dil, 6 
lea ax, qword ptr [rcx + rdi] 
and rsi, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rsi] 
add al, 24 
cmp dl, al 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rcx, 0b1111111111111 # instrumentation
and rbx, 0b111 # instrumentation
bt qword ptr [r14 + rcx], rbx 
movsx rdi, cx 
and rdi, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
bt word ptr [r14 + rdx], di 
and rsi, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rax], 4 
and rbx, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
movzx ax, byte ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rdi], 7 
and rcx, 0b1111111111111 # instrumentation
or sil, byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
xor bx, word ptr [r14 + rsi] 
or bx, -101 
and rdx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rdx], 2 
and rax, 0b1111111111111 # instrumentation
add word ptr [r14 + rax], -15 
and rsi, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rsi] 
lea di, qword ptr [rbx] 
test rax, rax 
and rbx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rbx], ecx 
and rcx, 0b1111111111111 # instrumentation
adc sil, byte ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdi], rsi 
cmovp si, cx 
test dx, 10328 
cmovle di, si 
sub dl, cl 
and rbx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rbx], 4 
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], 85 
and rsi, 0b1111111111111 # instrumentation
movzx si, byte ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
imul bx, word ptr [r14 + rdi], -15 
and rdi, 0b1111111111000 # instrumentation
and ecx, 0b111 # instrumentation
lock btc dword ptr [r14 + rdi], ecx 
sahf  
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
and rbx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rbx], rsi 
cmovs dx, bx 
add ax, cx 
cmovns rdi, rbx 
setnz cl 
lea eax, qword ptr [rbx + rbx + 54130] 
btc edi, 12 
mov ecx, -1472184163 
imul bl 
and rdi, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdi], 103 
and rdi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdi], al 
cmp bl, 121 
adc rax, 396847194 
and rcx, 92 
and rbx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rbx], dl 
lea rdx, qword ptr [rdx] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdx, 0b1111111111111 # instrumentation
btr ebx, esi 
xor ax, -23937 
mov al, dl 
and rdx, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rdx], 2 
and rdi, 0b1111111111000 # instrumentation
and ecx, 0b111 # instrumentation
lock btc dword ptr [r14 + rdi], ecx 
sub sil, -71 
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl 
btr eax, esi 
lea ebx, qword ptr [rax] 
and rcx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rcx], -69 
xor eax, 739916945 
xor dl, cl 
and rcx, 0b1111111111111 # instrumentation
sub dil, byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rsi], -1687818973 
setnz cl 
and rdx, 0b1111111111111 # instrumentation
movzx esi, word ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rbx], bx 
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], esi 
movsx bx, al 
cmp rsi, -53 
cmovbe rdx, rax 
and rcx, 0b1111111111111 # instrumentation
imul esi, dword ptr [r14 + rcx] 
imul ax, ax 
and rdi, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rdi], al 
cmovns rdi, rbx 
movzx rbx, dil 
or cl, cl 
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], 28 
setnz dil 
cmovno eax, edi 
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsf edx, ecx 
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
