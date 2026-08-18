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
and rcx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rcx], bl 
xchg dx, bx 
and rax, 0b1111111111111 # instrumentation
and rbx, qword ptr [r14 + rax] 
cwd  
lea si, qword ptr [rdi] 
and rdx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rdx], 20 
and rdx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rdx], ecx 
or dil, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv dil 
cmp al, -73 
mul al 
imul rsi, rsi, -20 
lea rbx, qword ptr [rax + rsi + 28876] 
and rax, 0b1111111111111 # instrumentation
bts word ptr [r14 + rax], 3 
mov cx, ax 
lea edx, qword ptr [rbx] 
lea si, qword ptr [rsi] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdi, 0b1111111111111 # instrumentation
cmp al, byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
dec word ptr [r14 + rbx] 
xor al, 91 
and rax, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rax], esi 
movzx eax, dil 
cmpxchg rdx, rsi 
and rax, 0b1111111111111 # instrumentation
cmp cx, word ptr [r14 + rax] 
add cl, 54 
xor dl, bl 
and rbx, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bts qword ptr [r14 + rbx], rsi 
and rdx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdx], dil 
cmovs rdx, rdi 
mov edx, eax 
and rbx, 0b1111111111111 # instrumentation
and ecx, dword ptr [r14 + rbx] 
lea bx, qword ptr [rdx + rcx + 9366] 
sub al, 66 
lea edi, qword ptr [rdi + rdi] 
bt rax, 189 
lea edx, qword ptr [rsi] 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 1 # instrumentation
and edx, dword ptr [r14 + rax] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rax] 
add al, 122 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnl ax, word ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
cmovnl edx, dword ptr [r14 + rsi] 
add dl, 35 
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], ax 
xor al, cl 
lea dx, qword ptr [rbx] 
lea cx, qword ptr [rdx] 
bt rax, 201 
and rdi, 0b1111111111111 # instrumentation
sub ecx, dword ptr [r14 + rdi] 
xchg di, ax 
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rdx 
and rdi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdi] 
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
and rsi, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rsi], dl 
cmovnbe edi, esi 
and rbx, 0b1111111111111 # instrumentation
add sil, byte ptr [r14 + rbx] 
setnle al 
and rdx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdx], -68 
test rax, 171986363 
cmp dil, sil 
cmp al, -23 
cmpxchg ebx, eax 
test dx, si 
and rdx, 0b1111111111111 # instrumentation
adc bx, word ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], dl 
lea eax, qword ptr [rsi] 
stc  
setbe al 
and rdx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdx], 77 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rax, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rax] 
cmp al, dl 
lea dx, qword ptr [rbx + rdx + 25320] 
and rcx, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
bt word ptr [r14 + rcx], dx 
bts cx, 61 
and rsi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rsi], al 
dec rdi 
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], -87 
imul bx, cx, 102 
add al, -127 # instrumentation
movsx cx, al 
sets dl 
and rbx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rbx], -29 
and rbx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rbx], 333930432 
and rsi, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rsi], edx 
and rbx, 0b1111111111111 # instrumentation
sbb bl, byte ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdi], bx 
and rbx, 0b1111111111111 # instrumentation
cmovnle di, word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], cl 
add dl, 122 
cmc  
and rcx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rcx], cl 
and rax, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rax] 
lea si, qword ptr [rdx + rsi + 10147] 
add al, al 
dec bl 
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx] 
neg al 
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], 61 
and rdx, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
btc word ptr [r14 + rdx], si 
cmp al, cl 
cmovl ax, cx 
test rdx, rdx 
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
