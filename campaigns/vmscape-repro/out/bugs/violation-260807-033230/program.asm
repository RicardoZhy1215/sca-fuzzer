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
or cl, -42 
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi] 
setle dl 
imul edx, ecx 
imul esi 
adc bx, 33 
cmovbe cx, cx 
btc rbx, rdi 
and rdi, 0b1111111111111 # instrumentation
imul rsi, qword ptr [r14 + rdi], 98 
sbb eax, 623374838 
and rsi, 0b1111111111111 # instrumentation
cmovl edx, dword ptr [r14 + rsi] 
dec rsi 
sbb al, cl 
setz dl 
and rdx, 0b1111111111111 # instrumentation
imul bx, word ptr [r14 + rdx], 56 
imul rdi, rax 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rax] 
cmovs edx, ecx 
sub eax, -1319115302 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], -10 
or cx, 1 # instrumentation
and dx, cx # instrumentation
shr dx, 1 # instrumentation
div cx 
add dl, 66 # instrumentation
cmovnbe rbx, rdx 
cld  
and rdx, 0b1111111111111 # instrumentation
movzx si, byte ptr [r14 + rdx] 
cmovbe ecx, edi 
and rdx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdx], bl 
cmp ax, 1949 
and rdx, 0b1111111111111 # instrumentation
adc dx, word ptr [r14 + rdx] 
cmp eax, -1017318123 
and rsi, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rsi], 73 
lea si, qword ptr [rbx + rcx] 
and rax, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rcx] 
btc bx, si 
and rsi, 0b1111111111111 # instrumentation
or ebx, dword ptr [r14 + rsi] 
add bl, cl 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], -79 
lea di, qword ptr [rsi + rdx] 
and rbx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rbx], rsi 
and rax, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rax] 
xchg bl, cl 
or ebx, 1 # instrumentation
and edx, ebx # instrumentation
shr edx, 1 # instrumentation
div ebx 
add al, 124 # instrumentation
cmc  
and rdi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdi], -66 
or edx, 1 # instrumentation
xor al, 27 
sbb rbx, rsi 
cmpxchg bl, bl 
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
add bl, 93 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
mov al, cl 
cmovz edx, edi 
and rbx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rbx], -2 
sub dil, -21 
or ax, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv ax 
cmp rax, -1133711120 
adc cl, 92 
cwd  
mul dl 
and rsi, 0b1111111111111 # instrumentation
cmovb rax, qword ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rsi], 58 
sbb al, dl 
and rcx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rcx], eax 
sbb dl, cl 
inc rdx 
and rax, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rax] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
sub bl, bl 
sub al, dl 
and rax, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rax], cl 
cmovs eax, esi 
cmovns edx, esi 
cmovnl ecx, edi 
xor al, dl 
cmovns edi, ebx 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rcx 
neg ecx 
cld  
lea ecx, qword ptr [rax + rax + 27257] 
cmovs ebx, eax 
and rax, 0b1111111111111 # instrumentation
cmp bl, byte ptr [r14 + rax] 
cmovl ax, bx 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], dl 
cmovnbe ax, bx 
and rbx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rbx], bx 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], bl 
cmovnbe esi, ebx 
and rbx, 0b1111111111111 # instrumentation
test word ptr [r14 + rbx], bx 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], al 
bts edi, edi 
mul cl 
lea rax, qword ptr [rsi + rsi] 
and rdi, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
imul bx 
add bl, 77 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnl eax, dword ptr [r14 + rdx] 
bts ebx, ebx 
add al, -71 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovnl edx, dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], -38 
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
