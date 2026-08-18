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
add bl, 80 # instrumentation
adc ax, 13388 
and rcx, 0b1111111111111 # instrumentation
or rdx, qword ptr [r14 + rcx] 
cmovz ecx, ebx 
setnz cl 
and rdx, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
not qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rsi] 
cwd  
movsx rdx, bx 
add dl, al 
bt ax, dx 
mov bl, bl 
add cl, al 
cmovz rsi, rcx 
and rax, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rax] 
sbb cl, bl 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111111 # instrumentation
xor cx, word ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmovnle ax, word ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsr rdi, qword ptr [r14 + rax] 
add al, -87 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rcx], cl 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rbx 
and rdi, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rdi], 6 
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rdi] 
add al, 66 # instrumentation
cmovnl rax, rsi 
and cl, al 
lea ebx, qword ptr [rax + rdi + 41361] 
setb dl 
and rsi, 0b1111111111111 # instrumentation
or al, byte ptr [r14 + rsi] 
adc di, -30 
and rsi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rsi], bl 
add cl, dl 
cmp rcx, 110 
cmovl edi, ebx 
and rdi, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rdi], 44 
add dl, 86 # instrumentation
cmovnle ecx, eax 
bt ebx, edx 
not bl 
lea dx, qword ptr [rdx] 
and rax, 0b1111111111111 # instrumentation
adc sil, byte ptr [r14 + rax] 
sbb edx, edi 
and rax, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rax] 
sub edx, eax 
lea rax, qword ptr [rcx + rax] 
xchg ebx, edx 
and rax, 0b1111111111111 # instrumentation
cmovno si, word ptr [r14 + rax] 
cmpxchg bl, cl 
and rsi, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rsi] 
add al, -79 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovnbe cx, word ptr [r14 + rsi] 
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
add dl, 93 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
dec bl 
adc cl, bl 
and rdx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdx], -86 
lea ebx, qword ptr [rdi] 
and rsi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rsi], -8 
or bl, al 
cmp rcx, rdi 
cmovp ax, di 
add dil, dil 
cmovl esi, ecx 
and rcx, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rcx] 
lea cx, qword ptr [rsi + rax] 
xor rdx, rdi 
mul bl 
lea rax, qword ptr [rax] 
btr ebx, 85 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, 80 # instrumentation
and rdi, 0b1111111111111 # instrumentation
adc rsi, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rcx] 
sbb eax, -51831097 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rcx], dx 
setnl dil 
and rdx, 0b1111111111111 # instrumentation
xor rbx, qword ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rsi], dil 
and rdx, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov dl, byte ptr [r14 + rdi] 
lea eax, qword ptr [rcx + rbx + 8475] 
sub ax, 17046 
mov edx, ecx 
setl al 
and rdi, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdi], rdi 
and rdi, 0b1111111111111 # instrumentation
cmovo ax, word ptr [r14 + rdi] 
xchg bl, cl 
movzx di, bl 
and rsi, 0b1111111111111 # instrumentation
movzx rcx, word ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rax], ax 
lea rax, qword ptr [rdx] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rax 
and rbx, 0b1111111111111 # instrumentation
cmovnp edx, dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rdx] 
inc si 
xadd dl, dl 
and rsi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rsi], esi 
sub dl, bl 
and rsi, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rsi], ebx 
xadd dl, cl 
lea rbx, qword ptr [rsi] 
cmovnl eax, eax 
cmovnb edx, ecx 
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
