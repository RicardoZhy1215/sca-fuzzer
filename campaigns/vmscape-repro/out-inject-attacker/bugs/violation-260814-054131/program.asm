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
.macro.set_h2g_target.vm.function_vm_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]
.exit_main_1:
.function_main_2:
.bb_main_2.entry:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.switch_h2g.vmvictim.2: nop qword ptr [rax + 0xff]
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.3: nop qword ptr [rax + 0xff]
.exit_main_3:
.function_main_4:
.bb_main_4.entry:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop  
.exit_main_4:
.section .data.vmvictim
.function_vmvictim_1:
.bb_vmvictim_1.entry:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
add cl, 104 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rsi, 0b1111111111111 # instrumentation
movsx ebx, word ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
sbb rdi, qword ptr [r14 + rdi] 
lea dx, qword ptr [rbx + rdi] 
or rdi, rdx 
setle al 
bt si, bx 
cmovz rax, rdx 
neg rsi 
xchg edx, esi 
xor rdi, -69 
mov ebx, -1288590901 
mul edi 
xor cl, -11 
xor cl, bl 
and rbx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rbx], 517441349 
cmovb di, dx 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
add dl, -58 # instrumentation
adc bl, -117 
movzx edx, cl 
lea edx, qword ptr [rax + rbx] 
cmovz rdx, rcx 
lea eax, qword ptr [rdi] 
and rdx, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bt qword ptr [r14 + rdx], rsi 
and sil, 99 
btc si, 104 
add cl, 58 # instrumentation
and rax, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rax] 
lea ecx, qword ptr [rdi] 
lea ax, qword ptr [rcx + rbx] 
mov bl, -71 
lea si, qword ptr [rdi + rcx + 12251] 
and rdx, 0b1111111111111 # instrumentation
and di, word ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
bts word ptr [r14 + rcx], 5 
add dl, 109 # instrumentation
cmovnp bx, cx 
and rbx, 0b1111111111111 # instrumentation
cmovs ax, word ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], al 
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], -14 
add cx, dx 
test si, 764 
and rbx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rbx], ax 
and rsi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rsi] 
sbb cl, dil 
mul rdi 
xchg dx, dx 
and rsi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rsi], dl 
cmovl cx, ax 
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rdi] 
add bl, 53 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovnbe si, word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], dl 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence  
.exit_vmvictim_3:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_0:
.function_vm_train:
.bb_vm_train.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
add rax, -16878939 
mov al, al 
sub bl, -7 
movsx rsi, bl 
and rax, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rax], ebx 
dec dl 
cmovnz ax, si 
and bl, dl 
and rsi, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rsi], 3 
cmpxchg rdi, rsi 
or ax, 28220 
lea rbx, qword ptr [rcx + rsi + 44815] 
imul rcx, rax, 19 
or sil, 30 
setnb cl 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
sub dl, cl 
sbb eax, 1313990068 
sbb al, 48 
or cl, bl 
and rbx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add al, -45 # instrumentation
not al 
cmovle rcx, rdi 
movzx esi, al 
test cl, -45 
cmovnle rdx, rsi 
cdq  
and rax, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rbx], al 
bt bx, di 
lfence  # noremove
.macro.switch_g2h.main.vm_1: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_3:
.function_vm_measure:
.bb_vm_measure.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_measure:
.section .data.main
.test_case_exit:nop
