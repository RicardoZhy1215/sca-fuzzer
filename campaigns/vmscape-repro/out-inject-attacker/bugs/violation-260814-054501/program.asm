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
add al, 53 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rbx, 0b1111111111111 # instrumentation
cmovnbe rbx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
adc rcx, qword ptr [r14 + rdx] 
bt ecx, esi 
movsx rdx, dl 
and rdi, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rdi], -59 
cmp bl, bl 
setbe bl 
imul rdx, rdx, -50 
xchg bl, sil 
add rdx, rax 
cmovnbe rbx, rbx 
sbb al, cl 
test bl, -121 
and rcx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rcx], rdx 
or edx, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
add dl, 32 # instrumentation
cmovnle cx, bx 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
and rsi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rsi], -35 
adc bl, dl 
adc bl, cl 
and rax, 0b1111111111111 # instrumentation
sub cx, word ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rdx], bl 
xor cl, cl 
cmovbe edx, ecx 
setb sil 
inc dil 
xchg dx, cx 
and rbx, rdi 
and rdi, 0b1111111111111 # instrumentation
or di, word ptr [r14 + rdi] 
sub eax, 1022832489 
and eax, -79417105 
or eax, -1055594916 
and rdi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdi], 57 
cmovle edi, ebx 
xadd dl, bl 
sbb dil, -100 
and rcx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 1 # instrumentation
and edx, dword ptr [r14 + rax] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rax] 
add bl, -48 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], -73 
adc cx, 23 
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], 96 
and rdx, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rdx], ecx 
and rsi, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rsi], -54 
cmovs eax, edx 
sub bl, al 
cmpxchg cl, bl 
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
and rsi, 0b1111111111111 # instrumentation
movsx cx, byte ptr [r14 + rsi] 
cmp bl, sil 
lea edi, qword ptr [rcx] 
and rcx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rcx], dil 
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 1 # instrumentation
and dx, word ptr [r14 + rcx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
sub si, word ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
movsx rbx, word ptr [r14 + rsi] 
and cl, bl 
adc bl, al 
neg bl 
setns bl 
and rcx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rcx] 
xor rax, -70 
bts edx, eax 
and rsi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rsi], cl 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
add bl, 26 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovb rcx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rbx], -11 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdx 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], -97 
adc eax, -1063504774 
bt rsi, rax 
and cl, bl 
and rcx, 0b1111111111111 # instrumentation
bts word ptr [r14 + rcx], 4 
lea cx, qword ptr [rdx] 
or rax, 0b1000000000000000000000000000000 # instrumentation
bsf rdx, rax 
sub cl, bl 
test ax, 712 
and rdi, 0b1111111111111 # instrumentation
movzx rcx, word ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdx], 96 
cmpxchg al, bl 
and rdx, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rdx], di 
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
