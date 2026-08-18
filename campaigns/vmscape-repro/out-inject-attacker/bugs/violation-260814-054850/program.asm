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
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and si, -51 
add rbx, rsi 
cmovle cx, dx 
and rdi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rdi], 31 
lea rbx, qword ptr [rcx] 
lea di, qword ptr [rcx + rbx] 
and rax, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rax], di 
or dx, 0b1000000000000000 # instrumentation
bsr di, dx 
add al, 68 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdi], dil 
and rdx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdx] 
bts edi, ebx 
add bl, 17 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rdi] 
sub al, cl 
and rsi, 0b1111111111111 # instrumentation
cmovns esi, dword ptr [r14 + rsi] 
lea edx, qword ptr [rsi + rax] 
cmp ax, -15322 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
and rdx, 0b1111111111111 # instrumentation
sub rbx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rdx], dx 
lea rdi, qword ptr [rdi + rsi + 8888] 
and rdx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdx], ebx 
lea rbx, qword ptr [rbx + rsi + 51758] 
xor rax, -1892537795 
imul al 
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl 
xor eax, -1814043817 
adc al, 91 
test dil, 111 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], cl 
cmovnl rcx, rax 
and rdx, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rdx], -86 
and rax, 0b1111111111111 # instrumentation
neg word ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rsi], 42 
and rax, 0b1111111111111 # instrumentation
imul rdx, qword ptr [r14 + rax], -107 
movsx edx, dx 
adc eax, -900849001 
or dil, -15 
sub ebx, 77 
and rsi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rsi], sil 
and rsi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rsi], sil 
and rcx, 0b1111111111111 # instrumentation
or dl, byte ptr [r14 + rcx] 
sub dl, 79 
cmovnl ax, ax 
and rdx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdx], -77 
and rdi, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdi], rsi 
and rbx, 0b1111111111111 # instrumentation
test word ptr [r14 + rbx], 11971 
and rdx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdx], bl 
and rdx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdx], rdx 
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
cmp bl, -23 
movsx rdi, ax 
lea rsi, qword ptr [rbx + rsi] 
bt ecx, esi 
cmp dl, bl 
setnle cl 
cmovo rcx, rdi 
cmovnbe dx, cx 
and rdx, 39 
adc cx, -34 
and rbx, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rbx] 
or al, cl 
and rax, 0b1111111111111 # instrumentation
and byte ptr [r14 + rax], al 
sub eax, -898593597 
and rbx, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rbx] 
sbb rdi, -111 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
or al, -39 
cmp sil, bl 
movsx rsi, dil 
and rax, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rax], -45 
add cl, -59 # instrumentation
cmovnp di, ax 
btr si, 32 
and rdi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rdi] 
cmovbe eax, edx 
xchg cl, al 
and rcx, 0b1111111111111 # instrumentation
and rbx, 0b111 # instrumentation
bt qword ptr [r14 + rcx], rbx 
and rbx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rbx], -107 
cmovnle bx, dx 
and rcx, 0b1111111111000 # instrumentation
xchg word ptr [r14 + rcx], di 
not si 
and rcx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rcx], esi 
cmpxchg dl, dl 
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
