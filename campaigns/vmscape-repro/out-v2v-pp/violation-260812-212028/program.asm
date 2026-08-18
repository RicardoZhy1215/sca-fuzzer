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
.macro.switch_h2g.vmvictim.1: nop qword ptr [rax + 0xff]
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]
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
cmp cl, bl 
neg rax 
cmovle rdx, rcx 
sub rbx, rsi 
and rsi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rsi], -6 
bts eax, 54 
add al, -103 # instrumentation
mov ax, -503 
cmovnb rsi, rax 
setb bl 
cmovp rcx, rbx 
or ax, 817 
xor ebx, edi 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rsi], bl 
and rax, 0b1111111111111 # instrumentation
sub word ptr [r14 + rax], -42 
test al, -84 
lea rbx, qword ptr [rip +.function_vmvictim_safe] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_disclosure:
.bb_vmvictim_disclosure.entry:
add al, -97 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovno ax, word ptr [r14 + rdx] 
lahf  
or eax, 1 # instrumentation
and edx, eax # instrumentation
shr edx, 1 # instrumentation
div eax 
add ax, -27193 
and rax, 0b1111111111111 # instrumentation
sub rbx, qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
xor cl, byte ptr [r14 + rax] 
and rcx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rcx], rsi 
and rax, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rax], eax 
add cl, 77 
or ebx, ecx 
cmp cl, -101 
and rcx, 0b1111111111111 # instrumentation
or ax, word ptr [r14 + rcx] 
adc dx, 43 
xor al, bl 
dec al 
and rcx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rcx], bx 
cmp rsi, 123 
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl 
add cl, -58 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovo esi, dword ptr [r14 + rax] 
adc dil, 45 
and rdi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdi], -811441482 
movzx ax, dl 
add ax, 20961 
mov bl, bl 
neg cl 
or cl, dl 
imul edx 
and rdi, 0b1111111111111 # instrumentation
xor edx, dword ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rbx], 106 
and rdi, 0b1111111111111 # instrumentation
xor ecx, dword ptr [r14 + rdi] 
sbb al, 124 
sbb dl, bl 
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence  
.exit_vmvictim_disclosure:
.function_vmvictim_safe:
.bb_vmvictim_safe.entry:
lfence  # noremove
.macro.switch_g2h.main.vmvictim_1: nop qword ptr [rax + 0xff]
lfence  
.exit_vmvictim_safe:
.section .data.vm
.function_vm_train:
.bb_vm_train.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
add bl, 43 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setnb al 
and rcx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rcx], dl 
bt si, 172 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rax], cx 
add bl, 23 
lea ebx, qword ptr [rcx] 
and rax, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rax], rsi 
lea di, qword ptr [rbx + rdi + 41640] 
test rax, -1407677682 
lea ax, qword ptr [rcx + rdi + 8473] 
lea ecx, qword ptr [rbx] 
or rax, 0b1000000000000000000000000000000 # instrumentation
bsr rsi, rax 
mul cl 
adc rax, -1977127502 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rdx 
and rdx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdx], bl 
lea rax, qword ptr [rip +.function_vm_disclosure] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_disclosure:
.bb_vm_disclosure.entry:
lfence  # noremove
.macro.switch_g2h.main.vm_1: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_disclosure:
.function_vm_safe:
.bb_vm_safe.entry:
lfence  # noremove
.macro.switch_g2h.main.vm_safe: nop qword ptr [rax + 0xff]
lfence  # noremove
.exit_vm_safe:
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_0:
.function_vm_measure:
.bb_vm_measure.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_measure:
.section .data.main
.test_case_exit:nop
