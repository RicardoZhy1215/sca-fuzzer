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
add dl, 86 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rbx, 0b1111111111111 # instrumentation
cmovle dx, word ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf rax, qword ptr [r14 + rax] 
add cl, 66 # instrumentation
setp bl 
or cx, -73 
bts eax, edx 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax 
not rax 
xadd rcx, rax 
lea edi, qword ptr [rax + rax + 53584] 
imul di, bx, -108 
or al, -100 
xchg ecx, eax 
and rbx, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rbx] 
cmovnle edx, eax 
and rbx, 0b1111111111111 # instrumentation
xor edi, dword ptr [r14 + rbx] 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
sub dx, 71 
and rcx, 0b1111111111111 # instrumentation
movzx edi, word ptr [r14 + rcx] 
and ax, -16367 
and rdx, 0b1111111111111 # instrumentation
cmovnle edx, dword ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rdx] 
cmovp si, cx 
btr eax, 86 
and rax, 0b1111111111111 # instrumentation
add cl, byte ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax 
sub al, -41 
setnp al 
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], 349875617 
not bl 
sub dl, cl 
cmovnb edi, eax 
nop esi 
or bl, 4 
test edx, 1515329057 
test rax, -677402369 
and al, -109 
and rcx, 0b1111111111111 # instrumentation
cmp si, word ptr [r14 + rcx] 
bts edi, 223 
lea rsi, qword ptr [rax + rdx + 17194] 
btc di, 202 
and rdi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rdi], dx 
test sil, -35 
add dl, dl 
xor dil, 17 
add dl, 116 # instrumentation
lahf  
bswap rcx 
and rdx, 0b1111111111111 # instrumentation
cmp cl, byte ptr [r14 + rdx] 
add ax, -17599 
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
lea rcx, qword ptr [rdi] 
and rax, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
btr dword ptr [r14 + rax], edx 
lea dx, qword ptr [rbx + rsi + 3127] 
lea rax, qword ptr [rdx + rcx] 
lea eax, qword ptr [rbx + rcx + 11618] 
and rcx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rcx], cl 
and rdx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdx], -24 
and rsi, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rsi], 37 
add bl, 15 # instrumentation
cmovns ax, si 
and rdi, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rdi], bx 
seto al 
and rbx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rbx], 2 
or dl, 76 
lea rsi, qword ptr [rsi + rdi + 37989] 
and rdi, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdi], 32 
mov bl, cl 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
add dl, 1 # instrumentation
cmovnbe rdi, rax 
lea rdx, qword ptr [rdx + rcx + 40573] 
or bl, al 
cmovb rdx, rcx 
cmovz si, dx 
mov bl, cl 
btr si, ax 
adc al, 82 
lea rbx, qword ptr [rdx + rdi] 
and rax, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rax] 
lea si, qword ptr [rcx + rbx + 47653] 
and rdx, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bts qword ptr [r14 + rdx], rsi 
xchg ax, dx 
sbb bl, al 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000000000000000 # instrumentation
bsr si, word ptr [r14 + rax] 
bts ebx, 192 
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
