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
sub al, al 
setnz al 
lea bx, qword ptr [rax + rcx + 23976] 
and rcx, 0b1111111111111 # instrumentation
adc ax, word ptr [r14 + rcx] 
sbb cl, dl 
and ax, 23332 
or dil, 1 # instrumentation
mov ax, 1 # instrumentation
div dil 
and rdx, 0b1111111111000 # instrumentation
and rdi, 0b111 # instrumentation
lock btr qword ptr [r14 + rdx], rdi 
adc al, -23 
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 1 # instrumentation
and rax, 0b1111111111111 # instrumentation
and qword ptr [r14 + rax], rdi 
lea ebx, qword ptr [rcx] 
and rdx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdx], 22 
xor dl, dl 
and rdx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rdx], cx 
and rdi, 0b1111111111111 # instrumentation
cmovno si, word ptr [r14 + rdi] 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
add bl, -93 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovo eax, dword ptr [r14 + rcx] 
test al, dil 
and rax, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
btc word ptr [r14 + rax], cx 
lea rcx, qword ptr [rdx] 
or esi, 0b1000000000000000000000000000000 # instrumentation
bsf edi, esi 
add al, -86 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovno ecx, dword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rbx], ax 
mov edi, 1947089793 
sub cl, 120 
mov sil, 0 
lea rcx, qword ptr [rcx] 
add al, 14 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb dl, byte ptr [r14 + rdx] 
sub rax, 838864166 
setnbe dl 
and rbx, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rbx] 
cmc  
and rsi, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rsi], ecx 
and rdx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rdx] 
btr edi, 147 
add cl, 76 # instrumentation
cmovp di, si 
and rsi, 0b1111111111111 # instrumentation
cmp esi, dword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rdx 
sub al, dl 
and rsi, 0b1111111111111 # instrumentation
cmovp rsi, qword ptr [r14 + rsi] 
or eax, 788721522 
xchg di, cx 
and rbx, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], dil 
and rax, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], 96 
and rsi, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rsi] 
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
lea di, qword ptr [rsi + rdi] 
dec bx 
lea ax, qword ptr [rax + rbx + 63169] 
and rsi, 0b1111111111111 # instrumentation
xor eax, dword ptr [r14 + rsi] 
movsx edi, dl 
cmovs rax, rdx 
and rdi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rdi], 127 
cmovnbe ebx, esi 
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl 
mul esi 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], 29 
clc  
and rdi, 0b1111111111111 # instrumentation
and rax, qword ptr [r14 + rdi] 
sub dl, cl 
test sil, -33 
cmovs rdi, rsi 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
bt dx, bx 
and rax, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rax], dil 
and rdx, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdx] 
xchg edi, eax 
and rcx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rcx], bx 
movzx si, cl 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 0b1000000000000000 # instrumentation
bsr ax, word ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmp bl, byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
and sil, byte ptr [r14 + rax] 
sbb al, -51 
and rdi, 0b1111111111111 # instrumentation
cmovz si, word ptr [r14 + rdi] 
and ax, -7514 
lea rbx, qword ptr [rbx] 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 66 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
add dl, 21 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi] 
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
