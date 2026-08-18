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
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
add dl, 117 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovns cx, word ptr [r14 + rdi] 
cmpxchg rbx, rdi 
test al, -24 
or dl, -21 
btr rdx, 106 
or rax, -452562515 
lea bx, qword ptr [rbx + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmp ebx, dword ptr [r14 + rbx] 
cmp cl, 96 
and rbx, 102 
test cl, 127 
cmovo cx, si 
and rcx, 0b1111111111111 # instrumentation
adc bx, word ptr [r14 + rcx] 
xor al, 90 
std  
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
add cl, cl 
and rsi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rsi], -73 
add ax, -1060 
and rbx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
movzx rsi, word ptr [r14 + rcx] 
xchg edx, edi 
and rdx, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rdx], 111 
and rcx, 0b1111111111111 # instrumentation
and rcx, qword ptr [r14 + rcx] 
or al, dl 
btc ebx, 153 
and rbx, 0b1111111111111 # instrumentation
and rdx, 0b111 # instrumentation
bts qword ptr [r14 + rbx], rdx 
and rax, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rax] 
imul ebx, edx, 7 
xor al, bl 
and rdi, 0b1111111111111 # instrumentation
cmpxchg word ptr [r14 + rdi], bx 
and rcx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rcx], edx 
and rbx, 0b1111111111111 # instrumentation
cmovz ecx, dword ptr [r14 + rbx] 
cmovl eax, ecx 
and rdi, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rax] 
xor eax, -412920981 
cmovs ax, ax 
mov esi, 2044508506 
lea ecx, qword ptr [rax + rcx] 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
add al, 92 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sbb eax, dword ptr [r14 + rsi] 
cmp al, 126 
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], -67 
and rsi, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
btr word ptr [r14 + rsi], di 
add dl, -2 # instrumentation
and rbx, 0b1111111111111 # instrumentation
movsx eax, byte ptr [r14 + rbx] 
cmovl ecx, eax 
setnb bl 
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
add dl, 59 # instrumentation
sbb rbx, rdi 
cwde  
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
and rsi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rsi], 5 
add cl, -98 # instrumentation
cmovl dx, di 
bt eax, ecx 
test sil, 111 
bts di, 131 
sbb rax, 1911805764 
adc rdx, -2 
cmovnbe di, di 
btr ecx, ecx 
test eax, -545258561 
mov sil, bl 
cmovno rbx, rsi 
btr bx, 188 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
add al, 32 # instrumentation
adc al, 30 
and rcx, 0b1111111111111 # instrumentation
cmovb ax, word ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rax] 
lea esi, qword ptr [rsi + rbx] 
add cl, -110 
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], -105 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 20 
movzx rsi, cx 
and rsi, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rsi], -113 
setnp al 
mov al, dl 
and rdi, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdi], 71 
and rax, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdi], dl 
and rcx, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rcx], -69 
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
