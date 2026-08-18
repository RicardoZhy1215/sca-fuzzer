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
add dl, -17 # instrumentation
lea si, qword ptr [rbx + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovnp rsi, qword ptr [r14 + rdx] 
cmp di, -83 
and rbx, 0b1111111111111 # instrumentation
cmovnb edi, dword ptr [r14 + rbx] 
sub sil, bl 
xor ax, -15474 
and rbx, 0b1111111111111 # instrumentation
cmovl dx, word ptr [r14 + rbx] 
sbb dx, -52 
adc dl, bl 
and rax, 0b1111111111111 # instrumentation
cmp esi, dword ptr [r14 + rax] 
movzx rax, cl 
and rcx, 0b1111111111111 # instrumentation
cmp dl, byte ptr [r14 + rcx] 
cmp ebx, 19 
mov di, bx 
adc si, di 
bt eax, edi 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
lea bx, qword ptr [rax + rsi] 
bt ax, 14 
imul rax, rdx, 47 
cmpxchg rbx, rbx 
add al, bl 
or bl, al 
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], -43 
and rdx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rdx], rdi 
cmovnbe si, di 
and rdi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdi], -107 
nop si 
inc rax 
and rsi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rsi], 75 
dec al 
and rcx, 0b1111111111111 # instrumentation
sbb bl, byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
btc cx, 234 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rcx] 
btr rcx, rbx 
and rdi, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rdi] 
and rdi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdi], cl 
and rcx, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
imul ebx, dword ptr [r14 + rsi] 
adc al, -116 
add edi, 81 
and rdx, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rdx], dx 
xadd dx, bx 
mov esi, ecx 
lea cx, qword ptr [rbx + rax] 
and rax, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rax] 
movsx si, bl 
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
xadd rcx, rdx 
or ax, 0b1000000000000000 # instrumentation
bsr dx, ax 
and ebx, 48 
and rbx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rbx], 44 
xor rcx, -88 
lea ebx, qword ptr [rsi] 
test dl, al 
or eax, 415735913 
neg cl 
cmovno rdi, rax 
and rsi, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rsi] 
or esi, eax 
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl 
xadd edi, ebx 
or edx, edx 
sbb cl, 5 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add al, 23 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovns dx, word ptr [r14 + rcx] 
lea rdx, qword ptr [rdi] 
and rcx, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rcx] 
imul cx, dx 
or eax, 1838034473 
and rdi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovnp bx, word ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rcx], -859654998 
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], edi 
setb dl 
xchg rcx, rsi 
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], 127 
cmovnle esi, ecx 
adc cx, di 
and rax, 0b1111111111111 # instrumentation
sbb eax, dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], -49 
bts ax, 189 
add cl, 122 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdx] 
and rcx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rcx], -92 
sbb esi, -1 
cmovnl di, cx 
add ax, -3764 
xadd bl, bl 
and rcx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rcx], -51 
cmovnl ecx, eax 
adc rax, 2138699414 
and rcx, 0b1111111111111 # instrumentation
sbb bl, byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rax], 98 
sahf  
lea dx, qword ptr [rbx + rdx] 
and rdx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rdx] 
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
