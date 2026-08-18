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
add cl, -22 # instrumentation
lea bx, qword ptr [rax + rax + 33601] 
and rax, 0b1111111111111 # instrumentation
cmovnz ax, word ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmp ax, word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rsi], -53 
or rsi, 0b1000000000000000000000000000000 # instrumentation
bsr rdx, rsi 
lea rdi, qword ptr [rdi + rsi] 
add al, 72 
and rdx, 0b1111111111111 # instrumentation
and esi, 0b111 # instrumentation
btc dword ptr [r14 + rdx], esi 
and rdi, 0b1111111111111 # instrumentation
adc rbx, qword ptr [r14 + rdi] 
stc  
or rsi, 0b1000000000000000000000000000000 # instrumentation
bsf rcx, rsi 
bt rax, rdi 
mov al, dl 
xchg rdx, rax 
and rdi, 0b1111111111000 # instrumentation
and edi, 0b111 # instrumentation
lock bts dword ptr [r14 + rdi], edi 
cmovz ax, ax 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, 122 # instrumentation
and rsi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rsi], sil 
lea di, qword ptr [rdi + rbx + 16797] 
sub al, 38 
lea rsi, qword ptr [rcx] 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
dec cl 
and rdx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdx], 30 
and rax, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rax] 
xchg al, al 
and rsi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rsi], 88 
and rdx, 0b1111111111111 # instrumentation
cmovbe ecx, dword ptr [r14 + rdx] 
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl 
xor cl, al 
and rdx, 0b1111111111111 # instrumentation
not word ptr [r14 + rdx] 
sbb bx, -86 
and sil, 127 
adc bl, al 
and rax, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rax], dx 
or dl, dl 
or rdi, 0b1000000000000000000000000000000 # instrumentation
bsf rsi, rdi 
add dl, 10 # instrumentation
and rdx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rdx] 
adc bl, cl 
and rcx, 0b1111111111111 # instrumentation
sbb ecx, dword ptr [r14 + rcx] 
xor eax, -1034276399 
or si, ax 
lea ebx, qword ptr [rbx + rcx + 45260] 
and rax, 0b1111111111111 # instrumentation
or edx, dword ptr [r14 + rax] 
add edi, edi 
and rcx, 0b1111111111111 # instrumentation
cmovz bx, word ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmovbe eax, dword ptr [r14 + rcx] 
dec dx 
and rdx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rdx], 26 
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
mul al 
or dl, dil 
xor rdx, rcx 
clc  
and rax, 0b1111111111111 # instrumentation
movzx rdx, word ptr [r14 + rax] 
lea si, qword ptr [rdi + rdi] 
cmovnp rdi, rax 
setbe dl 
lea ebx, qword ptr [rbx] 
lea ecx, qword ptr [rsi + rbx] 
xor ax, si 
and rsi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rsi] 
lea rcx, qword ptr [rax + rax] 
and bl, dl 
or rbx, rbx 
and rsi, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rsi], bx 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
neg ax 
and rax, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rax] 
test ax, 21743 
cmp rax, rdx 
bts di, dx 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rdi], bl 
cmp al, 121 
lea bx, qword ptr [rax + rdi] 
or eax, 537431845 
and rax, 0b1111111111111 # instrumentation
movsx esi, word ptr [r14 + rax] 
cmovo ecx, ecx 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rax] 
add cl, -85 # instrumentation
adc al, cl 
lea di, qword ptr [rax + rdi] 
and rdi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rdi], bl 
and rbx, 0b1111111111111 # instrumentation
and word ptr [r14 + rbx], 22 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 0b1000 # instrumentation
and byte ptr [r14 + rbx], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rbx] 
sub dl, al 
and rdi, 0b1111111111111 # instrumentation
imul rax, qword ptr [r14 + rdi], -27 
adc eax, 653897441 
and rdi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rdi], 2 
add cl, 19 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnle ax, word ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovnp eax, dword ptr [r14 + rsi] 
inc edx 
and rbx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rbx], sil 
btc ecx, edx 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsf rcx, qword ptr [r14 + rdi] 
add dl, -31 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
movsx esi, word ptr [r14 + rsi] 
inc dl 
sbb dl, -83 
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
