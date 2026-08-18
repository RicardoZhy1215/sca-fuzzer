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
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rdx], 4 
and rcx, 0b1111111111111 # instrumentation
and rbx, 0b111 # instrumentation
btc qword ptr [r14 + rcx], rbx 
and rbx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rbx], ecx 
or ebx, 0b1000 # instrumentation
and bl, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv ebx 
or rsi, rax 
and rdx, rsi 
cmp esi, 101 
and rcx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rcx], 25 
add al, -99 
sub cl, al 
test dl, 59 
lea edx, qword ptr [rax] 
and rdi, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rdi] 
or al, dl 
cmovo rax, rdx 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
cmp dil, -105 
and cl, 27 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax 
adc al, dl 
cmp ax, -62 
and rbx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rbx], rsi 
and rsi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
sub rsi, qword ptr [r14 + rbx] 
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl 
btc di, 88 
and rcx, -18 
and rsi, 0b1111111111111 # instrumentation
mul word ptr [r14 + rsi] 
cmp rdx, 3 
and rax, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rax], 0 
and rdi, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rdi] 
add dl, dl 
mul dl 
imul di 
add cl, 27 # instrumentation
setns cl 
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], bx 
setle sil 
sub cl, cl 
and rsi, 0b1111111111111 # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdx], 33 
and rbx, 0b1111111111111 # instrumentation
xor dl, byte ptr [r14 + rbx] 
setns cl 
and rdx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdx], dx 
and rcx, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
test qword ptr [r14 + rax], rdx 
and rdx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdx], rax 
and rbx, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
add word ptr [r14 + rcx], -43 
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
lea cx, qword ptr [rcx + rdx + 21313] 
lea rdx, qword ptr [rsi] 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 1 # instrumentation
and dx, word ptr [r14 + rsi] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rsi] 
cmpxchg al, al 
sbb cl, al 
cmovb ecx, ebx 
cmpxchg ax, si 
or bl, 39 
setns cl 
btr si, 59 
add bl, 37 # instrumentation
lea ax, qword ptr [rbx + rdi] 
seto dl 
and rax, 0b1111111111111 # instrumentation
cmovb ax, word ptr [r14 + rax] 
mov rcx, 6855792686829076577 
setle dl 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], rcx 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
sub cl, al 
add edi, 56 
lea ax, qword ptr [rbx] 
cmovp rax, rdi 
and rax, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rax] 
movsx rdi, al 
and rdx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdx] 
movzx rsi, ax 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], dl 
and rdx, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rdx] 
cmovnle di, di 
btr ax, 103 
sub edi, eax 
cbw  
add al, 65 
or eax, -72 
and rcx, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rdi] 
movsx rdx, dx 
and rcx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rcx] 
lea rdx, qword ptr [rdi + rcx] 
and rsi, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rsi], ebx 
add ax, di 
and rax, 0b1111111111111 # instrumentation
neg word ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rdi] 
and ecx, -89 
btr si, 100 
lea ax, qword ptr [rbx + rdi + 14202] 
add dl, bl 
sets al 
xadd cl, cl 
lea rdx, qword ptr [rcx + rdi] 
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
