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
and rax, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rax], di 
and rsi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rsi] 
or al, cl 
and rdx, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rdx] 
mov cl, cl 
cwde  
sbb al, 84 
test cl, bl 
and rdx, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
bt word ptr [r14 + rdx], di 
and rdi, 0b1111111111111 # instrumentation
cmovnb rdi, qword ptr [r14 + rdi] 
lea edx, qword ptr [rdi] 
lea edi, qword ptr [rdx + rbx] 
inc cl 
nop ax 
xchg al, cl 
and rax, 0b1111111111111 # instrumentation
sub cl, byte ptr [r14 + rax] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
cmovnle ebx, dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdx], edi 
and rcx, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rcx], 7 
and rdx, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rdx], 2 
add bl, -21 # instrumentation
cmovnp rcx, rdi 
cmovnz dx, ax 
mov esi, 374587039 
and rcx, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rcx], dl 
and rsi, 0b1111111111111 # instrumentation
or sil, byte ptr [r14 + rsi] 
sub dl, 9 
setnbe dl 
lea rbx, qword ptr [rbx + rsi] 
cmp ebx, 26 
and rsi, 0b1111111111111 # instrumentation
cmovp cx, word ptr [r14 + rsi] 
mov dx, ax 
sub rbx, rax 
and rsi, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
btr dword ptr [r14 + rsi], edi 
add al, 83 # instrumentation
setnbe dl 
and rax, 0b1111111111111 # instrumentation
cmovns ecx, dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
and rcx, 0b111 # instrumentation
bt qword ptr [r14 + rdi], rcx 
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl 
and rbx, 0b1111111111111 # instrumentation
or al, byte ptr [r14 + rbx] 
nop  
test rax, rdx 
xadd bl, dl 
and rdx, 0b1111111111111 # instrumentation
cmovnp di, word ptr [r14 + rdx] 
xchg rdi, rbx 
cmovz rbx, rsi 
mov ebx, edi 
movzx di, bl 
and rsi, 0b1111111111111 # instrumentation
sbb rbx, qword ptr [r14 + rsi] 
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
btc rdi, rsi 
sbb al, 80 
lea bx, qword ptr [rcx + rcx] 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], al 
cmp rbx, rcx 
or rdi, 0b1000000000000000000000000000000 # instrumentation
bsr rbx, rdi 
add cl, -11 # instrumentation
and rdx, 0b1111111111111 # instrumentation
adc edx, dword ptr [r14 + rdx] 
lea edi, qword ptr [rcx + rdi] 
sub ebx, ecx 
dec al 
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi] 
add al, 71 # instrumentation
setnle dl 
add al, 80 
and rdi, 0b1111111111111 # instrumentation
cmovp rdx, qword ptr [r14 + rdi] 
xor dil, 14 
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], 76 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
or rax, 0b1000000000000000000000000000000 # instrumentation
bsf rsi, rax 
add rbx, -16 
lea rax, qword ptr [rbx + rcx] 
and rbx, 0b1111111111111 # instrumentation
add bl, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rbx], 1 
and rdx, 0b1111111111111 # instrumentation
and dl, byte ptr [r14 + rdx] 
sbb bl, bl 
xchg bl, cl 
adc ax, -14686 
xor al, 5 
setnb cl 
imul ebx, eax, 91 
and rax, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rax] 
sub cl, cl 
xor si, -51 
and rdi, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
btr word ptr [r14 + rdi], dx 
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], cl 
and rdi, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rdx], eax 
cmc  
sub al, -70 
and rax, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rax] 
setl al 
lea ebx, qword ptr [rsi + rdx] 
and rbx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rbx], 1 
and rdi, 0b1111111111111 # instrumentation
adc dl, byte ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmp eax, dword ptr [r14 + rax] 
lea edi, qword ptr [rsi + rcx + 9064] 
and rsi, 0b1111111111111 # instrumentation
movsx esi, word ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rdx], dl 
setnb al 
xor eax, -1251073101 
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
