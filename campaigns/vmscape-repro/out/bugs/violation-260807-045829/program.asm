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
add dl, -40 # instrumentation
cmovnp eax, ebx 
sbb rax, -113 
sets sil 
and rbx, 0b1111111111111 # instrumentation
cmovs ecx, dword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdx], edi 
sbb dl, cl 
xadd bl, bl 
test cl, al 
lea rcx, qword ptr [rax] 
add bl, 86 
sbb bl, sil 
test ebx, ebx 
and rdi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdi], 4 
setz al 
and rdx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdx], -24 
and rcx, 0b1111111111111 # instrumentation
and rax, qword ptr [r14 + rcx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, -40 # instrumentation
and rdx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rdx], 61 
lea edx, qword ptr [rcx + rax] 
movsx rbx, cl 
adc dx, bx 
cwde  
and rax, 0b1111111111111 # instrumentation
cmovnl edi, dword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rcx] 
sub cx, di 
and rcx, 0b1111111111111 # instrumentation
imul di, word ptr [r14 + rcx] 
cmpxchg si, di 
and rsi, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rsi] 
lea ebx, qword ptr [rbx] 
cmovnl rbx, rbx 
btc rbx, 136 
and rdx, 0b1111111111111 # instrumentation
cmovnb rsi, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], ax 
lea rcx, qword ptr [rbx + rcx] 
cmovb rsi, rsi 
and rdi, 0b1111111111111 # instrumentation
movzx esi, word ptr [r14 + rdi] 
sbb si, si 
lea ecx, qword ptr [rdx] 
setbe bl 
cmp dil, 29 
and rsi, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rsi], cx 
lea edi, qword ptr [rax + rax] 
and rbx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rbx], -93 
and rdi, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rdi] 
add al, -36 # instrumentation
and rdi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdi] 
setp dl 
and rdi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rdi], bx 
adc eax, 97965009 
and rbx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rbx] 
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
add bl, -66 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
lea rax, qword ptr [rbx + rax] 
lea di, qword ptr [rbx] 
cmovbe bx, bx 
adc si, si 
and rbx, 0b1111111111111 # instrumentation
or ebx, dword ptr [r14 + rbx] 
movzx bx, cl 
lea rax, qword ptr [rcx + rdx] 
cmovo edx, eax 
lea bx, qword ptr [rdx + rbx] 
mov eax, esi 
setnl bl 
setnz cl 
or eax, 760969501 
or si, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv si 
cwd  
cmpxchg rdi, rsi 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
xadd ax, dx 
and rsi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rsi], dl 
and rdx, 0b1111111111111 # instrumentation
cmovnz rsi, qword ptr [r14 + rdx] 
setnb dl 
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000000000000000 # instrumentation
bsr bx, word ptr [r14 + rcx] 
add al, -86 # instrumentation
adc al, cl 
setnb dl 
neg sil 
or cl, cl 
add cl, al 
or cl, al 
and rdx, 0b1111111111111 # instrumentation
and dl, byte ptr [r14 + rdx] 
lea bx, qword ptr [rcx + rdx + 24215] 
and rbx, 0b1111111111111 # instrumentation
btr word ptr [r14 + rbx], 6 
add dl, -25 # instrumentation
cmovnl si, ax 
cmovnl ebx, ebx 
lea bx, qword ptr [rax] 
nop edx 
and rcx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rcx], ecx 
and rax, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rax], bl 
and rax, 0b1111111111000 # instrumentation
lock bts dword ptr [r14 + rax], 4 
and rdx, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rdx] 
cmovp rdx, rsi 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 88 
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rbx], al 
xchg ecx, esi 
and rbx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rbx], 520485921 
and rcx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rcx], ecx 
and rsi, 0b1111111111111 # instrumentation
and rdx, 0b111 # instrumentation
bt qword ptr [r14 + rsi], rdx 
and rcx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rcx] 
lea eax, qword ptr [rbx + rbx] 
cmovp bx, si 
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
