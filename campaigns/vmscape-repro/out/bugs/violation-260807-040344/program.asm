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
imul di, si, -31 
or edx, eax 
and rdx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rdx] 
sub si, ax 
lea cx, qword ptr [rcx + rcx] 
and rax, 0b1111111111111 # instrumentation
imul word ptr [r14 + rax] 
add al, -73 # instrumentation
cmovnp cx, si 
or di, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv di 
add al, 104 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnz si, word ptr [r14 + rdx] 
or ax, -53 
imul dx 
imul rbx, rcx, 28 
add dl, -75 # instrumentation
btr rbx, rcx 
and rsi, 0b1111111111111 # instrumentation
cmovz rdi, qword ptr [r14 + rsi] 
xchg ebx, edi 
movsx ecx, cl 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 1 # instrumentation
and edx, dword ptr [r14 + rax] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rax] 
cmp bl, al 
lea ax, qword ptr [rdx + rax] 
and rbx, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rbx] 
movsx esi, al 
sub ax, -5835 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rdi 
and rsi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdi], 9 
mul ax 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rdi] 
add dl, -126 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovns ecx, dword ptr [r14 + rcx] 
xadd edx, ebx 
and rax, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rax], di 
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 1 # instrumentation
cmp di, bx 
sub sil, 126 
neg edx 
and rsi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rsi], eax 
sub rax, -260526594 
cmovnb dx, ax 
and rsi, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx 
and rax, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rax], bl 
test cl, dl 
and rcx, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rsi], 46 
sbb dl, -101 
and rax, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rax], rsi 
add rbx, rsi 
lea si, qword ptr [rcx + rcx + 18855] 
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
and rbx, 0b1111111111111 # instrumentation
mov ecx, dword ptr [r14 + rbx] 
add ax, 11885 
sub edi, edi 
clc  
and al, 85 
cbw  
xchg esi, eax 
lea rcx, qword ptr [rax] 
lea edi, qword ptr [rdx] 
setp bl 
lea edx, qword ptr [rdx] 
xchg edx, eax 
setns al 
and rsi, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rsi] 
xor dl, dl 
cmovz rdi, rax 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and dl, dl 
and rbx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
lea rdi, qword ptr [rdi] 
movsx edi, al 
sbb rbx, rsi 
cmp dl, bl 
xor bl, cl 
and rbx, 0b1111111111111 # instrumentation
cmovs dx, word ptr [r14 + rbx] 
imul edx, eax 
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], -1343063799 
and rsi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
cmovb rdx, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
bt word ptr [r14 + rsi], dx 
add cl, bl 
and rsi, 0b1111111111111 # instrumentation
adc bx, word ptr [r14 + rsi] 
movzx rdx, cl 
lea esi, qword ptr [rdx + rbx + 64286] 
cmovs eax, eax 
and rax, 0b1111111111111 # instrumentation
cmovs si, word ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rdi], cx 
and rax, 0b1111111111111 # instrumentation
cmovnbe di, word ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovb si, word ptr [r14 + rbx] 
imul edx, ebx, -32 
lea ecx, qword ptr [rax + rbx + 15799] 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rsi], ebx 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
bt edx, esi 
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], dl 
xadd bl, al 
and rdx, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rcx] 
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
