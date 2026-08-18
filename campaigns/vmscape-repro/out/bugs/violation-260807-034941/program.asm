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
btc cx, si 
dec ebx 
xchg edx, eax 
setns sil 
cmovno rdi, rbx 
xor ax, -23421 
sub rax, 42 
and rdi, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdi] 
or bl, -124 
test rdi, 127343660 
adc cx, 67 
imul bl 
add bl, -59 
and rax, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rax] 
and al, -54 
cmp cl, -61 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
mov rcx, rcx 
lea ebx, qword ptr [rbx + rdx] 
cmp sil, cl 
and rcx, 0b1111111111111 # instrumentation
movsx edx, word ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mov dil, byte ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rdi] 
cmovs rcx, rcx 
and rsi, 0b1111111111111 # instrumentation
sub cl, byte ptr [r14 + rsi] 
cmovns rcx, rsi 
lea esi, qword ptr [rbx] 
and rdx, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rdx], 61 
lea bx, qword ptr [rax + rsi] 
not cl 
and rdx, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rdx] 
lea cx, qword ptr [rsi + rbx] 
or sil, -28 
xor ax, 21 
lea rbx, qword ptr [rcx + rax + 3360] 
lea rax, qword ptr [rcx + rdi + 52822] 
and al, cl 
lea rcx, qword ptr [rbx + rax] 
cbw  
and rbx, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmovs ax, word ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rsi], 1 
and rsi, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rsi], ax 
and rax, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rdx] 
cmovz ebx, edi 
lea esi, qword ptr [rdx + rbx] 
cmovl ebx, esi 
cmovnp rsi, rcx 
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
mov bl, -27 
and rsi, 0b1111111111111 # instrumentation
cmp dl, byte ptr [r14 + rsi] 
test cl, -9 
nop  
lea ecx, qword ptr [rdx] 
test rcx, rdi 
lea ax, qword ptr [rcx] 
and rcx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rcx] 
sub bl, cl 
and ax, -18650 
cmovnz ecx, ebx 
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl 
btr edi, 93 
add rax, -30312099 
or eax, 1139620378 
and rdi, 0b1111111111111 # instrumentation
movzx ebx, word ptr [r14 + rdi] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], -47 
mov cl, -90 
and rdi, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi] 
cmp bl, cl 
add ax, -22 
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
or edi, dword ptr [r14 + rsi] 
movzx ecx, dl 
sbb sil, 45 
and rax, 0b1111111111111 # instrumentation
add sil, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
and dword ptr [r14 + rsi], -104 
bts edx, 28 
add cl, -18 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rsi], cx 
cmovp esi, ebx 
and rsi, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rsi], ax 
and rax, 0b1111111111111 # instrumentation
xor ebx, dword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmovno edi, dword ptr [r14 + rcx] 
lea edi, qword ptr [rdx + rbx + 18637] 
and rdi, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdi], -45 
xor dl, -21 
and rdx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdx], 359883975 
and rax, 0b1111111111111 # instrumentation
test qword ptr [r14 + rax], 405663412 
add dl, dl 
lea cx, qword ptr [rbx] 
and rbx, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rbx] 
mov esi, 685354787 
and rcx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rcx], al 
setp dl 
add bl, dl 
and rcx, 0b1111111111111 # instrumentation
or cl, byte ptr [r14 + rcx] 
add cx, cx 
and rax, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rax], 0 
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
