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
add al, 56 # instrumentation
cmovo rsi, rdi 
setnl cl 
and rdi, 0b1111111111111 # instrumentation
cmovnp ebx, dword ptr [r14 + rdi] 
bts rax, 152 
lea bx, qword ptr [rdx] 
and rax, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rax], 5 
lea rdi, qword ptr [rsi + rcx + 57782] 
or esi, 0b1000000000000000000000000000000 # instrumentation
bsf edx, esi 
lea rbx, qword ptr [rdi] 
test ax, 5528 
lea rbx, qword ptr [rax + rax] 
and rdx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rdx], ecx 
add cl, 35 
or al, -20 
lea si, qword ptr [rdx] 
and rsi, 0b1111111111111 # instrumentation
and rdx, 0b111 # instrumentation
bts qword ptr [r14 + rsi], rdx 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdi 
sub dx, dx 
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], ecx 
and rbx, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rbx], -126 
sub rax, -677337518 
sub bl, cl 
cmp dil, 23 
test ax, 26287 
mov di, ax 
and rbx, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rbx], rcx 
and rdi, 0b1111111111111 # instrumentation
sbb rbx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rax], bl 
and rdi, 0b1111111111111 # instrumentation
cmp edx, dword ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi 
setnl bl 
and rsi, 0b1111111111111 # instrumentation
adc ax, word ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rdx] 
movsx ecx, bl 
and rbx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rbx], cl 
lea rsi, qword ptr [rsi + rsi + 21962] 
and rax, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rax], -90 
and rdi, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdi], 62 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 0b1000000000000000 # instrumentation
bsr si, word ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rax] 
dec di 
and rbx, 0b1111111111111 # instrumentation
cmovnp rax, qword ptr [r14 + rbx] 
and bl, 50 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
cmovb ebx, edi 
adc cl, -25 
and rdx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdx], edx 
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
or sil, 1 # instrumentation
mov ax, 1 # instrumentation
div sil 
add dl, 99 # instrumentation
and rdx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdx], rdx 
and rdi, 0b1111111111111 # instrumentation
mul word ptr [r14 + rdi] 
setb bl 
sub al, cl 
cmp rcx, rdx 
lea rdi, qword ptr [rdi] 
movsx rax, bx 
cmovb rdx, rdx 
lea rdx, qword ptr [rdi + rdi] 
and rax, 0b1111111111111 # instrumentation
cmovns edi, dword ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rbx], -1 
test rbx, 1856732833 
lea ax, qword ptr [rsi + rcx] 
and rax, 0b1111111111111 # instrumentation
movsx dx, byte ptr [r14 + rax] 
cmp al, dl 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
and rcx, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rcx], rax 
and rbx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rbx], 6 
and rdx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rdx] 
mul dx 
sbb cl, -124 
sbb sil, sil 
and rax, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rax], ebx 
and rcx, 0b1111111111111 # instrumentation
and rax, 0b111 # instrumentation
bt qword ptr [r14 + rcx], rax 
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], sil 
and rcx, 0b1111111111111 # instrumentation
add bx, word ptr [r14 + rcx] 
btc eax, 79 
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsf rdi, qword ptr [r14 + rbx] 
lea eax, qword ptr [rsi + rdi] 
and ax, 13715 
and rax, 0b1111111111000 # instrumentation
and rdi, 0b111 # instrumentation
lock bts qword ptr [r14 + rax], rdi 
add bl, -102 # instrumentation
cmovle rax, rdx 
sbb rax, 32095182 
and rax, 0b1111111111000 # instrumentation
and rax, 0b111 # instrumentation
lock bts qword ptr [r14 + rax], rax 
sub ax, 23027 
and rsi, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rsi], ecx 
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], di 
and rax, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rax], dl 
and rcx, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rcx], 15 
setnb al 
and rdi, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rdi], 6 
add bl, 12 # instrumentation
setnle cl 
or ecx, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv ecx 
add dl, -123 # instrumentation
cmovnle rax, rsi 
and rsi, 0b1111111111111 # instrumentation
not qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
cmovnl cx, word ptr [r14 + rcx] 
and rbx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rbx], -49 
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
