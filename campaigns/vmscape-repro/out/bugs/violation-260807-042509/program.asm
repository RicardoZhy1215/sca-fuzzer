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
add al, 27 # instrumentation
movzx ecx, bl 
and rsi, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rsi] 
and rax, 575397837 
movsx edx, cx 
clc  
nop dx 
sub rax, 99 
sbb dl, bl 
and rbx, 0b1111111111111 # instrumentation
and sil, byte ptr [r14 + rbx] 
dec esi 
sub dil, -116 
and rdx, 0b1111111111111 # instrumentation
adc rcx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdx], -19 
test al, 74 
and rdi, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmp di, word ptr [r14 + rbx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111000 # instrumentation
and si, 0b111 # instrumentation
lock btr word ptr [r14 + rdx], si 
add dl, -112 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovnle cx, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], -79 
and rdi, 0b1111111111111 # instrumentation
cmovbe eax, dword ptr [r14 + rdi] 
cmp dil, 105 
and rbx, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rbx] 
sbb bl, bl 
test bl, bl 
and rbx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rbx] 
add bl, -91 # instrumentation
cmovnle rbx, rdx 
and rdx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rdx], 6 
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], dl 
and rbx, 0b1111111111111 # instrumentation
cmovno esi, dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], al 
and rbx, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rbx], 5 
cmp al, 1 
imul rbx, rbx 
add bl, -111 # instrumentation
lea edx, qword ptr [rsi + rbx] 
cmovnle ebx, esi 
and rcx, 0b1111111111111 # instrumentation
sbb eax, dword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsr ecx, dword ptr [r14 + rdi] 
add bl, 14 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovns ax, word ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rcx], edi 
cmovbe esi, eax 
btc rdx, 162 
and eax, 347601867 
xadd bl, al 
test al, -98 
and rcx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rcx] 
cmovz rax, rbx 
and rdx, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rdx] 
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
add cl, -38 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
not si 
setno al 
cmovnl di, ax 
cmp al, dl 
cmp al, bl 
and rax, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rax] 
mul edi 
or esi, esi 
and rcx, 0b1111111111111 # instrumentation
movsx bx, byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], 101 
and rdx, 0b1111111111111 # instrumentation
cmovnl dx, word ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
and ecx, 0b111 # instrumentation
lock btc dword ptr [r14 + rdi], ecx 
and rcx, 0b1111111111111 # instrumentation
cmovz si, word ptr [r14 + rcx] 
lea ax, qword ptr [rax + rsi + 54886] 
test rcx, -1547787336 
lea eax, qword ptr [rsi] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add dl, 99 # instrumentation
and rdi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdi], dil 
and rdi, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdi], 51 
and rdx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdx], -42 
and rax, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rax], -522657033 
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmovz ebx, dword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmovnle edx, dword ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rax], rbx 
and dl, cl 
and rsi, 0b1111111111000 # instrumentation
lock not qword ptr [r14 + rsi] 
mov bl, bl 
and rbx, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
adc al, byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rdx] 
and rbx, -123 
and rax, 0b1111111111111 # instrumentation
add edi, dword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rsi], 87 
and rdi, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rdi] 
lea rdi, qword ptr [rcx + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovo rcx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
sub ebx, dword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmovo rdi, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
adc dx, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
dec word ptr [r14 + rdi] 
setbe bl 
cmovnbe rdi, rdx 
and rbx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rbx] 
add cl, -54 # instrumentation
cmovns dx, di 
or bl, dl 
xchg rdi, rax 
lea di, qword ptr [rax + rax] 
test eax, 1267201310 
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
