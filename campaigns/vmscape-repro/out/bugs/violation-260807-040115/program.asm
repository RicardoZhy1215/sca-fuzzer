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
and rsi, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rsi], 66 
cmovnbe eax, edi 
imul rcx, rdx, 127 
and rbx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rbx], 13 
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], al 
and rdx, 0b1111111111111 # instrumentation
cmovs di, word ptr [r14 + rdx] 
lea cx, qword ptr [rax + rax] 
mov ebx, ecx 
cmp al, dil 
sbb bl, dl 
add bl, cl 
cwde  
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], -7 
cmovbe edx, eax 
and rax, 0b1111111111111 # instrumentation
and rbx, 0b111 # instrumentation
bt qword ptr [r14 + rax], rbx 
cmp rax, -113 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rsi, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rsi], dil 
and rcx, 0b1111111111111 # instrumentation
cmovle ecx, dword ptr [r14 + rcx] 
adc rbx, rcx 
setnl sil 
cmp dl, cl 
and rdx, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rdx], 114 
lea rdi, qword ptr [rdi + rcx] 
sbb dl, cl 
and rsi, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rsi], bl 
and rax, 0b1111111111111 # instrumentation
add dl, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
xor rdi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rsi], -55 
movzx rdx, al 
and rcx, 0b1111111111111 # instrumentation
test word ptr [r14 + rcx], -16586 
sbb al, 42 
and rdi, 0b1111111111111 # instrumentation
not word ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
cmovnle cx, word ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rbx] 
sbb al, 7 
and rdi, 0b1111111111111 # instrumentation
cmovnp cx, word ptr [r14 + rdi] 
and rsi, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rsi] 
cmp dl, al 
and rcx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rcx], 36 
cmovns si, si 
and rdi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rdi], 33 
and rbx, 0b1111111111111 # instrumentation
cmovbe cx, word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rsi], eax 
and rsi, 0b1111111111111 # instrumentation
and edi, dword ptr [r14 + rsi] 
adc bl, sil 
and rcx, 0b1111111111111 # instrumentation
cmovo dx, word ptr [r14 + rcx] 
mov dl, bl 
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
movsx rsi, bl 
and rdx, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rdx], 7 
cmp bx, -12 
xor dil, dl 
add ax, 18901 
and rbx, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bts word ptr [r14 + rbx], si 
add al, -1 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdi] 
clc  
cmp cl, 101 
test cl, -61 
setbe bl 
lea ebx, qword ptr [rcx] 
cmovz rdi, rbx 
sub al, dl 
lea dx, qword ptr [rsi + rsi + 36744] 
lea si, qword ptr [rdx + rbx + 35613] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
imul esi, esi, -26 
add bl, 113 # instrumentation
nop  
cmovnbe rdi, rbx 
and rdi, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdi], ecx 
and rax, 0b1111111111111 # instrumentation
cmovno edi, dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
xor cl, byte ptr [r14 + rdx] 
movsx edx, dl 
and rbx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rbx], -128 
bt rbx, rcx 
and rax, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rax], 6 
or dl, 1 # instrumentation
lea di, qword ptr [rax] 
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], -108 
sub dil, 50 
or eax, -1979101694 
and rax, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rax], di 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx 
lea eax, qword ptr [rdi] 
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 2 
and rsi, 0b1111111111111 # instrumentation
sub dl, byte ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
xor sil, byte ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rbx], -45 
setb cl 
and rbx, 0b1111111111111 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx] 
btc si, 234 
and rax, 0b1111111111111 # instrumentation
sbb dil, byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdi], 92 
and rax, 0b1111111111111 # instrumentation
cmovnp rax, qword ptr [r14 + rax] 
and ax, 28770 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000000000000000 # instrumentation
bsr cx, word ptr [r14 + rax] 
add cl, 97 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovl edx, dword ptr [r14 + rax] 
mul si 
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
