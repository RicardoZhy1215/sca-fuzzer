.intel_syntax noprefix
.section .data.main
.function_main_0:
.bb_main_0.entry:
.macro.set_h2g_target.attacker.function_attacker_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.0: nop qword ptr [rax + 0xff]
.exit_main_0:
.function_main_1:
.bb_main_1.entry:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.attacker.function_attacker_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.1: nop qword ptr [rax + 0xff]
.exit_main_1:
.function_main_2:
.bb_main_2.entry:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_k2u_target.victim.function_victim_1: nop qword ptr [rax + 0xff]
.macro.set_u2k_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.switch_k2u.victim.1: nop qword ptr [rax + 0xff]
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_u2k.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.attacker.function_attacker_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.attacker.2: nop qword ptr [rax + 0xff]
.exit_main_3:
.function_main_4:
.bb_main_4.entry:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop  
.exit_main_4:
.section .data.victim
.function_victim_1:
.bb_victim_1.entry:
.macro.landing_k2u.victim_1: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
lea rsi, qword ptr [rsi + rbx + 22515] 
lea dx, qword ptr [rsi] 
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], 62 
xor cl, dl 
adc ecx, 7 
sbb al, cl 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rdi 
lea si, qword ptr [rcx + rdi] 
and rsi, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rsi], -51 
or eax, 0b1000000000000000000000000000000 # instrumentation
bsf ebx, eax 
add al, 97 # instrumentation
adc al, -101 
sbb dl, bl 
adc rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
and word ptr [r14 + rcx], di 
lea ebx, qword ptr [rsi + rsi + 39615] 
sbb cl, al 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
movzx rsi, dil 
and rcx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rcx], -1745271975 
and rcx, 0b1111111111111 # instrumentation
not word ptr [r14 + rcx] 
add bl, dl 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], -50 
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rsi], -66 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx 
and rdx, 0b1111111111111 # instrumentation
cmovz edi, dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdi] 
cmovl rax, rax 
and rsi, 0b1111111111111 # instrumentation
adc cx, word ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], -67 
not dil 
and rdi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdi], dx 
cmp bl, cl 
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx] 
btc edx, 9 
add dl, 117 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovnp rsi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf rdi, qword ptr [r14 + rax] 
test si, -19476 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 0b1000000000000000 # instrumentation
bsf di, word ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
bt dword ptr [r14 + rcx], edi 
adc al, 47 
and rcx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rcx], -81 
and rbx, 0b1111111111111 # instrumentation
or rdi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rdx], -38 
and rsi, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rsi] 
lea ecx, qword ptr [rax + rbx] 
lfence  # noremove
.macro.switch_u2k.main.1: nop qword ptr [rax + 0xff]
lfence  
.exit_victim_safe:
.section .data.attacker
.function_attacker_0:
.bb_attacker_0.entry:
.macro.landing_h2g.attacker_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.attacker_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.attacker_0: nop qword ptr [rax + 0xff]
lfence  
.exit_attacker_0:
.function_attacker_train:
.bb_attacker_train.entry:
.macro.landing_h2g.attacker_1: nop qword ptr [rax + 0xff]
and dl, cl 
xor eax, eax 
bts rax, 195 
and rdi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdi], ebx 
adc eax, -1753716568 
and rcx, 0b1111111111111 # instrumentation
not dword ptr [r14 + rcx] 
setbe bl 
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], bl 
movsx esi, di 
or rax, -918432166 
movsx si, cl 
add rcx, rbx 
cmc  
or ax, 30601 
adc rcx, rbx 
and rdx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdx], bl 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
lea rdx, qword ptr [rsi + rbx] 
btc rbx, 86 
lea dx, qword ptr [rbx + rdx + 63540] 
and rcx, 0b1111111111111 # instrumentation
movzx rdi, word ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
or edx, dword ptr [r14 + rsi] 
cmpxchg cl, bl 
cmovo esi, edx 
and rsi, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], 121 
and rbx, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], 46 
and rax, 0b1111111111111 # instrumentation
cmovp rbx, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
sbb eax, 786141764 
and rsi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rsi], 121 
add rbx, 84 
test cl, 78 
lea bx, qword ptr [rdx + rax] 
add al, dl 
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
and rcx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rcx], 63 
setnb bl 
and rcx, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rcx], 3 
lea edi, qword ptr [rdi + rdx] 
xchg rcx, rax 
and rdi, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rbx], 6 
setb sil 
lea di, qword ptr [rdi + rsi + 34196] 
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], -25 
lfence  
.macro.switch_g2h.main.attacker_1: nop qword ptr [rax + 0xff]
lfence  
.exit_attacker_gadget:
.function_attacker_measure:
.bb_attacker_measure.entry:
.macro.landing_h2g.attacker_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.attacker_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_attacker_measure:
.section .data.main
.test_case_exit:nop
