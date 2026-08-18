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
add bl, -13 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setz dil 
setnp al 
adc ax, -10262 
and rax, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rax] 
lea rdx, qword ptr [rdi] 
cmp rdx, 92 
or cl, dl 
inc dl 
btc rcx, rbx 
mov cl, dl 
xchg rbx, rcx 
or cl, -42 
and rdi, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rdi], di 
and rdi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], 126 
and rcx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rcx], si 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add al, 53 # instrumentation
setnbe dl 
imul di, di 
and rsi, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rsi], 6 
xor bl, dl 
lea cx, qword ptr [rdi] 
and rax, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rax], 61 
and rdx, 0b1111111111111 # instrumentation
test word ptr [r14 + rdx], si 
and rcx, 0b1111111111111 # instrumentation
mov dx, word ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmovbe di, word ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rsi], eax 
sbb al, al 
and rcx, 0b1111111111111 # instrumentation
cmovp ebx, dword ptr [r14 + rcx] 
setp dl 
sub bl, 112 
or dil, 60 
mul si 
add al, 67 # instrumentation
mov eax, 1105811690 
and rcx, 0b1111111111111 # instrumentation
cmovl edi, dword ptr [r14 + rcx] 
cmp cl, -72 
movzx ecx, dl 
and rcx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rcx], 1224096093 
cmp bl, dl 
xchg edx, esi 
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], bl 
mov rbx, rbx 
cmovnb esi, edi 
and rdi, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdi] 
adc bx, dx 
and rsi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rsi], 94 
and edi, edi 
imul edi, ebx 
and rcx, 0b1111111111111 # instrumentation
cmp di, word ptr [r14 + rcx] 
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
add dl, -25 # instrumentation
mov di, 31696 
std  
adc sil, -108 
imul ecx, edx 
imul rsi, rsi, -43 
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], esi 
adc dl, cl 
sub bl, cl 
sub eax, -1943129902 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsf rcx, qword ptr [r14 + rdi] 
add bl, 62 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], dl 
lea rax, qword ptr [rbx + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovle dx, word ptr [r14 + rbx] 
lea di, qword ptr [rbx + rsi] 
setnle sil 
imul ebx 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rdi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdi], dl 
adc si, di 
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx] 
lea rdx, qword ptr [rax + rax] 
setnp bl 
lea ax, qword ptr [rdx + rdx] 
setl dl 
cmp al, cl 
adc dil, -103 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], 24 
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], al 
or bl, cl 
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], -39 
and rcx, 0b1111111111111 # instrumentation
xor cl, byte ptr [r14 + rcx] 
adc bl, 112 
sbb ax, ax 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
movsx cx, byte ptr [r14 + rdi] 
sub rax, 1426225571 
adc dl, dl 
and rsi, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rsi], 94 
add bl, -70 
cmovnz cx, ax 
lea bx, qword ptr [rdi + rdx + 5585] 
and rax, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rax] 
cmovno ebx, esi 
and rsi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rsi], eax 
and rdx, 0b1111111111111 # instrumentation
cmovnb ax, word ptr [r14 + rdx] 
cmovnp rdi, rax 
and rdi, 0b1111111111111 # instrumentation
imul rsi, qword ptr [r14 + rdi], -96 
add bl, 90 # instrumentation
setle bl 
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
