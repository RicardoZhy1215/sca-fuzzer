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
test cl, -4 
setz dil 
mov di, -20805 
bts di, ax 
and rsi, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rax] 
cmp dil, 89 
test sil, 106 
cmovle edx, eax 
mov bl, 90 
mul sil 
and al, dl 
and rbx, 0b1111111111111 # instrumentation
movsx ecx, word ptr [r14 + rbx] 
lea cx, qword ptr [rdi + rbx] 
and bl, cl 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
xor bl, -35 
cmp rdx, rdi 
and rsi, 0b1111111111111 # instrumentation
add ecx, dword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rcx], 92 
sbb ax, 68 
sub rcx, 100 
bt rdi, rbx 
sbb eax, -2133510217 
lea rdi, qword ptr [rsi + rbx] 
cmovp rax, rbx 
and rdi, 0b1111111111111 # instrumentation
add dl, byte ptr [r14 + rdi] 
adc al, 55 
and rbx, 0b1111111111111 # instrumentation
cmovp ebx, dword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rsi] 
lea bx, qword ptr [rdx + rbx + 62317] 
and rdx, 0b1111111111111 # instrumentation
sbb esi, dword ptr [r14 + rdx] 
lea rdi, qword ptr [rsi + rdx + 14765] 
test al, dl 
cwde  
test cl, 78 
and rbx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rbx], edi 
and rdi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rdi], 21 
and rdi, 0b1111111111111 # instrumentation
movsx edi, word ptr [r14 + rdi] 
sub dl, -41 
and rbx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rbx], ecx 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdx 
and rax, 0b1111111111111 # instrumentation
movsx eax, byte ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rbx], 30 
and rbx, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
or bx, word ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rbx] 
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
or bl, bl 
and rax, 0b1111111111111 # instrumentation
test dword ptr [r14 + rax], esi 
lea edi, qword ptr [rbx + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmovnp ebx, dword ptr [r14 + rdi] 
add dl, dl 
add rbx, rax 
cmovno edi, ecx 
lea rdi, qword ptr [rdi] 
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], edx 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], rcx 
imul rdx, rcx, -116 
cmp dl, -20 
test rdi, -1214616476 
lea si, qword ptr [rsi + rax + 21323] 
and rsi, 0b1111111111111 # instrumentation
cmovns esi, dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], al 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
add dl, -33 # instrumentation
and rcx, 0b1111111111111 # instrumentation
setl byte ptr [r14 + rcx] 
nop rax 
cmovl si, di 
and rdi, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdi], edx 
lea rbx, qword ptr [rax] 
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], 84 
and rbx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rbx], eax 
and rsi, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rsi] 
movzx bx, dl 
and rax, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rax], -96 
and rax, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rax], cl 
lea ebx, qword ptr [rbx] 
cmovns dx, di 
and rax, 0b1111111111111 # instrumentation
sub esi, dword ptr [r14 + rax] 
bt ebx, 76 
and rdx, 0b1111111111111 # instrumentation
cmovz ecx, dword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdx], 47 
mov cl, cl 
and rcx, 0b1111111111111 # instrumentation
cmovnp ecx, dword ptr [r14 + rcx] 
lea bx, qword ptr [rdx + rcx] 
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsr esi, ecx 
and rsi, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rsi], esi 
sbb ebx, -83 
cmovnp si, di 
bts dx, 61 
and rbx, 0b1111111111111 # instrumentation
xor dl, byte ptr [r14 + rbx] 
lea rbx, qword ptr [rdx + rax + 43764] 
setnbe cl 
and rsi, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf ax, word ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rdx] 
xor dl, cl 
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
