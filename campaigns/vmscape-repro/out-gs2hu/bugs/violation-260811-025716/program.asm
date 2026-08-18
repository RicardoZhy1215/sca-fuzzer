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
add bl, 115 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setl dl 
and rax, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rax], -96 
setl cl 
test bx, di 
or cl, sil 
and rdi, 0b1111111111111 # instrumentation
cmovo rdi, qword ptr [r14 + rdi] 
cmp bl, cl 
and rax, 0b1111111111111 # instrumentation
movsx edx, word ptr [r14 + rax] 
cmp cl, 120 
test dx, -14452 
adc cl, dil 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
dec dl 
imul ecx, edx 
and rdi, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
btr word ptr [r14 + rdi], si 
add bl, 53 # instrumentation
cmovs esi, ecx 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rbx, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rbx], dx 
and rax, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rax], rbx 
cdq  
and rcx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rsi], cl 
lea rax, qword ptr [rcx] 
and rsi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rsi], 26097 
and rcx, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmp bl, byte ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rcx] 
add al, -61 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rsi] 
cmovnle rdx, rax 
and rsi, 0b1111111111111 # instrumentation
movzx si, byte ptr [r14 + rsi] 
cmovnp si, dx 
imul di, dx, -4 
and rbx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rbx], dl 
sbb al, -8 
and rdi, 0b1111111111111 # instrumentation
and al, byte ptr [r14 + rdi] 
and al, al 
setnp dl 
and rsi, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rsi] 
lea bx, qword ptr [rdx + rbx + 1297] 
and rcx, 0b1111111111111 # instrumentation
cmovnz cx, word ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovnp edx, dword ptr [r14 + rbx] 
cmovp dx, cx 
and rdx, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rdx] 
or cl, dl 
setnz bl 
and rdi, 0b1111111111111 # instrumentation
add word ptr [r14 + rdi], cx 
lea rbx, qword ptr [rdx + rdx + 64074] 
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
sub ax, 32755 
sbb al, al 
lea si, qword ptr [rcx + rax + 31868] 
btc edi, 246 
sub eax, 1658303640 
xchg rdx, rax 
inc ax 
lea ax, qword ptr [rsi + rsi + 37860] 
and rdi, 0b1111111111111 # instrumentation
cmovnle bx, word ptr [r14 + rdi] 
cmovnb bx, di 
sub al, cl 
cmp rdi, -38 
xor dl, cl 
lea rsi, qword ptr [rdi + rdi] 
cmovnl esi, eax 
sub al, -53 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
test cl, 120 
lea ecx, qword ptr [rbx + rdi] 
adc eax, esi 
and rdi, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rdi], ax 
and rcx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rcx], 86 
test dl, bl 
and rcx, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rcx] 
adc rbx, 36 
and rdi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdi], 2071435117 
cmpxchg rdx, rdi 
and dl, bl 
and rdi, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdi], 110 
add eax, -1548062740 
clc  
and rdi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdi], -18812 
and rdx, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rdx] 
nop rcx 
mov edx, ebx 
adc sil, -67 
and rax, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rax], 48 
mul al 
movzx si, cl 
or al, cl 
bswap ecx 
and rbx, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rbx] 
xadd bl, dl 
and rdx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rdx], 0 
cmp al, -39 
cmp rax, rdx 
adc ebx, edx 
and rbx, 0b1111111111111 # instrumentation
cmovs bx, word ptr [r14 + rbx] 
setbe dil 
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
