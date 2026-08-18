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
add cl, -112 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
movzx ecx, bl 
and rdx, 0b1111111111111 # instrumentation
adc cl, byte ptr [r14 + rdx] 
or dl, 22 
and rdx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdx], -61 
cmovs rcx, rsi 
and rax, 0b1111111111111 # instrumentation
or dx, word ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], al 
imul cx, bx, 42 
and rcx, 0b1111111111111 # instrumentation
dec word ptr [r14 + rcx] 
cmovnp rdx, rdi 
xor al, al 
lea esi, qword ptr [rdi] 
or dx, 123 
and rax, 0b1111111111111 # instrumentation
cmovz rbx, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmovns ax, word ptr [r14 + rdi] 
and sil, sil 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
xor al, cl 
cmp rdi, -85 
setnz al 
and rdi, 0b1111111111111 # instrumentation
cmovz di, word ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rcx], ecx 
and rbx, 0b1111111111111 # instrumentation
movsx si, byte ptr [r14 + rbx] 
cmovns rdi, rax 
mov rdi, -7226345627702256733 
and rsi, 0b1111111111111 # instrumentation
adc rdi, qword ptr [r14 + rsi] 
setnl dil 
sbb bl, -57 
imul cl 
and rsi, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rsi], rsi 
or cx, cx 
and rcx, 0b1111111111111 # instrumentation
dec word ptr [r14 + rcx] 
movzx ebx, bl 
and rax, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rax], cl 
and rdx, 0b1111111111111 # instrumentation
imul rax, qword ptr [r14 + rdx] 
cmpxchg bl, al 
and rbx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rbx], sil 
and rdi, 0b1111111111111 # instrumentation
cmovs dx, word ptr [r14 + rdi] 
sub al, -7 
btc rdi, rdx 
neg rbx 
and rsi, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rsi], 118 
and rax, 0b1111111111111 # instrumentation
cmovnl ecx, dword ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rdi], di 
xadd ecx, eax 
and rbx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rbx], dl 
and rsi, 0b1111111111111 # instrumentation
sub rdi, qword ptr [r14 + rsi] 
movzx edi, di 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 26 
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
and rsi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rsi], ebx 
and rax, 0b1111111111111 # instrumentation
add ebx, dword ptr [r14 + rax] 
sub ax, -32616 
and rsi, 0b1111111111111 # instrumentation
and dword ptr [r14 + rsi], -77 
mov cl, dl 
mov al, dl 
lea rax, qword ptr [rcx + rdx + 40938] 
sbb al, -25 
xchg esi, ebx 
sub al, sil 
or eax, 1374518464 
not rdx 
setbe dil 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rdi], dl 
and cl, cl 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
inc cl 
and rax, 0b1111111111111 # instrumentation
movzx cx, byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], dl 
or esi, 1 # instrumentation
and edx, esi # instrumentation
shr edx, 1 # instrumentation
div esi 
add al, 42 # instrumentation
adc rax, rax 
and rax, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rax] 
not al 
bts rdi, rbx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
add cl, -2 # instrumentation
cmovl ax, si 
add rsi, 109 
lea rsi, qword ptr [rsi] 
lea edx, qword ptr [rbx + rcx] 
inc rcx 
mov bx, -28748 
and cl, 78 
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], cl 
and rbx, 0b1111111111111 # instrumentation
cmovl cx, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmovnl edx, dword ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rbx] 
xor ax, -8186 
mov rbx, 5342692656799918209 
and rcx, 0b1111111111111 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rsi] 
neg si 
lea edx, qword ptr [rdi] 
xchg esi, esi 
and rcx, 0b1111111111111 # instrumentation
xor dx, word ptr [r14 + rcx] 
inc cx 
sbb bl, 48 
sbb cl, -55 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
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
