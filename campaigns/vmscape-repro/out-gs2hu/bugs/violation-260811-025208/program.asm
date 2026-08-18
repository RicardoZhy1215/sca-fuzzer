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
test eax, -1970177222 
and cl, -98 
and rcx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rcx], 68 
cmp bl, cl 
test edi, edx 
cmp ax, -29645 
lea rdi, qword ptr [rbx + rdx + 23763] 
setbe bl 
xchg bx, ax 
mul rcx 
and rbx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rbx], al 
sbb dl, dil 
lea dx, qword ptr [rcx + rdx + 30783] 
xchg ebx, edi 
lea eax, qword ptr [rsi + rsi + 7769] 
inc di 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rdi, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bts qword ptr [r14 + rdi], rsi 
add rax, 1370449134 
test rax, -1100858688 
xchg rcx, rcx 
sahf  
and rax, 0b1111111111111 # instrumentation
add byte ptr [r14 + rax], al 
and rax, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
btr word ptr [r14 + rax], di 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 58 
and rax, 0b1111111111111 # instrumentation
cmovnl bx, word ptr [r14 + rax] 
mov edi, ebx 
sub rdi, -57 
and rax, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rax], bl 
mov eax, ebx 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], 28 
and rcx, 0b1111111111111 # instrumentation
cmovnbe si, word ptr [r14 + rcx] 
and cl, 108 
and rdx, 0b1111111111111 # instrumentation
cmovnle si, word ptr [r14 + rdx] 
or bl, bl 
or cl, al 
adc cl, -12 
nop dx 
movsx di, al 
bts rbx, rdx 
and rsi, 0b1111111111000 # instrumentation
and ax, 0b111 # instrumentation
lock btc word ptr [r14 + rsi], ax 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], -102 
cmovnb esi, ecx 
and rbx, 0b1111111111111 # instrumentation
or rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovnbe ecx, dword ptr [r14 + rbx] 
btr dx, 106 
or dl, 0 
and rcx, 0b1111111111111 # instrumentation
mov dl, byte ptr [r14 + rcx] 
lea bx, qword ptr [rsi + rdx + 55858] 
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
xor dil, -60 
setnz al 
and rdx, 0b1111111111111 # instrumentation
add dil, byte ptr [r14 + rdx] 
adc dl, 119 
sbb eax, -2081259334 
adc rcx, rdx 
and rdx, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rdx], 2 
add dl, 75 # instrumentation
lea edx, qword ptr [rax + rsi + 63930] 
and rdx, 0b1111111111111 # instrumentation
cmovnl esi, dword ptr [r14 + rdx] 
cmovl edx, edi 
and cl, -88 
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx] 
lea ax, qword ptr [rsi + rsi] 
cmovnp rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
movsx si, byte ptr [r14 + rdx] 
lea dx, qword ptr [rdi + rdi + 63575] 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rdx, 0b1111111111111 # instrumentation
and edi, dword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rdi], bx 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rdi] 
add cl, 64 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovb rdx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add ecx, dword ptr [r14 + rbx] 
add rax, -174512277 
and rcx, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rbx] 
and dil, -52 
lea dx, qword ptr [rdx] 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rsi] 
add cl, 13 # instrumentation
mov cl, bl 
adc cl, -51 
and rax, 0b1111111111111 # instrumentation
cmovnz edi, dword ptr [r14 + rax] 
test ax, -3757 
and rcx, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rcx], ebx 
and rcx, 0b1111111111111 # instrumentation
movzx rdx, word ptr [r14 + rcx] 
or rsi, 0b1000000000000000000000000000000 # instrumentation
bsr rbx, rsi 
imul cl 
sbb bl, al 
sahf  
and rdi, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rdi] 
movsx ecx, bl 
and rsi, 0b1111111111111 # instrumentation
cmovns edi, dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
imul eax, dword ptr [r14 + rsi] 
add bl, 55 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovbe esi, dword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
movzx rdi, word ptr [r14 + rax] 
seto cl 
not cx 
and rcx, 0b1111111111111 # instrumentation
cmovnb edx, dword ptr [r14 + rcx] 
cmovbe bx, dx 
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
