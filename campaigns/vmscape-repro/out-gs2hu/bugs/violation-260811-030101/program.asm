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
mul cl 
xor si, bx 
or rax, 102 
and rcx, 0b1111111111111 # instrumentation
and word ptr [r14 + rcx], dx 
mov al, cl 
and rcx, 0b1111111111111 # instrumentation
sub ebx, dword ptr [r14 + rcx] 
sbb dil, dil 
cmovnl rcx, rcx 
sub ax, 10110 
and rcx, 0b1111111111111 # instrumentation
cmovnbe rsi, qword ptr [r14 + rcx] 
not dl 
and rdx, 0b1111111111000 # instrumentation
lock bts word ptr [r14 + rdx], 1 
and rdx, 0b1111111111111 # instrumentation
sub dx, word ptr [r14 + rdx] 
mov dx, bx 
xchg rax, rdi 
cmp rax, rax 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rcx, 0b1111111111111 # instrumentation
bt word ptr [r14 + rcx], 3 
or cl, 3 
sbb dl, bl 
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], -124 
and rax, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rax], 2 
and rdi, 0b1111111111111 # instrumentation
movzx rcx, word ptr [r14 + rdi] 
adc rax, 1877964395 
imul rsi, rbx 
bts ax, ax 
add bl, 7 # instrumentation
setb bl 
and rsi, 0b1111111111111 # instrumentation
cmovns edx, dword ptr [r14 + rsi] 
xchg cx, ax 
and rbx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rbx], dil 
and rdi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdi], bl 
add al, cl 
or bl, dl 
and rdx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdx], -34 
and rdx, rsi 
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], eax 
neg dx 
and rdi, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rcx], -100 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rsi], dil 
bts rdx, 30 
adc rsi, -20 
and rcx, 0b1111111111111 # instrumentation
movsx esi, byte ptr [r14 + rcx] 
setnbe bl 
lea rcx, qword ptr [rdx] 
lea ecx, qword ptr [rax + rcx + 34409] 
sets dl 
and rax, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rax], 101 
sbb rax, rdi 
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
add dl, 70 # instrumentation
setz dl 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rax] 
lea si, qword ptr [rdx + rax + 20929] 
movsx rdi, cx 
and dil, al 
sub rax, -859830417 
and rcx, 0b1111111111111 # instrumentation
cmovnbe ebx, dword ptr [r14 + rcx] 
inc eax 
and rdi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdi], dx 
sbb rax, -47 
setno bl 
test ax, -9717 
and rax, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rax], rdi 
and rdi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdi], dil 
or bl, bl 
or rax, -1137604567 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
add al, -61 # instrumentation
and rsi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rsi], 63 
and rdi, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rdi], 7 
and rdi, 0b1111111111111 # instrumentation
movsx eax, byte ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rdx], edi 
setp al 
nop bx 
bts edi, 227 
and rsi, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rsi], 7 
lea rdi, qword ptr [rdx + rax + 37289] 
xor cx, dx 
setnb dl 
and rsi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rsi] 
bt eax, edi 
add cl, 76 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovns ecx, dword ptr [r14 + rdx] 
sub ax, -111 
and rcx, 0b1111111111111 # instrumentation
and al, dl 
xchg cl, sil 
and rbx, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rbx], rdx 
and rbx, 0b1111111111111 # instrumentation
cmovbe edx, dword ptr [r14 + rbx] 
sbb al, cl 
adc rax, 1927361376 
lea bx, qword ptr [rdi + rsi] 
and al, 97 
and rsi, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bts qword ptr [r14 + rsi], rsi 
add dl, -99 # instrumentation
and rax, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
or eax, dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
not dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
and esi, dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmovp si, word ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], -108 
and rsi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rsi], al 
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
