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
and rdx, 0b1111111111000 # instrumentation
and eax, 0b111 # instrumentation
lock btr dword ptr [r14 + rdx], eax 
xchg bx, ax 
adc dl, dil 
lea edx, qword ptr [rbx + rax] 
lea edi, qword ptr [rax + rdi + 38700] 
bt ecx, ecx 
add bl, -38 # instrumentation
setbe al 
cmovnp rcx, rdx 
lea edi, qword ptr [rbx] 
inc rax 
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000000000000000 # instrumentation
bsf di, word ptr [r14 + rcx] 
lea rcx, qword ptr [rdi + rsi] 
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], sil 
cmovnl si, si 
test dil, 93 
mov bl, al 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add al, 66 # instrumentation
cmovz si, cx 
and rdx, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
sbb rsi, qword ptr [r14 + rax] 
btc bx, 10 
and rdx, 0b1111111111111 # instrumentation
sbb di, word ptr [r14 + rdx] 
not al 
and rcx, 0b1111111111111 # instrumentation
adc rdi, qword ptr [r14 + rcx] 
lea edi, qword ptr [rsi + rdi] 
add cl, cl 
xor dx, bx 
and rbx, 0b1111111111000 # instrumentation
and eax, 0b111 # instrumentation
lock btc dword ptr [r14 + rbx], eax 
and rax, 0b1111111111111 # instrumentation
test qword ptr [r14 + rax], 1464081730 
test dil, -63 
inc cx 
mov rcx, rax 
imul edx, edx, -94 
and rcx, 0b1111111111111 # instrumentation
adc si, word ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
add word ptr [r14 + rdx], bx 
and rdx, 0b1111111111111 # instrumentation
movzx rdx, word ptr [r14 + rdx] 
sub al, dl 
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000000000000000 # instrumentation
bsr ax, word ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rdi] 
cwd  
and rcx, 0b1111111111111 # instrumentation
imul edx, dword ptr [r14 + rcx], 103 
mov dl, dl 
lea ecx, qword ptr [rdx] 
and rcx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rcx] 
lea rdx, qword ptr [rdi] 
cmp cl, cl 
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], edx 
and rdi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rdi], 97 
and rdi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rdi] 
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
or dl, al 
and rbx, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rbx], eax 
test ax, 20172 
cmovo edi, edx 
sbb rdi, rcx 
xor bl, 92 
lea edi, qword ptr [rdx] 
cbw  
and rcx, 0b1111111111111 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rbx], -6 
cmovnbe ax, ax 
and rdx, 0b1111111111111 # instrumentation
imul esi, dword ptr [r14 + rdx] 
or rsi, -52 
adc al, 100 
setnp sil 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
add al, 81 # instrumentation
adc ax, 18946 
add al, al 
xor dil, 96 
and rsi, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rsi] 
cmovle rsi, rbx 
setl cl 
setz bl 
sub dl, dl 
and rcx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rcx], cx 
and rdi, 0b1111111111111 # instrumentation
cmovbe ax, word ptr [r14 + rdi] 
test dl, 1 
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], al 
xor eax, 1827495314 
and rsi, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rsi], rsi 
and rdi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdi], bx 
and rbx, 0b1111111111111 # instrumentation
add cx, word ptr [r14 + rbx] 
adc sil, 25 
and cl, cl 
and rdx, 0b1111111111111 # instrumentation
movzx dx, byte ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdi], -98 
and rdx, -42 
and rcx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], -103 
add al, dl 
and rsi, 0b1111111111111 # instrumentation
and bl, bl 
neg dil 
btc rsi, rsi 
and rdx, 0b1111111111111 # instrumentation
cmp bl, byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rsi], dl 
mov ebx, edx 
or al, al 
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
