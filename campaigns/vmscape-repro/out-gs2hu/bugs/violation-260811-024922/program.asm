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
add cl, -97 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovle ax, di 
cmovb rdi, rcx 
cmovp ax, ax 
adc ax, ax 
setnbe al 
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdx 
adc ax, 20206 
and rsi, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bt word ptr [r14 + rsi], si 
xchg dx, ax 
and rsi, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
bts word ptr [r14 + rsi], di 
test dil, dl 
movsx rax, dl 
xchg rdi, rax 
bt rsi, rcx 
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], bl 
xor rax, 331508014 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rcx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rcx], edx 
and rax, 0b1111111111111 # instrumentation
and byte ptr [r14 + rax], cl 
and rdx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdx], 99 
and rdi, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rdi], di 
and rdx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rdx], 5 
add dl, -107 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmp esi, dword ptr [r14 + rdi] 
dec rbx 
cmovle ebx, ebx 
lea rdx, qword ptr [rax] 
and rdx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rdx], 127 
cmovo ecx, ebx 
xor rax, 978093742 
setnbe bl 
and rdi, 0b1111111111111 # instrumentation
movzx edi, byte ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rcx], dl 
lea edx, qword ptr [rcx + rbx + 50499] 
and rdi, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rdi] 
xor eax, -51 
and rdx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rdx], 57 
mov cl, -114 
lea dx, qword ptr [rdi] 
cmp rdi, -20 
or sil, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv sil 
add cl, 31 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnle edx, dword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rcx] 
or bx, si 
and rdi, 0b1111111111111 # instrumentation
cmovno edx, dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rsi 
movsx rbx, cx 
test di, 9352 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 98 
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
and rdi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rdi], -5 
cmovnl bx, si 
sbb dl, 22 
bt ecx, ecx 
add cl, cl 
sbb dl, sil 
or dl, -85 
and rax, 0b1111111111111 # instrumentation
cmovnp cx, word ptr [r14 + rax] 
and rcx, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rcx] 
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsf edi, ecx 
add bl, 70 # instrumentation
lea rax, qword ptr [rax + rsi] 
sbb rbx, rcx 
adc ecx, esi 
lea si, qword ptr [rsi + rbx + 27242] 
test ax, 22591 
and rsi, 0b1111111111111 # instrumentation
cmovnz rsi, qword ptr [r14 + rsi] 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
inc dl 
btc ebx, edi 
bt si, 127 
and rbx, 0b1111111111111 # instrumentation
or rbx, qword ptr [r14 + rbx] 
adc dl, cl 
and rbx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rsi], 96 
and rbx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rbx], -32 
and rsi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rsi], cx 
neg dl 
xor bl, al 
and rdi, 0b1111111111111 # instrumentation
inc word ptr [r14 + rdi] 
lea rdx, qword ptr [rdi] 
and rsi, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rsi], eax 
lea edi, qword ptr [rsi + rax] 
and rcx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rcx], ecx 
and rdi, 0b1111111111111 # instrumentation
cmovp edi, dword ptr [r14 + rdi] 
cmovo rsi, rbx 
and rcx, 0b1111111111111 # instrumentation
and rdx, 0b111 # instrumentation
bt qword ptr [r14 + rcx], rdx 
cmovz rdi, rdi 
and rdx, 0b1111111111000 # instrumentation
xchg word ptr [r14 + rdx], cx 
and rcx, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmp bl, byte ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
or dil, byte ptr [r14 + rdx] 
cmp bl, al 
and rax, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
lea rax, qword ptr [rdi] 
and rax, 0b1111111111111 # instrumentation
sub dl, byte ptr [r14 + rax] 
lea cx, qword ptr [rbx] 
adc ebx, edi 
movsx bx, dil 
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
