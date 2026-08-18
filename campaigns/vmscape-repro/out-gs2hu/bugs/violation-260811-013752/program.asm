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
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], 18 
cmpxchg cl, dl 
xadd dl, cl 
not cl 
xor ax, 11829 
or rax, -1261684131 
lea rbx, qword ptr [rbx] 
sbb si, -54 
not rdx 
cmp dil, 8 
and rdx, 0b1111111111111 # instrumentation
add word ptr [r14 + rdx], 71 
sbb al, -97 
or al, 104 
sub bl, al 
adc cx, -114 
adc ax, -4832 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rdx, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rdx], edx 
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], dl 
and dl, -110 
adc al, al 
sbb ax, 23026 
and rsi, 0b1111111111111 # instrumentation
setl byte ptr [r14 + rsi] 
cmp rax, 549043221 
cmovs rbx, rcx 
lea cx, qword ptr [rax] 
add bl, cl 
and rbx, 0b1111111111111 # instrumentation
or di, word ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovnz edi, dword ptr [r14 + rbx] 
lea ax, qword ptr [rax] 
test rdi, 1902245872 
and rdx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
and esi, 0b111 # instrumentation
lock bts dword ptr [r14 + rsi], esi 
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], -123082581 
cmpxchg bl, al 
and rsi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rsi], dl 
and rsi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rsi], ecx 
and rcx, 0b1111111111000 # instrumentation
lock bts word ptr [r14 + rcx], 2 
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 127 
cmovbe di, bx 
cmp cl, al 
bt dx, di 
add bl, 82 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovs cx, word ptr [r14 + rdi] 
sub eax, -663022952 
cmovbe cx, di 
sbb al, -56 
lea esi, qword ptr [rax + rbx + 13896] 
cmovno eax, edi 
lea rdx, qword ptr [rdx + rax] 
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
add al, 73 # instrumentation
adc dl, al 
sbb dx, bx 
sahf  
and rbx, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rbx], si 
mov dl, al 
and rcx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
add edi, dword ptr [r14 + rsi] 
or al, -118 
and rcx, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rcx], 3 
sbb dl, 15 
xor rax, 1536536929 
cmovnle cx, bx 
add rax, -1761748413 
and rcx, 0b1111111111111 # instrumentation
cmovnl cx, word ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rsi], 3 
movzx edi, di 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rdx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
adc al, byte ptr [r14 + rax] 
lea rcx, qword ptr [rax + rsi + 24320] 
and rcx, 0b1111111111111 # instrumentation
cmovp edx, dword ptr [r14 + rcx] 
btr ecx, edx 
add ecx, esi 
or bx, 3 
cmovnb edx, esi 
and rcx, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rcx] 
lea rdi, qword ptr [rsi + rbx] 
and rdi, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdi], edi 
and rbx, 0b1111111111111 # instrumentation
cmovb rcx, qword ptr [r14 + rbx] 
movzx di, al 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], -3 
and rdi, 0b1111111111111 # instrumentation
xor rdi, qword ptr [r14 + rdi] 
xchg bl, al 
and rsi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rsi], 118 
and rdi, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
bt word ptr [r14 + rdi], di 
cmovbe dx, cx 
and rsi, 0b1111111111111 # instrumentation
not word ptr [r14 + rsi] 
cmp ax, ax 
and rsi, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdi 
test dil, -57 
or ax, -28332 
and rax, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rax], 3 
cmovb ecx, ebx 
and al, 20 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], rax 
xadd dl, cl 
cmp edi, 10 
setnbe dl 
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
