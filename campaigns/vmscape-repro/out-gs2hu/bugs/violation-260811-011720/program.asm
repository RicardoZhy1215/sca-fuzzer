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
add al, 18 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovnbe ecx, edi 
movsx si, cl 
bswap rsi 
and sil, -68 
and rdx, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdx] 
adc ebx, ecx 
lea cx, qword ptr [rbx + rbx + 36929] 
lea rsi, qword ptr [rdi + rax + 59149] 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rdx], edx 
lea cx, qword ptr [rdx] 
inc rbx 
xadd dl, dl 
and rsi, 0b1111111111111 # instrumentation
cmovnl dx, word ptr [r14 + rsi] 
xor di, -67 
test al, cl 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and edi, ecx 
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 1 # instrumentation
and edx, dword ptr [r14 + rsi] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
xor bx, word ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rax 
or al, dl 
and rbx, 0b1111111111111 # instrumentation
cmovo si, word ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf rax, qword ptr [r14 + rax] 
setnz bl 
lea dx, qword ptr [rsi] 
and rdx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdx], edx 
and rbx, 0b1111111111111 # instrumentation
sbb cl, byte ptr [r14 + rbx] 
cmp al, 3 
add dl, bl 
btr edi, 102 
add al, 25 # instrumentation
and rsi, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rdi] 
lea edx, qword ptr [rcx] 
and rdx, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
not qword ptr [r14 + rbx] 
cmpxchg rax, rbx 
adc rcx, 65 
sbb si, si 
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rax 
test rbx, 222024069 
inc edi 
and rbx, 0b1111111111111 # instrumentation
movsx esi, byte ptr [r14 + rbx] 
not rdx 
cmovz ebx, ecx 
and rsi, 0b1111111111111 # instrumentation
cmovle edx, dword ptr [r14 + rsi] 
and ax, 32550 
cmp eax, -836970035 
setnp bl 
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
add dl, -19 # instrumentation
sets dl 
cmp dl, cl 
movzx rdi, si 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
lea rcx, qword ptr [rax + rax + 45448] 
mov ax, -31239 
lea rcx, qword ptr [rdx + rdi] 
xchg rax, rdx 
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsf eax, ecx 
or dx, -33 
bt dx, 180 
xor ecx, -119 
add dl, al 
cmovle rsi, rdi 
lea si, qword ptr [rcx] 
dec edi 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rdx, 0b1111111111111 # instrumentation
bts word ptr [r14 + rdx], 6 
add bl, 40 # instrumentation
cmovns si, bx 
and rbx, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rbx], 4 
xor bl, -125 
and rbx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rbx], 92 
add cl, -69 # instrumentation
and rdi, 0b1111111111111 # instrumentation
movsx esi, word ptr [r14 + rdi] 
lahf  
and rax, 0b1111111111111 # instrumentation
test qword ptr [r14 + rax], rax 
and rsi, 0b1111111111111 # instrumentation
and word ptr [r14 + rsi], cx 
lea rax, qword ptr [rsi + rdi + 40676] 
lea edx, qword ptr [rax] 
and rsi, 0b1111111111111 # instrumentation
cmovnp rbx, qword ptr [r14 + rsi] 
add al, 24 # instrumentation
setbe bl 
and rdi, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmovnbe rdi, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
not qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add al, 53 # instrumentation
sbb cx, 93 
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi] 
sub dx, -108 
add al, cl 
and rdx, 0b1111111111111 # instrumentation
cmovo dx, word ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx] 
setnbe dl 
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], ecx 
sbb bl, cl 
lea edi, qword ptr [rsi + rsi] 
movzx esi, bx 
and rdx, 0b1111111111111 # instrumentation
cmovbe edx, dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov sil, byte ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdx], -41 
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
