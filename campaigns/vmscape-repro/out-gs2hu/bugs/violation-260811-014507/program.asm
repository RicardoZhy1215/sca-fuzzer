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
movsx ebx, dx 
sub cl, cl 
or dl, bl 
mov bl, 32 
or cx, 3 
and rbx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rbx] 
setle bl 
adc sil, -19 
test rsi, rcx 
lea rax, qword ptr [rdi] 
and dl, sil 
sub rdx, 43 
and rcx, 0b1111111111111 # instrumentation
sub ecx, dword ptr [r14 + rcx] 
xor ax, -13895 
lea bx, qword ptr [rax] 
and rbx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rbx], edx 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000000000000000 # instrumentation
bsr di, word ptr [r14 + rax] 
xchg si, ax 
and rdi, 0b1111111111111 # instrumentation
btc word ptr [r14 + rdi], 3 
and rdx, 0b1111111111111 # instrumentation
add esi, dword ptr [r14 + rdx] 
mov rdi, rdx 
imul si, cx, -113 
and rax, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rax], 5 
and rsi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rsi], esi 
and rbx, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rbx] 
add dl, -76 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnp esi, dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], bl 
cmovbe edx, ecx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
inc ecx 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax 
sub al, bl 
and eax, 686424702 
lea cx, qword ptr [rdx + rdi] 
sub eax, 1341285208 
cmovnp rsi, rsi 
and rsi, 0b1111111111111 # instrumentation
adc eax, dword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdx], bl 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 57 
and rsi, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], 65 
and rsi, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rsi], -11 
and rsi, 0b1111111111111 # instrumentation
sbb rdx, qword ptr [r14 + rsi] 
cmovbe rdi, rbx 
or bl, -103 
and dl, 15 
and rcx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rcx], al 
and rax, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rax] 
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
inc cx 
and rcx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], 49 
cmovns rcx, rsi 
sbb dil, bl 
and rax, 0b1111111111111 # instrumentation
and ecx, dword ptr [r14 + rax] 
cmp dl, 91 
adc al, -8 
imul rcx, rax, -6 
sbb dl, al 
sbb cl, cl 
sub cl, 126 
cbw  
and rdi, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rdi] 
xor al, 7 
or rdx, 0b1000000000000000000000000000000 # instrumentation
bsf rsi, rdx 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rax, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rax], sil 
not dil 
test eax, 1094302109 
adc rcx, rsi 
and rbx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rbx], cx 
and rdx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rdx], si 
and rdx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdx], -92 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rcx], ecx 
or ax, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv ax 
and rdi, 0b1111111111111 # instrumentation
add bx, word ptr [r14 + rdi] 
sub eax, esi 
and cl, dl 
and rcx, 0b1111111111111 # instrumentation
movzx ecx, byte ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
and eax, 0b111 # instrumentation
btc dword ptr [r14 + rcx], eax 
and rcx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rcx], dl 
and rdi, 0b1111111111111 # instrumentation
cmovnz edx, dword ptr [r14 + rdi] 
add si, 74 
bt edi, edi 
and rsi, 0b1111111111111 # instrumentation
cmp al, byte ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi] 
and cl, dl 
and rcx, 0b1111111111111 # instrumentation
cmovb rdx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 0b1000000000000000 # instrumentation
bsf si, word ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rbx], bl 
cmovno rcx, rcx 
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 0 
and rax, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdi], bx 
cmp dl, cl 
and rdx, 0b1111111111111 # instrumentation
sbb edx, dword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rsi], si 
lea rcx, qword ptr [rcx] 
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
