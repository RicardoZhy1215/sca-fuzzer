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
add dl, -26 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
lea ecx, qword ptr [rax + rbx] 
cmovbe di, di 
and rdx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdx], -28 
or bl, 0b1000 # instrumentation
and bl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv bl 
add cl, 26 # instrumentation
cmovnle ax, cx 
setnle dl 
cmovnl ax, si 
and rsi, 0b1111111111111 # instrumentation
not qword ptr [r14 + rsi] 
add bl, bl 
or eax, 1444615826 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rdx], ecx 
adc esi, 16 
imul rdx, rdi, -68 
add eax, -699507749 
sub al, -110 
cmp eax, 849273888 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
not cl 
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], dl 
not ax 
setz sil 
and rsi, 0b1111111111111 # instrumentation
cmovo ebx, dword ptr [r14 + rsi] 
neg cl 
and rax, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rax], bl 
and cl, -83 
and rsi, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rsi], dl 
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsf edx, dword ptr [r14 + rdi] 
add bl, 65 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdi] 
sbb cl, bl 
inc dl 
and rsi, 0b1111111111111 # instrumentation
and ebx, 0b111 # instrumentation
bt dword ptr [r14 + rsi], ebx 
add dl, bl 
neg edx 
lea rbx, qword ptr [rax] 
and rdi, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdi] 
test eax, -1542387173 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
cwde  
and rdi, 0b1111111111111 # instrumentation
cmp cl, byte ptr [r14 + rdi] 
or bl, cl 
neg bl 
lea di, qword ptr [rsi + rbx] 
and bl, bl 
and rcx, 0b1111111111000 # instrumentation
and rbx, 0b111 # instrumentation
lock btr qword ptr [r14 + rcx], rbx 
xor esi, ebx 
and rdx, 0b1111111111111 # instrumentation
movsx rdx, word ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], 7 
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx] 
and eax, -1992593336 
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
setnbe sil 
sbb dil, 118 
lea rdi, qword ptr [rax + rdi + 44523] 
xadd esi, eax 
test ax, -3941 
or dl, al 
cmp rdi, 3 
cmovbe rcx, rcx 
cbw  
and cl, 89 
and rcx, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rcx], 2 
inc bx 
and rdi, 0b1111111111111 # instrumentation
cmovb rdi, qword ptr [r14 + rdi] 
sbb cl, al 
cmp rax, 1404236627 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rbx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rbx], bl 
mov bl, al 
cmp rax, -1326042961 
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], rdi 
lea ecx, qword ptr [rdx] 
bt bx, cx 
test ebx, edi 
lea cx, qword ptr [rbx] 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], -14 
and rdx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdx], cl 
and rbx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rbx], cl 
and rdx, 0b1111111111111 # instrumentation
cmovp bx, word ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
and eax, dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
sbb ebx, dword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmovle cx, word ptr [r14 + rdi] 
xchg edx, ecx 
xadd al, cl 
and rcx, 0b1111111111111 # instrumentation
not word ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmovp eax, dword ptr [r14 + rcx] 
adc bl, bl 
and rcx, 0b1111111111111 # instrumentation
btr word ptr [r14 + rcx], 3 
and rbx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rbx], ecx 
and rbx, 0b1111111111111 # instrumentation
cmovb rcx, qword ptr [r14 + rbx] 
test dil, 118 
lea rax, qword ptr [rcx + rbx + 64059] 
or rax, -293326692 
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov dil, byte ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rcx], ebx 
and esi, esi 
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
