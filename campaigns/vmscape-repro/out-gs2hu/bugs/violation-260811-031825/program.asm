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
and rbx, 0b1111111111111 # instrumentation
xor ebx, dword ptr [r14 + rbx] 
lea rsi, qword ptr [rdx + rdx] 
seto cl 
mov dl, dl 
and rsi, 0b1111111111111 # instrumentation
cmp dx, word ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
adc word ptr [r14 + rsi], -94 
and rcx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rcx], 0b1000000000000000000000000000000 # instrumentation
bsr rsi, qword ptr [r14 + rcx] 
xor dl, dl 
and rax, 0b1111111111111 # instrumentation
neg word ptr [r14 + rax] 
sbb rbx, rcx 
and rdx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdx], cl 
xadd esi, esi 
add ax, -13158 
cmovnz di, di 
lea bx, qword ptr [rbx] 
and rax, 0b1111111111111 # instrumentation
and esi, 0b111 # instrumentation
bt dword ptr [r14 + rax], esi 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], -890383707 
setp cl 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], esi 
and rcx, 0b1111111111111 # instrumentation
btr word ptr [r14 + rcx], 4 
and rax, 0b1111111111111 # instrumentation
cmovnb rdi, qword ptr [r14 + rax] 
test ax, -6607 
cmp cl, 8 
and rsi, 0b1111111111111 # instrumentation
cmovle eax, dword ptr [r14 + rsi] 
xadd dl, bl 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rax] 
mov bx, -12590 
and rsi, 0b1111111111111 # instrumentation
sub ax, word ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], si 
cmovnl eax, edx 
bts edi, 11 
or cx, -22 
and rax, 0b1111111111111 # instrumentation
cmovnz rdi, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmovnle ebx, dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mov ecx, dword ptr [r14 + rax] 
dec rdx 
and rsi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rsi] 
add al, -105 
sub al, 22 
and rdx, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rdx] 
cmp ax, 20981 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1958229272 
cmovns ecx, ebx 
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
mov eax, -310745270 
lea di, qword ptr [rdi + rbx + 63041] 
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
cmp si, 23 
lea rdx, qword ptr [rsi] 
cmp al, cl 
movsx ebx, cl 
adc al, bl 
and rsi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rsi], 72 
and rsi, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rsi] 
lea edx, qword ptr [rax] 
or al, 63 
and rdx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdx] 
or esi, 0b1000000000000000000000000000000 # instrumentation
bsr esi, esi 
add dl, -117 # instrumentation
and rax, 0b1111111111111 # instrumentation
sbb dl, byte ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
not dword ptr [r14 + rcx] 
lea rdi, qword ptr [rdx + rdx + 44555] 
lea ebx, qword ptr [rsi + rsi + 48387] 
lea ecx, qword ptr [rsi + rcx] 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rdx, 0b1111111111111 # instrumentation
cmp ax, word ptr [r14 + rdx] 
cmp eax, -905996002 
xor cl, dl 
setno al 
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax] 
btr rdx, rdx 
and rbx, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
btc dword ptr [r14 + rbx], edx 
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], dl 
xor eax, 92 
and rsi, 0b1111111111111 # instrumentation
cmovl eax, dword ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], 8 
and rdi, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rdi], 4 
btr edi, edi 
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf rcx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rsi], bx 
imul bx 
add dl, -52 # instrumentation
and rdi, 0b1111111111111 # instrumentation
movsx edx, word ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
cmovle eax, dword ptr [r14 + rcx] 
sbb rsi, rdx 
lea edx, qword ptr [rdx + rcx + 51133] 
and rbx, 0b1111111111111 # instrumentation
movzx edx, byte ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsf rdi, qword ptr [r14 + rbx] 
and bl, 63 
lea edi, qword ptr [rdi] 
and rcx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rcx] 
and rdx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdx], bl 
movzx rbx, dil 
and al, dl 
and rax, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rax] 
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
