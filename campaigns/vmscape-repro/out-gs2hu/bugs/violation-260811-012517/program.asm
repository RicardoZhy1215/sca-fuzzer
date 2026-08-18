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
add bl, 38 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovnz bx, dx 
xchg cx, ax 
sub ax, 18546 
test cl, 34 
lea rsi, qword ptr [rbx + rbx + 10120] 
lea ax, qword ptr [rdi + rdx + 32331] 
movsx cx, dl 
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
and rbx, 0b1111111111111 # instrumentation
not qword ptr [r14 + rbx] 
btc dx, 6 
cmovnb eax, esi 
sub al, -2 
and rax, 0b1111111111111 # instrumentation
cmovo edx, dword ptr [r14 + rax] 
cmovl si, bx 
bts edx, 239 
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 1 # instrumentation
and dx, word ptr [r14 + rcx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rcx] 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
lea rax, qword ptr [rcx + rbx] 
and rax, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rax], sil 
and rdx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rdx], -15 
and rax, 0b1111111111111 # instrumentation
adc dil, byte ptr [r14 + rax] 
lea ebx, qword ptr [rax + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmovnl esi, dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
or al, byte ptr [r14 + rax] 
imul al 
add ax, 4828 
lea si, qword ptr [rcx] 
and rbx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rbx], 40 
and rdx, 0b1111111111111 # instrumentation
test word ptr [r14 + rdx], 23420 
sub bl, dl 
and rbx, 0b1111111111111 # instrumentation
cmovnb dx, word ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rsi], sil 
and rcx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rcx], bl 
and rbx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rbx], cl 
and rdx, 0b1111111111111 # instrumentation
sbb ebx, dword ptr [r14 + rdx] 
and al, 114 
setno bl 
or rsi, 0b1000000000000000000000000000000 # instrumentation
bsf rcx, rsi 
and rsi, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rsi], ecx 
sbb si, dx 
and rax, 0b1111111111111 # instrumentation
not word ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
movzx di, byte ptr [r14 + rbx] 
or bl, 47 
and rsi, 0b1111111111111 # instrumentation
mul word ptr [r14 + rsi] 
dec esi 
add rdx, rdi 
and rdx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdx], dl 
btc cx, 11 
add al, 39 # instrumentation
setnp dil 
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
and rcx, 0b1111111111111 # instrumentation
sub sil, byte ptr [r14 + rcx] 
cmpxchg ebx, edx 
cmovnbe rbx, rbx 
sbb sil, -127 
or esi, edi 
and al, 34 
mul si 
add cl, -49 # instrumentation
clc  
lea edx, qword ptr [rbx + rsi + 3715] 
cmovnz ebx, eax 
or rdx, 0b1000000000000000000000000000000 # instrumentation
bsr rbx, rdx 
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
add bl, -96 # instrumentation
adc cl, bl 
lea si, qword ptr [rax + rbx] 
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
mov dl, cl 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
add dl, 107 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rcx], dl 
lea bx, qword ptr [rdx + rdi + 20158] 
mov si, ax 
and rcx, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rcx] 
cmp al, -9 
and rdi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rdi], 34 
sbb cl, 55 
and rdx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdx], 63 
xchg di, dx 
and rbx, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rbx], 7 
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], sil 
and rax, 0b1111111111111 # instrumentation
mul word ptr [r14 + rax] 
add al, -8 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rbx], di 
cmovs ax, si 
and rbx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rbx], eax 
mov cl, cl 
test dil, bl 
and rcx, 0b1111111111000 # instrumentation
and ebx, 0b111 # instrumentation
lock btc dword ptr [r14 + rcx], ebx 
add cl, al 
and rcx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rcx], al 
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 33 
and rax, 0b1111111111111 # instrumentation
xor rsi, qword ptr [r14 + rax] 
sub cl, -75 
lea bx, qword ptr [rdx] 
sbb rax, 2093373559 
and rdi, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 1 # instrumentation
and dx, word ptr [r14 + rsi] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rsi] 
add bl, 44 # instrumentation
setnbe cl 
and rax, 0b1111111111111 # instrumentation
add word ptr [r14 + rax], 68 
and rdx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rdx] 
or cl, bl 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rcx], bx 
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
