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
xor al, al 
movsx edi, cl 
mov cl, al 
sbb dl, bl 
lea rcx, qword ptr [rbx + rax] 
lea cx, qword ptr [rcx] 
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
and cx, 0b111 # instrumentation
lock btc word ptr [r14 + rdx], cx 
add dl, -93 # instrumentation
setns al 
sub dl, bl 
setz dl 
adc dl, al 
and rdi, 0b1111111111111 # instrumentation
adc esi, dword ptr [r14 + rdi] 
movzx ebx, al 
setnz al 
neg rax 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
btc rdi, rcx 
and rbx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rbx], edi 
and rdx, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rdx] 
and al, 2 
bt rax, rdi 
test eax, 1160670098 
and rbx, 0b1111111111111 # instrumentation
cmovnbe di, word ptr [r14 + rbx] 
inc cx 
and rdi, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rdi], si 
lea cx, qword ptr [rbx] 
lea ecx, qword ptr [rbx + rdi] 
mov al, al 
and rsi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rsi] 
cmovle dx, dx 
and rdx, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rdx] 
sbb al, cl 
movzx ebx, bl 
or al, -14 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsr ecx, dword ptr [r14 + rdx] 
add rcx, -29 
and rdx, 0b1111111111111 # instrumentation
cmovs bx, word ptr [r14 + rdx] 
or esi, 0b1000000000000000000000000000000 # instrumentation
bsf edx, esi 
and eax, 55 
xor al, -36 
imul esi 
add dl, -88 # instrumentation
and rbx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rbx], dl 
and rdx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
add ecx, dword ptr [r14 + rcx] 
btr rcx, 244 
mul eax 
xor cl, dl 
and rbx, 0b1111111111111 # instrumentation
cmovno esi, dword ptr [r14 + rbx] 
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
and rax, 0b1111111111111 # instrumentation
test dword ptr [r14 + rax], 109957297 
btr ebx, 71 
xor rdi, 5 
and rsi, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rsi] 
cmc  
and rdi, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rdi], -1 
sbb dl, 35 
sub al, -58 
and rax, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rax] 
lea ecx, qword ptr [rdx + rax] 
and rbx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rbx], -53 
xor al, al 
and rbx, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rbx], rcx 
lea eax, qword ptr [rdx + rax + 65299] 
bts rsi, rdi 
and rdx, 0b1111111111111 # instrumentation
cmovb si, word ptr [r14 + rdx] 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rbx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
sbb rsi, qword ptr [r14 + rdi] 
movzx ebx, dx 
inc dl 
lea ebx, qword ptr [rsi] 
adc ecx, edi 
and rcx, 0b1111111111111 # instrumentation
adc rdx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rax], bl 
add di, -34 
setnb dl 
and rax, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rax], eax 
cmovs rdi, rsi 
and rdx, 0b1111111111111 # instrumentation
adc dl, byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rbx] 
sbb cl, sil 
xor bl, bl 
and al, -60 
and rcx, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rcx] 
and bl, -69 
and rbx, 0b1111111111111 # instrumentation
xor cl, byte ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
xor rbx, qword ptr [r14 + rdx] 
cmovle rbx, rcx 
cmovnb edx, ecx 
and rcx, 0b1111111111111 # instrumentation
movsx ecx, byte ptr [r14 + rcx] 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rcx], sil 
movsx eax, sil 
and rdx, 0b1111111111111 # instrumentation
and rdx, 0b111 # instrumentation
btr qword ptr [r14 + rdx], rdx 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
cwd  
xchg eax, eax 
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
