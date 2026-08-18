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
movsx ecx, bl 
and rbx, 0b1111111111111 # instrumentation
and word ptr [r14 + rbx], 126 
xchg eax, eax 
and rcx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rcx], cl 
and rdx, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rdx] 
not ecx 
bts rdi, rax 
adc dx, dx 
and rbx, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rbx] 
neg cx 
and rsi, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rsi], edx 
xor cl, al 
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], di 
sets sil 
and rsi, 0b1111111111111 # instrumentation
cmovno ax, word ptr [r14 + rsi] 
xadd sil, dil 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rbx, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rbx], 5 
btc dx, 35 
sub si, 32 
and rax, 0b1111111111111 # instrumentation
cmovp di, word ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rdi], 90 
and rsi, 0b1111111111111 # instrumentation
cmovnb rcx, qword ptr [r14 + rsi] 
bt rdi, rdx 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
test rdi, rdi 
sbb dl, -57 
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx] 
sbb al, cl 
and rbx, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rbx] 
or al, dl 
sbb ax, 21871 
and rcx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rcx], -112 
and rdi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdi], bl 
and rdi, 0b1111111111111 # instrumentation
cmovo rbx, qword ptr [r14 + rdi] 
or eax, 1080003149 
and rdi, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rdi], 1 
add bl, -14 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovle si, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdi], di 
and rax, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rax], -45 
add dil, 27 
and rbx, 0b1111111111111 # instrumentation
sub cx, word ptr [r14 + rbx] 
cmovnz si, si 
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], dl 
and rdx, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rbx], dl 
cmovb esi, ebx 
xor cl, dl 
and rdi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdi] 
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
xchg bl, dl 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
inc rax 
and rdx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdx], ecx 
sub edi, 127 
imul dl 
imul dl 
lea bx, qword ptr [rdx + rcx + 4586] 
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], rbx 
bts ebx, 179 
adc al, 69 
xchg rdx, rbx 
and rsi, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rsi] 
add dl, -30 # instrumentation
cmovz di, dx 
inc rsi 
and rdx, 0b1111111111111 # instrumentation
cmovnl eax, dword ptr [r14 + rdx] 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
cmp rdi, rax 
adc eax, -807419970 
and rax, 0b1111111111111 # instrumentation
cmp dl, byte ptr [r14 + rax] 
setno bl 
and al, 81 
and rdi, 0b1111111111111 # instrumentation
inc word ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmovz di, word ptr [r14 + rax] 
add eax, 1495074517 
and rdi, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rdi] 
add dl, 51 
and rsi, 0b1111111111111 # instrumentation
cmovnl edx, dword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx] 
cmp cl, -75 
cmp cl, dl 
and rcx, 0b1111111111111 # instrumentation
movzx eax, word ptr [r14 + rcx] 
lea esi, qword ptr [rsi] 
and rsi, 0b1111111111111 # instrumentation
cmovle si, word ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
and sil, byte ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
and di, 0b111 # instrumentation
lock bts word ptr [r14 + rdx], di 
and rbx, 0b1111111111111 # instrumentation
sub word ptr [r14 + rbx], ax 
mov rdx, 5180798297974340210 
mul cl 
test rax, -2104880691 
movsx ax, al 
cwd  
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], 7 
and rcx, 0b1111111111111 # instrumentation
cmp ebx, dword ptr [r14 + rcx] 
and rbx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rbx], 7 
xadd ax, si 
sbb dl, dl 
adc dl, dl 
adc rdx, rax 
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
