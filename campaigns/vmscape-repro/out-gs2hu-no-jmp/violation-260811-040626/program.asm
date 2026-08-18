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
btc ax, dx 
xor dl, cl 
or eax, -1607671750 
and rax, 0b1111111111111 # instrumentation
cmovns ax, word ptr [r14 + rax] 
and ax, 6654 
xor dl, al 
and rdi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rdi] 
adc dl, dl 
test edx, -1514318118 
dec cl 
lea eax, qword ptr [rdx + rdx + 744] 
or rbx, 0b1000000000000000000000000000000 # instrumentation
bsf rcx, rbx 
add al, 43 # instrumentation
and rsi, 0b1111111111111 # instrumentation
adc cl, byte ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf dx, word ptr [r14 + rsi] 
btr rcx, 177 
lea rsi, qword ptr [rbx + rax + 49166] 
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add dl, -23 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovnz edx, dword ptr [r14 + rdi] 
lea ax, qword ptr [rbx + rdi] 
imul si 
and rsi, 0b1111111111111 # instrumentation
adc rdx, qword ptr [r14 + rsi] 
or al, 9 
lea dx, qword ptr [rcx] 
and rax, 0b1111111111111 # instrumentation
sbb cl, byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rbx], cl 
and rcx, 0b1111111111111 # instrumentation
adc edi, dword ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rsi], bx 
or cx, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv cx 
add bl, 67 # instrumentation
sbb al, al 
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rbx], -27 
and al, 113 
adc dl, -118 
and rcx, 0b1111111111111 # instrumentation
cmovb si, word ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
and bl, byte ptr [r14 + rax] 
xor al, 33 
mov dl, dl 
cmovno cx, dx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax 
neg rcx 
adc bl, cl 
bts esi, ebx 
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], dl 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], -79 
and dl, al 
or sil, 68 
adc edx, 115 
and rcx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], 57 
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
add bl, 100 # instrumentation
lea rdx, qword ptr [rsi + rsi + 42781] 
sbb cl, 104 
cmovnle dx, si 
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], -67 
or rdi, rdx 
and rax, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rax], 65 
add al, 93 
setns al 
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], eax 
inc esi 
lea ebx, qword ptr [rax + rsi + 39443] 
lea rdx, qword ptr [rcx + rcx + 42161] 
and rdi, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rdi] 
setbe cl 
mov al, cl 
bts ecx, 21 
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
xchg bl, bl 
and rsi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rsi], rcx 
and rdi, 0b1111111111111 # instrumentation
cmovp rax, qword ptr [r14 + rdi] 
lea cx, qword ptr [rcx + rdx + 30333] 
and rcx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rcx], eax 
inc cl 
cwd  
or eax, 0b1000000000000000000000000000000 # instrumentation
bsf edx, eax 
add dl, 81 # instrumentation
and rcx, 0b1111111111111 # instrumentation
adc al, byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
and ax, 0b111 # instrumentation
lock btr word ptr [r14 + rdx], ax 
and rcx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rcx] 
inc ax 
and rcx, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
or dl, byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 1 # instrumentation
and edx, dword ptr [r14 + rbx] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rbx] 
and dil, sil 
btr ebx, esi 
sbb bl, cl 
and rcx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rcx], 93 
lea rax, qword ptr [rcx] 
and rdi, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov bx, word ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rdx], al 
setle bl 
cmp al, 52 
and rcx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rcx], esi 
or cl, bl 
and rdi, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
btc dword ptr [r14 + rdi], edx 
add al, -24 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovns dx, word ptr [r14 + rsi] 
adc dl, 99 
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], -140411972 
lfence  
.macro.switch_g2h.main.attacker_1: nop qword ptr [rax + 0xff]
lfence  
.exit_attacker_gadget:
.function_attacker_measure:
.bb_attacker_measure.entry:
.macro.landing_h2g.attacker_2: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.measurement_end.attacker_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_attacker_measure:
.section .data.main
.test_case_exit:nop
