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
add dl, -107 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
mov al, bl 
adc al, 11 
sub esi, 41 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
or al, bl 
or cx, cx 
and rdx, 0b1111111111111 # instrumentation
mov ebx, dword ptr [r14 + rdx] 
sub si, -13 
and rsi, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rsi], ax 
sub ax, 78 
cmovnz dx, dx 
movsx si, al 
and rbx, 0b1111111111111 # instrumentation
or dl, byte ptr [r14 + rbx] 
cmovp bx, di 
or sil, 126 
and rax, 0b1111111111111 # instrumentation
cmp ebx, dword ptr [r14 + rax] 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add al, -102 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rcx] 
adc al, 79 
xchg bl, al 
or bl, 0b1000 # instrumentation
and bl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv bl 
and rcx, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rcx], 3 
and rdx, 0b1111111111111 # instrumentation
sub rcx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovp bx, word ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], eax 
lea eax, qword ptr [rax] 
lea rdx, qword ptr [rdx] 
setnb dl 
cmovns rsi, rax 
and rdi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdi], -64 
and rax, 0b1111111111111 # instrumentation
mov dl, byte ptr [r14 + rax] 
cmp cl, bl 
and rbx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rbx], cl 
and rdx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdx], edi 
and rdi, 0b1111111111111 # instrumentation
cmovnle edx, dword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovb rbx, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bts word ptr [r14 + rsi], si 
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], bl 
and rcx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rcx] 
cmovb edx, eax 
and rsi, 0b1111111111111 # instrumentation
cmovnle dx, word ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rbx], al 
and rax, 0b1111111111111 # instrumentation
cmovnb ax, word ptr [r14 + rax] 
adc bl, -106 
and rbx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], sil 
and rbx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rbx], ebx 
bswap rbx 
lea ebx, qword ptr [rbx + rsi + 41828] 
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
add cl, -21 # instrumentation
sbb bl, 31 
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl 
and rcx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rcx] 
imul ebx, esi 
cmp al, bl 
cmovle rbx, rdx 
cmovnz si, bx 
sbb al, 10 
movsx rsi, al 
and rsi, 0b1111111111000 # instrumentation
and esi, 0b111 # instrumentation
lock btr dword ptr [r14 + rsi], esi 
and rcx, 0b1111111111111 # instrumentation
xor eax, dword ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rsi], esi 
adc bl, bl 
add cl, -125 
or esi, 1 # instrumentation
and edx, esi # instrumentation
shr edx, 1 # instrumentation
div esi 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rdi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdi], cl 
and rdi, 0b1111111111111 # instrumentation
cmovb ecx, dword ptr [r14 + rdi] 
cmovs rdi, rdi 
xadd al, bl 
and rdi, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdi], 123 
and rcx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rcx], si 
and rbx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rbx], -20 
movsx ebx, cx 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsr eax, dword ptr [r14 + rdx] 
add cl, -95 # instrumentation
and rdi, 0b1111111111111 # instrumentation
adc si, word ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
xor edx, dword ptr [r14 + rsi] 
adc rcx, rax 
add bl, 105 
xor dx, bx 
and rcx, 0b1111111111111 # instrumentation
sub sil, byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
imul ebx, dword ptr [r14 + rbx] 
cmp rsi, -44 
and rbx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rbx], edi 
and rdi, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rdi], 96 
sub al, dl 
test eax, -389003494 
and rdx, 0b1111111111111 # instrumentation
sub cl, byte ptr [r14 + rdx] 
lea rcx, qword ptr [rcx + rdi + 61195] 
and rcx, 0b1111111111000 # instrumentation
and edi, 0b111 # instrumentation
lock bts dword ptr [r14 + rcx], edi 
cmovbe ecx, edi 
imul rax 
and rbx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rbx], edx 
not dil 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], -102 
and rdx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdx], 56 
btr edx, 217 
and rdx, 0b1111111111111 # instrumentation
or cl, byte ptr [r14 + rdx] 
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
