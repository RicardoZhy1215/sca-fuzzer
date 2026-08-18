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
and al, dl 
setbe al 
test dl, cl 
xor al, 100 
neg dil 
setbe al 
xor di, ax 
and rsi, 0b1111111111111 # instrumentation
movsx rsi, word ptr [r14 + rsi] 
or ax, -31224 
and rax, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rax], -117 
cmovnz rdi, rsi 
xchg al, cl 
and rdi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdi], cl 
sbb al, -20 
dec rcx 
and rsi, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bts qword ptr [r14 + rsi], rsi 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add al, 99 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovno edi, dword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
movsx si, byte ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rcx], edx 
and rdi, 0b1111111111111 # instrumentation
and rax, 0b111 # instrumentation
bt qword ptr [r14 + rdi], rax 
bt edx, 72 
and rax, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rbx], 51 
cmovl ecx, ebx 
add cx, -49 
and rdi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock bts dword ptr [r14 + rdx], 2 
and rsi, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rsi], rax 
cmovnl esi, ecx 
cmovs rcx, rax 
and rdi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
mov cx, word ptr [r14 + rsi] 
cmovz esi, esi 
and rbx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rbx], edx 
lea eax, qword ptr [rsi + rbx] 
and rbx, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rax], al 
mov dl, dil 
add rax, rbx 
and rbx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rbx] 
or bx, 1 # instrumentation
and dx, bx # instrumentation
shr dx, 1 # instrumentation
div bx 
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx] 
sbb di, cx 
cmp sil, dil 
and rcx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rcx], al 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rbx 
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
add dl, -44 # instrumentation
sbb bl, al 
and rcx, 0b1111111111111 # instrumentation
movsx edi, byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
test qword ptr [r14 + rax], rdx 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], ecx 
cmp dl, bl 
xchg rcx, rax 
imul di, si, -95 
mov eax, 1421331073 
and rcx, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rcx] 
sub rsi, rdx 
sbb al, bl 
imul di, si, -4 
inc cl 
adc al, 43 
imul dl 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rax, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmovns si, word ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rsi], ebx 
and rax, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rax], 3 
cdq  
add al, -40 
lea ebx, qword ptr [rdi] 
and rdx, 0b1111111111111 # instrumentation
and sil, byte ptr [r14 + rdx] 
cmc  
and rbx, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rbx] 
lea edi, qword ptr [rax + rbx + 59741] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
add cl, 87 # instrumentation
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], ax 
and rsi, 0b1111111111111 # instrumentation
cmovl cx, word ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rcx] 
and rcx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rcx], dl 
lea ax, qword ptr [rdx + rdi + 50714] 
and rax, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rax], dx 
sub rax, rax 
and rbx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rbx], -85 
and rax, 0b1111111111111 # instrumentation
add byte ptr [r14 + rax], cl 
and rbx, 0b1111111111111 # instrumentation
cmovnb eax, dword ptr [r14 + rbx] 
mul dl 
or cl, -89 
cmp al, 54 
and rbx, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rbx] 
adc bx, di 
and rcx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rcx], edx 
and rbx, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], bl 
and rax, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rax], dil 
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
