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
add cl, -28 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovnz ebx, esi 
test bl, cl 
add dil, cl 
and rdi, 0b1111111111111 # instrumentation
not qword ptr [r14 + rdi] 
imul cx, ax 
test ax, -31667 
xor al, 30 
or si, 0b1000000000000000 # instrumentation
bsr si, si 
neg rdi 
not di 
adc rdi, 42 
and rax, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rax], cl 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], dil 
and rbx, 0b1111111111111 # instrumentation
movsx edi, byte ptr [r14 + rbx] 
btr bx, 84 
and rbx, 0b1111111111111 # instrumentation
imul esi, dword ptr [r14 + rbx] 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], dl 
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], 17 
and ebx, ecx 
and rdx, 0b1111111111111 # instrumentation
cmovp di, word ptr [r14 + rdx] 
cmovnp rbx, rdi 
and rbx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rbx], dil 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdi 
and rdi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdi], bl 
lea rcx, qword ptr [rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovbe ax, word ptr [r14 + rsi] 
or edi, 0b1000000000000000000000000000000 # instrumentation
bsf eax, edi 
add dl, 105 # instrumentation
cmovnp bx, di 
and rbx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rbx], -96 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], 26 
and rax, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rax], -15 
and rcx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rcx], 0 
xchg edi, esi 
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], 5 
mov bl, bl 
and rax, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rcx] 
lea esi, qword ptr [rdi] 
and rcx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rcx], 1 
nop rsi 
add si, -22 
and rbx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rbx], 0 
xadd al, dl 
xor al, -69 
and rdx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdx], 3 
lea ax, qword ptr [rdi + rsi + 7900] 
xadd ecx, eax 
test rax, 1035345541 
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
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsr rbx, qword ptr [r14 + rax] 
add cl, 90 # instrumentation
and rbx, 0b1111111111111 # instrumentation
not word ptr [r14 + rbx] 
setle al 
test ecx, -1800167811 
cmp eax, eax 
lea rsi, qword ptr [rax + rsi + 45667] 
xor cl, 122 
and al, 111 
cmovnl dx, dx 
and rdx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsr rsi, qword ptr [r14 + rdx] 
add al, 32 # instrumentation
cmovnp eax, esi 
or sil, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv sil 
add bl, -104 # instrumentation
setno cl 
and cl, -40 
and esi, eax 
movsx ax, bl 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
cmp ecx, -48 
sbb cl, cl 
and rcx, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rbx], rsi 
bt ax, ax 
add si, cx 
lea di, qword ptr [rdx] 
xor ebx, 59 
and rdi, 0b1111111111111 # instrumentation
cmovl edx, dword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
movzx rcx, word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], sil 
and rax, 0b1111111111111 # instrumentation
cmovb si, word ptr [r14 + rax] 
movzx rdx, dil 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], -89 
sbb eax, 99 
mov rdx, -2750980631697167855 
and rdi, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rdi] 
xchg esi, ecx 
seto dil 
lea rdx, qword ptr [rdx + rax] 
and cl, dl 
and rbx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rbx] 
setnb dl 
xor al, 118 
and rsi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rsi], bl 
adc cl, -64 
and rdx, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rdx], 63 
add al, 112 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rax] 
mul eax 
cmovb ebx, edi 
and rsi, 0b1111111111111 # instrumentation
not word ptr [r14 + rsi] 
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
