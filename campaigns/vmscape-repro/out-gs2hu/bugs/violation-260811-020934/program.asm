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
or eax, -1778844057 
xadd edx, edi 
sub bl, -109 
and rbx, 62 
and rax, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rax], 7 
btc ecx, 147 
and rbx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rbx], -89 
lea si, qword ptr [rsi + rdx] 
cmovs ebx, eax 
sbb dil, 126 
and rdi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rdi], 95 
or dx, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
add bl, 19 # instrumentation
cmovnp rsi, rcx 
and rbx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rbx], 35 
xchg cl, dl 
cmovbe rax, rcx 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], 92 
xchg si, ax 
setnl sil 
and rcx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rcx], 3 
add dl, -60 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovo rdi, qword ptr [r14 + rax] 
cmovnp rax, rsi 
and rdx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdx], rsi 
not dx 
xor rax, -1816126730 
cmp dl, bl 
setnp al 
and rax, 0b1111111111111 # instrumentation
imul rdx, qword ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rax 
and rsi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rsi] 
sub dl, -52 
and rsi, 0b1111111111111 # instrumentation
xor cl, byte ptr [r14 + rsi] 
mov cl, cl 
and rdx, 0b1111111111111 # instrumentation
bts word ptr [r14 + rdx], 5 
and rax, 0b1111111111111 # instrumentation
mov dx, word ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
add edx, dword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
movsx edx, word ptr [r14 + rsi] 
cmovnl di, dx 
and rsi, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rsi] 
add sil, -67 
or ecx, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv ecx 
add cl, -32 # instrumentation
cmovle rdx, rcx 
and rax, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rax], cx 
and rbx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rbx] 
setnp dl 
and rsi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rsi], 43 
cmp dl, cl 
cmovp di, dx 
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
test al, 54 
or ebx, 10 
and rcx, 0b1111111111111 # instrumentation
cmovnp bx, word ptr [r14 + rcx] 
or di, 50 
inc bl 
sub eax, esi 
btr ecx, edi 
lea ecx, qword ptr [rcx] 
mul rdi 
and eax, esi 
and rcx, 0b1111111111000 # instrumentation
and bx, 0b111 # instrumentation
lock bts word ptr [r14 + rcx], bx 
add bl, -53 # instrumentation
cmovl eax, eax 
not sil 
and rax, 0b1111111111111 # instrumentation
mov dil, byte ptr [r14 + rax] 
add cl, 72 
xor esi, -26 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rbx, 0b1111111111111 # instrumentation
neg word ptr [r14 + rbx] 
xchg dl, cl 
and rdi, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdi] 
xor esi, edx 
add sil, 9 
and rcx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rcx], 4 
and rbx, 0b1111111111111 # instrumentation
add word ptr [r14 + rbx], -71 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], -98 
and rax, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rax], esi 
and rcx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rcx], dl 
and rdx, 0b1111111111111 # instrumentation
imul edx, dword ptr [r14 + rdx], 19 
sbb dx, bx 
lea rax, qword ptr [rcx + rbx + 29834] 
lea rbx, qword ptr [rax + rcx] 
sbb al, al 
adc rax, rdx 
and rcx, 0b1111111111111 # instrumentation
sub edx, dword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
and rax, 0b111 # instrumentation
bt qword ptr [r14 + rax], rax 
add dl, -4 # instrumentation
cmovno esi, ecx 
and rsi, 0b1111111111111 # instrumentation
cmovb ax, word ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
test word ptr [r14 + rsi], -29340 
adc edi, edx 
mov dil, sil 
and dil, 74 
sub edi, edx 
and rcx, 0b1111111111111 # instrumentation
bt word ptr [r14 + rcx], 6 
and rax, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rax], 2 
and rax, 0b1111111111111 # instrumentation
add byte ptr [r14 + rax], al 
dec ax 
sbb rbx, rax 
and rcx, 0b1111111111111 # instrumentation
imul rsi, qword ptr [r14 + rcx] 
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
