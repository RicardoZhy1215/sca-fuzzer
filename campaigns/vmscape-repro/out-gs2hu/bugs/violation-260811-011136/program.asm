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
or eax, -374224534 
lea rbx, qword ptr [rdi] 
cmovnb edi, eax 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf rcx, qword ptr [r14 + rax] 
or cl, bl 
and rcx, 0b1111111111000 # instrumentation
lock bts dword ptr [r14 + rcx], 1 
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rdi] 
cmp rdi, 55 
and rax, 0b1111111111111 # instrumentation
cmovp di, word ptr [r14 + rax] 
mul al 
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
cbw  
test al, -22 
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr rbx, qword ptr [r14 + rbx] 
test bx, ax 
setnz sil 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add dl, -77 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdx], dl 
setb bl 
and rax, 0b1111111111111 # instrumentation
movzx ax, byte ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rsi], dil 
and rax, 0b1111111111111 # instrumentation
cmovns ecx, dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 40 
lea rbx, qword ptr [rbx + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rbx], 52 
cmovbe ax, si 
mov edi, 40336100 
lea rbx, qword ptr [rax + rsi + 22003] 
and rax, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rax] 
add dl, 16 # instrumentation
cmovb rax, rsi 
and rsi, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rsi] 
lea eax, qword ptr [rsi + rbx + 14915] 
cmovb rax, rax 
and rdx, 0b1111111111111 # instrumentation
cmovnz edx, dword ptr [r14 + rdx] 
sbb cl, -9 
add dl, al 
xor rax, 74 
xor bl, bl 
or ax, 10871 
and rax, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rax], edx 
add cl, cl 
cmp rdi, 118 
xor al, 95 
and rcx, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rcx] 
not edx 
and rcx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rcx], 5 
neg dil 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 1 # instrumentation
and edx, dword ptr [r14 + rax] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rax], 19 
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
test dil, 23 
inc bx 
and rdx, 0b1111111111111 # instrumentation
cmovp eax, dword ptr [r14 + rdx] 
btc rbx, 195 
add dl, -14 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovno di, word ptr [r14 + rax] 
dec edx 
lea edx, qword ptr [rdi + rsi] 
cmp dl, -128 
and rcx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rcx], esi 
and rsi, 0b1111111111111 # instrumentation
and dl, byte ptr [r14 + rsi] 
cmovle esi, ebx 
xadd ecx, eax 
setnp dil 
setnl dl 
movsx rcx, si 
setnl dl 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
or al, -86 
sbb bl, dl 
test bl, 107 
lea rdi, qword ptr [rbx + rdx] 
and rsi, 0b1111111111111 # instrumentation
imul edx, dword ptr [r14 + rsi] 
lea cx, qword ptr [rcx + rbx] 
xchg si, ax 
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], 43 
and rdx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rdx] 
sbb rax, 174300031 
and rsi, 0b1111111111111 # instrumentation
cmovl bx, word ptr [r14 + rsi] 
and cx, -106 
and rdx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdx], -92 
bt di, cx 
mov di, cx 
and rax, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rsi], -17 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 0b1000000000000000 # instrumentation
bsf dx, word ptr [r14 + rbx] 
add dl, -10 # instrumentation
sbb rdx, 74 
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 6 
add cl, 105 # instrumentation
cmovp dx, ax 
and rdx, 0b1111111111111 # instrumentation
cmovnbe esi, dword ptr [r14 + rdx] 
btc rbx, rdx 
add dl, 89 # instrumentation
setnp dil 
and rdx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdx], al 
or cx, 0b1000000000000000 # instrumentation
bsr si, cx 
and rax, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rax], rsi 
mov bx, -32086 
test cl, -103 
and rdi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rdi], ax 
lea esi, qword ptr [rax] 
cmovl ecx, ecx 
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
