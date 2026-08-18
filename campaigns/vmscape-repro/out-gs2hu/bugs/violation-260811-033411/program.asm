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
imul si, si 
xor al, al 
setno bl 
xor si, -39 
lea rdi, qword ptr [rdi + rdi] 
xor ax, -13565 
cmovnle eax, esi 
cmp edx, -20 
sbb sil, -86 
xor al, cl 
mov dl, bl 
setnp dil 
or dl, cl 
test al, 12 
and rcx, 0b1111111111111 # instrumentation
cmovnp bx, word ptr [r14 + rcx] 
lea eax, qword ptr [rax] 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
lea edx, qword ptr [rax] 
and rcx, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rcx] 
setnp dl 
lea ebx, qword ptr [rcx] 
sub cl, bl 
and rbx, 0b1111111111111 # instrumentation
movzx ecx, byte ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
movsx rcx, word ptr [r14 + rdi] 
or dil, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv dil 
add dl, -78 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi] 
setb al 
lea ebx, qword ptr [rsi] 
lea rdx, qword ptr [rdx + rdx + 65364] 
setnl cl 
and rdi, 0b1111111111111 # instrumentation
cmovnl si, word ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rax], 2 
and rbx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rbx], dil 
lea eax, qword ptr [rdx + rdx + 5096] 
nop eax 
lea ebx, qword ptr [rax] 
sbb eax, -589140121 
and rsi, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rsi], edi 
and rdx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdx] 
or cl, bl 
inc ecx 
and rcx, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
and dx, word ptr [r14 + rax] 
mov dl, dl 
lea edi, qword ptr [rdx + rcx + 8036] 
cdq  
and rdi, 0b1111111111111 # instrumentation
and word ptr [r14 + rdi], si 
and rsi, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rsi], 2 
sub edx, 46 
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
add al, -47 # instrumentation
and rbx, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rbx] 
test al, -38 
cmovnp eax, ebx 
and rdx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rdx], -32 
or cl, al 
test cx, ax 
btr dx, 186 
and al, 111 
xor cl, 104 
lea dx, qword ptr [rsi + rcx] 
imul eax 
and rsi, 0b1111111111111 # instrumentation
cmovno eax, dword ptr [r14 + rsi] 
sbb dl, -73 
and rsi, 0b1111111111111 # instrumentation
add ax, word ptr [r14 + rsi] 
imul rdx 
mov rdi, -6183626753977216953 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rdx, 0b1111111111000 # instrumentation
and rdx, 0b111 # instrumentation
lock btr qword ptr [r14 + rdx], rdx 
add dl, 37 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rax] 
and al, dil 
sbb dl, -83 
sub cl, dl 
cmovp dx, dx 
and rdi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rdx] 
lea dx, qword ptr [rdx + rcx + 61422] 
and rax, -975354219 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], -87 
setnb bl 
and rax, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rax], bl 
mov dl, dl 
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rbx 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsr ecx, dword ptr [r14 + rsi] 
xor dl, cl 
and rdi, 0b1111111111111 # instrumentation
dec word ptr [r14 + rdi] 
seto dl 
and rdx, 0b1111111111111 # instrumentation
add cl, byte ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
adc bl, byte ptr [r14 + rdi] 
sbb dil, 107 
not al 
and rdx, 0b1111111111111 # instrumentation
add word ptr [r14 + rdx], 94 
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], eax 
btc dx, 4 
and rdx, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rdx], 0 
or ebx, -25 
and rax, 0b1111111111111 # instrumentation
not dword ptr [r14 + rax] 
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
