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
add cl, 51 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rax, 0b1111111111111 # instrumentation
sbb di, word ptr [r14 + rax] 
sbb cl, al 
movzx esi, cl 
and rsi, 0b1111111111111 # instrumentation
and bx, word ptr [r14 + rsi] 
and eax, 1921190438 
and edx, esi 
and rcx, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rcx], al 
btr ecx, 145 
add dl, -116 # instrumentation
cmovp si, cx 
inc esi 
and rcx, 0b1111111111111 # instrumentation
cmovnb bx, word ptr [r14 + rcx] 
setns al 
cmovs ecx, esi 
and cl, dl 
cmovnp cx, si 
and rsi, 0b1111111111111 # instrumentation
imul eax, dword ptr [r14 + rsi], -13 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add bl, -109 # instrumentation
adc bl, 104 
lea rbx, qword ptr [rbx] 
setnle sil 
cmovo edx, edx 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf esi, dword ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], edi 
and rax, 0b1111111111111 # instrumentation
test word ptr [r14 + rax], di 
and rsi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rsi], cl 
cmovnle rdi, rsi 
sbb cl, dl 
and rdi, 0b1111111111111 # instrumentation
neg word ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], cl 
xor al, sil 
neg ax 
sbb al, -88 
and rax, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rax], 89 
movzx si, al 
sub dl, dl 
and rsi, 0b1111111111111 # instrumentation
mov cl, byte ptr [r14 + rsi] 
movsx cx, bl 
and rdx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rdx], sil 
sub dl, 64 
setnb al 
cmovs bx, cx 
and rdi, 0b1111111111111 # instrumentation
adc cx, word ptr [r14 + rdi] 
mov cl, dl 
and rbx, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rbx], 3 
and rax, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rax], 1904685872 
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl 
or bl, bl 
or dil, al 
btc ax, si 
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
add al, 2 # instrumentation
sbb al, dl 
and rsi, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rsi] 
btr bx, si 
sbb dl, dl 
and rdi, 0b1111111111111 # instrumentation
adc al, byte ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rbx] 
not rbx 
imul bx 
setb bl 
sbb bl, bl 
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl 
xor cl, dl 
cmpxchg cl, dl 
and rsi, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rsi], cx 
and rdi, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rdi], 80 
cmovns si, bx 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 0b1000000000000000000000000000000 # instrumentation
bsf ecx, dword ptr [r14 + rcx] 
add al, -71 # instrumentation
adc cl, -10 
and rsi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], -36 
and rbx, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
movsx si, byte ptr [r14 + rax] 
lea eax, qword ptr [rdi + rdi + 44222] 
xchg cl, bl 
and rax, 0b1111111111111 # instrumentation
cmovnle bx, word ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rcx] 
add cl, 115 # instrumentation
and rcx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rcx], -111 
and rdx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdx], rbx 
and rbx, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rbx] 
movzx edi, bl 
and rdx, 0b1111111111111 # instrumentation
and word ptr [r14 + rdx], -10 
and rdi, 0b1111111111111 # instrumentation
imul di, word ptr [r14 + rdi] 
add bl, -58 # instrumentation
setnle cl 
and rax, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rax], 107 
lea rcx, qword ptr [rdx + rbx + 15684] 
lea edi, qword ptr [rcx + rbx] 
and rax, 0b1111111111111 # instrumentation
cmovnb rdi, qword ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rdx], -38 
and rcx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rcx], 2 
add cl, 85 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovno di, word ptr [r14 + rdx] 
lea edx, qword ptr [rcx + rsi] 
sub eax, -349870618 
and bl, 79 
lea rdx, qword ptr [rdx + rdi + 48465] 
sub cx, cx 
movsx rbx, di 
and rcx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rcx], eax 
lea ax, qword ptr [rbx + rdi + 10235] 
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
