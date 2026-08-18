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
add cl, -71 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
mov cl, cl
sbb al, dl
or edx, 40
cmovbe rsi, rdi
and rdi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdi]
cmovl rbx, rdx
movsx rdx, ax
and ecx, edi
test rax, 2129909942
and rbx, 0b1111111111111 # instrumentation
sbb edi, dword ptr [r14 + rbx]
btr rdx, 33
sbb ax, -28254
neg rdx
cmp dl, 37
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsr eax, dword ptr [r14 + rsi]
and cl, cl
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add cl, dl
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], esi
mov rbx, rdi
sbb al, 60
setnp cl
stc
and rsi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsr rdi, qword ptr [r14 + rsi]
xor bl, dl
and rsi, 0b1111111111111 # instrumentation
movzx rsi, word ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], 123
and rcx, 0b1111111111111 # instrumentation
cmovz rdi, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rdi]
add dl, dl
cmovno rbx, rdi
mov ecx, edx
xor dl, 94
sbb bl, al
cmovns bx, dx
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rdx
and rdx, 0b1111111111111 # instrumentation
dec word ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovp edx, dword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmovo ebx, dword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rcx], -73
test al, -100
stc
and rcx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rcx], eax
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], -38
movzx si, bl
and rax, 0b1111111111111 # instrumentation
cmpxchg word ptr [r14 + rax], cx
and rdi, 0b1111111111000 # instrumentation
lock not qword ptr [r14 + rdi]
sub rdi, 93
and rdx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rdx], 7
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
sub ecx, ebx
and cl, al
xadd dl, al
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 45
sub dl, bl
imul rbx, rcx, 79
lea rdi, qword ptr [rsi + rcx]
lea cx, qword ptr [rcx + rdi]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rcx
cmovnz di, di
setnl cl
and rdi, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rdi]
or al, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv al
add dl, 84 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovo rdx, qword ptr [r14 + rsi]
cmc
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
add al, -121 # instrumentation
and rdx, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rdx]
add si, dx
and rsi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rsi], eax
setle cl
cmc
dec al
and rsi, 0b1111111111111 # instrumentation
mov edi, dword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
movsx ebx, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovo cx, word ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
sub rsi, qword ptr [r14 + rbx]
sbb dl, -55
lea dx, qword ptr [rsi + rcx]
dec al
and rsi, 0b1111111111111 # instrumentation
and esi, dword ptr [r14 + rsi]
mov edx, -1865638413
sub al, dl
and rdx, 0b1111111111111 # instrumentation
bts cx, bx
add dl, 41 # instrumentation
xchg ax, ax
and rsi, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rbx], -1014138717
sbb rdi, -126
and rsi, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rsi], 51
and rdx, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rdx], 2
and rbx, 0b1111111111111 # instrumentation
xor dl, byte ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
and ebx, 0b111 # instrumentation
lock btr dword ptr [r14 + rcx], ebx
cmovz esi, esi
lea rdx, qword ptr [rbx + rdx + 21395]
and rsi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rsi], 71
and rdx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rdx], 6
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], -103
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
