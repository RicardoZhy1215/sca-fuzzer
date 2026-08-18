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
and ax, 9625 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx 
and rdx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdx], edi 
lea di, qword ptr [rcx + rdx + 23799] 
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], -17 
and rdx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdx], bl 
cmp ax, -17795 
cmp eax, -1110026962 
neg si 
adc dl, dl 
lea ecx, qword ptr [rax] 
and rcx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rcx] 
test ax, -2361 
and rax, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rax] 
sub rdx, 9 
cmovns si, bx 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add bl, 29 # instrumentation
cmovno ax, ax 
bt dx, 76 
and rbx, 0b1111111111111 # instrumentation
and esi, 0b111 # instrumentation
btr dword ptr [r14 + rbx], esi 
and rsi, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rsi], 14 
and rbx, 0b1111111111111 # instrumentation
and rdi, qword ptr [r14 + rbx] 
sbb esi, esi 
and rbx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rbx], bx 
and rdi, 0b1111111111111 # instrumentation
add edi, dword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov cl, byte ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
imul esi, dword ptr [r14 + rdx], 37 
add dl, -101 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovnl si, word ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rdx], al 
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsf rcx, qword ptr [r14 + rbx] 
xor dl, al 
add esi, edi 
and rdx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdx], ecx 
and rcx, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rcx] 
test dl, 47 
and rsi, 0b1111111111111 # instrumentation
not word ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rcx], dl 
and rbx, 0b1111111111111 # instrumentation
test word ptr [r14 + rbx], 25084 
and rcx, 0b1111111111111 # instrumentation
adc eax, dword ptr [r14 + rcx] 
sbb ebx, ecx 
and rcx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rcx], 61 
cmp al, -113 
and rax, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rax], 4 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 67 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rax] 
lea dx, qword ptr [rdi + rsi] 
and rdx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rdx], edi 
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
add dl, 13 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovz edi, dword ptr [r14 + rdi] 
bt eax, 242 
add al, 93 # instrumentation
cmovnle esi, edx 
setp dl 
and rbx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rbx], -88 
and rcx, 0b1111111111111 # instrumentation
cmovbe edx, dword ptr [r14 + rcx] 
lea edi, qword ptr [rax + rdx] 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
xor bl, -75 
cmovno edi, ebx 
lea ecx, qword ptr [rsi + rax + 10154] 
lea rdi, qword ptr [rcx] 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rsi 
movzx eax, dx 
cmovo rax, rdi 
and rdx, 0b1111111111111 # instrumentation
cmovz esi, dword ptr [r14 + rdx] 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
add dl, -21 # instrumentation
sbb al, al 
and rax, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rax] 
inc ax 
lea rbx, qword ptr [rax] 
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], -62 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rsi], cx 
and rdi, 0b1111111111111 # instrumentation
not dword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rdx], 5 
and rax, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], 48 
bt rax, rcx 
and rbx, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rbx], 0 
dec edx 
add rax, rbx 
and rbx, 0b1111111111111 # instrumentation
cmovb rdi, qword ptr [r14 + rbx] 
and rbx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rbx], dl 
and rdi, 0b1111111111111 # instrumentation
adc dil, byte ptr [r14 + rdi] 
xadd rcx, rcx 
and rsi, 0b1111111111111 # instrumentation
and al, byte ptr [r14 + rsi] 
add bx, -86 
sbb rcx, -73 
or dx, 0b1000000000000000 # instrumentation
bsf di, dx 
add al, -8 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 0b1000 # instrumentation
and byte ptr [r14 + rbx], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rbx] 
cmp esi, -52 
inc bl 
cmovnp rcx, rcx 
and rsi, 0b1111111111000 # instrumentation
and rdi, 0b111 # instrumentation
lock btc qword ptr [r14 + rsi], rdi 
or ecx, 35 
and rdx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdx], -2032186631 
and rbx, 0b1111111111111 # instrumentation
add word ptr [r14 + rbx], cx 
dec rdi 
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
