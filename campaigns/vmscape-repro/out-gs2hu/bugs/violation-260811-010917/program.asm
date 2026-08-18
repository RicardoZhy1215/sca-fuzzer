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
add dl, 42 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovo si, si 
add al, dl 
and rcx, 0b1111111111111 # instrumentation
and ebx, 0b111 # instrumentation
btr dword ptr [r14 + rcx], ebx 
add bl, 94 # instrumentation
setp cl 
and rbx, 0b1111111111111 # instrumentation
cmovz ax, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdi], 26935 
and rcx, 0b1111111111111 # instrumentation
cmovz ecx, dword ptr [r14 + rcx] 
mov bl, 11 
and rax, 0b1111111111111 # instrumentation
cmovb rax, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
imul rax, qword ptr [r14 + rbx], -47 
setb al 
sbb rcx, -60 
neg edx 
lea cx, qword ptr [rbx] 
xchg cl, bl 
lea di, qword ptr [rcx + rbx + 9904] 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add cl, -1 # instrumentation
and rbx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rbx], al 
and rsi, 0b1111111111111 # instrumentation
and rcx, qword ptr [r14 + rsi] 
and dl, bl 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rcx] 
xor al, dl 
adc rdx, -121 
xor dl, -36 
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], cl 
setns cl 
cmovno eax, eax 
movzx di, cl 
setnbe al 
and rbx, 0b1111111111000 # instrumentation
and eax, 0b111 # instrumentation
lock btc dword ptr [r14 + rbx], eax 
add dl, -5 # instrumentation
seto sil 
sub al, -108 
adc bx, dx 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi 
and rdx, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rdx], 4 
and rdx, 0b1111111111111 # instrumentation
add dil, byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rcx], esi 
and rcx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rcx], -71 
and rsi, 0b1111111111000 # instrumentation
lock bts dword ptr [r14 + rsi], 1 
and rdx, 0b1111111111111 # instrumentation
movsx ax, byte ptr [r14 + rdx] 
lea edx, qword ptr [rax + rbx + 29959] 
and rax, 0b1111111111111 # instrumentation
cmovnz rdx, qword ptr [r14 + rax] 
test dl, al 
cmovle bx, bx 
cmp cx, cx 
and rdx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rdx], 88 
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
test eax, -1112865376 
or al, cl 
and rcx, 0b1111111111111 # instrumentation
cmovl edi, dword ptr [r14 + rcx] 
setnle bl 
lea cx, qword ptr [rcx] 
and eax, -125 
setnz cl 
cmp al, -87 
xor edi, edx 
and rcx, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rcx] 
setnle al 
add dl, bl 
mov al, al 
xchg dl, bl 
or sil, 1 # instrumentation
mov ax, 1 # instrumentation
div sil 
add dl, -66 # instrumentation
cmovnb edx, eax 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
and rbx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rbx], -29 
adc cl, al 
cmovz ecx, esi 
inc dl 
sets cl 
adc sil, 84 
or rax, rdi 
lea edi, qword ptr [rdi + rbx] 
lea eax, qword ptr [rax + rdx + 48453] 
and rbx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rbx], 110 
and rdi, 0b1111111111111 # instrumentation
adc dl, byte ptr [r14 + rdi] 
test eax, 299688314 
and rax, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rax], cl 
and rdi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdi], -39 
and rax, 0b1111111111111 # instrumentation
cmovno esi, dword ptr [r14 + rax] 
lea rdi, qword ptr [rcx] 
and rdi, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdi] 
lea edi, qword ptr [rsi] 
xchg dx, ax 
and rax, 0b1111111111111 # instrumentation
add cl, byte ptr [r14 + rax] 
xchg cx, ax 
imul bl 
add dl, -45 # instrumentation
movsx edx, bl 
and rdx, 0b1111111111111 # instrumentation
cmovnp rdi, qword ptr [r14 + rdx] 
test bl, -21 
and rsi, 0b1111111111111 # instrumentation
cmovno rdi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rcx], al 
and rbx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rbx], -39 
cwd  
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx] 
and rdi, 48 
not cx 
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
