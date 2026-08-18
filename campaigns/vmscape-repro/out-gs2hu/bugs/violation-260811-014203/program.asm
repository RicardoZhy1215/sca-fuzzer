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
add dl, -99 
or esi, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv esi 
xor dx, bx 
dec dl 
xchg al, bl 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx 
setl sil 
bt esi, edx 
and rax, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rax], -110 
and rax, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rax], -15 
and rcx, 0b1111111111111 # instrumentation
cmovnb cx, word ptr [r14 + rcx] 
xor eax, -398412359 
sub di, 67 
and rbx, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rbx], 0 
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx] 
lea rdi, qword ptr [rdi] 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
and rdx, 0b1111111111111 # instrumentation
and rdi, qword ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
cmovb eax, dword ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rax], -61 
sbb dil, dl 
adc rax, 1220351230 
and rsi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rsi], -101 
and rbx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rbx], dil 
and rdx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rdx], -100 
and rdi, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 1 # instrumentation
and dx, word ptr [r14 + rcx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rcx], 4 
xor bl, sil 
and rsi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rsi], al 
imul rsi, rdx, 7 
imul dl 
or sil, 1 # instrumentation
mov ax, 1 # instrumentation
div sil 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rdx], cl 
and rdx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdx], dl 
cmp bl, dl 
cmpxchg rsi, rsi 
cmp al, al 
bt rcx, 206 
movzx ebx, si 
btr edi, eax 
and rdx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdx], dl 
cmovnb rcx, rcx 
btc rsi, 165 
add cl, 54 # instrumentation
cmovs dx, di 
and rdi, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rax], rdi 
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
and rcx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rcx], al 
lea esi, qword ptr [rbx + rax] 
and rax, 0b1111111111111 # instrumentation
mul word ptr [r14 + rax] 
lea bx, qword ptr [rax + rbx] 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rcx] 
mov eax, -95198181 
and rdi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rdi] 
imul eax, eax, -37 
sub al, -4 
or cl, bl 
and rdi, 0b1111111111111 # instrumentation
cmovnp rcx, qword ptr [r14 + rdi] 
setl dil 
lea rdx, qword ptr [rdi + rbx] 
movzx di, sil 
cmovp rdi, rdi 
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsf ecx, ecx 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
mul bl 
test ax, -29888 
and rbx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
and al, byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
cmovb eax, dword ptr [r14 + rcx] 
adc bl, 116 
stc  
and rdx, 0b1111111111111 # instrumentation
adc cl, byte ptr [r14 + rdx] 
or edx, 0b1000000000000000000000000000000 # instrumentation
bsr esi, edx 
add bl, -8 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovo di, word ptr [r14 + rbx] 
lea rcx, qword ptr [rdi + rdi + 25432] 
and rdi, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdi], 82 
movsx edi, dl 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf esi, dword ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rdi], cx 
and rsi, 0b1111111111111 # instrumentation
imul di, word ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
sub rbx, qword ptr [r14 + rax] 
mul bl 
and rcx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rcx] 
adc al, al 
movsx ebx, al 
or eax, 1390054082 
add dl, bl 
and rdx, 0b1111111111111 # instrumentation
and rax, 0b111 # instrumentation
bt qword ptr [r14 + rdx], rax 
add al, 50 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovp rdi, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
imul si, ax 
test al, sil 
add rbx, rdx 
and rcx, 0b1111111111111 # instrumentation
movsx esi, word ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rsi], -89 
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
