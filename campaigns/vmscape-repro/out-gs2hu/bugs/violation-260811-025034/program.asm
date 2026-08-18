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
add di, di 
cmovnb rdx, rdi 
lea edx, qword ptr [rdx + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovp rdi, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rcx] 
cmovno di, ax 
cmp dx, bx 
and rdx, 0b1111111111111 # instrumentation
mov ebx, dword ptr [r14 + rdx] 
sbb bl, -8 
and rbx, 0b1111111111111 # instrumentation
adc dx, word ptr [r14 + rbx] 
btr bx, si 
and ax, -10754 
and rax, 0b1111111111111 # instrumentation
and eax, dword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rcx], bl 
sbb al, dil 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
lea rsi, qword ptr [rdi + rdi + 31530] 
and cl, -92 
and rdi, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdi], 38 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
add cl, 95 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovz edi, dword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
and eax, dword ptr [r14 + rdi] 
neg edi 
and rsi, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rdx], -58 
or al, 1 # instrumentation
mov ax, 1 # instrumentation
div al 
and rsi, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rsi] 
movsx rcx, dx 
mov dl, cl 
and rdi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdi], -17 
and rsi, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rsi], bx 
lea bx, qword ptr [rbx + rbx + 35856] 
add rax, 704555602 
lea rdx, qword ptr [rbx] 
xor eax, -170293442 
lea ax, qword ptr [rdi + rax + 61530] 
and rdx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdx] 
lea rsi, qword ptr [rsi + rdi + 4598] 
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], dx 
cmovs eax, edx 
setnb bl 
and rcx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rcx], rdx 
btr rcx, 155 
sub sil, sil 
cmovnbe rax, rdi 
and rcx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rcx] 
mov rbx, 8921126502505134529 
setnb dil 
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
add ax, dx 
setb al 
setnl dl 
mov al, cl 
cmp cl, dl 
and rsi, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rsi] 
sbb ax, di 
mov rcx, rdx 
imul di 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], ebx 
cmovs si, ax 
and rdi, 0b1111111111111 # instrumentation
movsx edi, word ptr [r14 + rdi] 
cmovle ecx, esi 
or edx, 1 # instrumentation
add al, 89 # instrumentation
lea rcx, qword ptr [rcx + rax + 61031] 
cmovp cx, cx 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
lea rax, qword ptr [rbx + rsi + 8195] 
and rdi, 0b1111111111111 # instrumentation
movzx di, byte ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rcx] 
mov bx, -3768 
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], sil 
and rdx, 0b1111111111111 # instrumentation
imul rsi, qword ptr [r14 + rdx] 
dec al 
and rsi, 0b1111111111111 # instrumentation
cmovo edx, dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
xor ecx, dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rdi], 6 
and rbx, 0b1111111111111 # instrumentation
cmovb ecx, dword ptr [r14 + rbx] 
sbb rdx, rcx 
and rax, 0b1111111111111 # instrumentation
bt word ptr [r14 + rax], 4 
test al, dil 
and rdi, 0b1111111111111 # instrumentation
cmovo ax, word ptr [r14 + rdi] 
cmpxchg bl, dl 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rcx 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rax], cl 
setl cl 
or cl, cl 
mul ecx 
add dl, -51 # instrumentation
nop ax 
cmovbe bx, bx 
inc cx 
and rdx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
cmovz di, word ptr [r14 + rcx] 
lahf  
and rdi, 0b1111111111111 # instrumentation
cmovnle eax, dword ptr [r14 + rdi] 
cmovno rbx, rsi 
and rsi, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rsi] 
cmovl rdi, rcx 
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
