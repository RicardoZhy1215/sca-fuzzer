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
cmp di, di 
imul al 
and bl, 96 
setp al 
or al, al 
lea esi, qword ptr [rbx + rdi + 53208] 
cmovb ax, cx 
and rdi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rdi], 2 
xor cl, cl 
lea eax, qword ptr [rbx] 
imul rdx 
and rcx, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rcx], 3 
and rbx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rbx], dl 
lea edi, qword ptr [rcx] 
test dl, 37 
setns dil 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
or cl, 99 
and rax, 0b1111111111111 # instrumentation
adc bl, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
or ecx, dword ptr [r14 + rsi] 
xor cl, cl 
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
imul edx, dword ptr [r14 + rsi], 119 
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], -126 
and dl, 108 
or ecx, 1 # instrumentation
and edx, ecx # instrumentation
shr edx, 1 # instrumentation
div ecx 
and rcx, 0b1111111111111 # instrumentation
movsx esi, word ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax 
setp al 
and rax, 0b1111111111111 # instrumentation
mov word ptr [r14 + rax], ax 
and rdi, 0b1111111111000 # instrumentation
and ax, 0b111 # instrumentation
lock bts word ptr [r14 + rdi], ax 
imul al 
and rdx, -49 
and rsi, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rsi], si 
lea edx, qword ptr [rdi] 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
or bx, -119 
and rdx, 0b1111111111111 # instrumentation
neg word ptr [r14 + rdx] 
or al, 22 
and rcx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rcx], edx 
and rsi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rsi], bl 
and rax, 0b1111111111111 # instrumentation
and rcx, 0b111 # instrumentation
bts qword ptr [r14 + rax], rcx 
add bl, 105 # instrumentation
cmovnbe ax, bx 
cmovno rdx, rdx 
setl bl 
sbb rdi, rbx 
and rdx, 0b1111111111111 # instrumentation
and ax, word ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rdx 
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
add al, 90 # instrumentation
cmovnz rcx, rsi 
lea ax, qword ptr [rdx] 
neg esi 
sub dil, -92 
and rdi, 0b1111111111111 # instrumentation
imul ebx, dword ptr [r14 + rdi] 
sub al, bl 
movzx ax, bl 
xor eax, 1164072660 
cmp bl, 106 
and rax, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rax] 
sub cl, bl 
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rcx] 
cmovl rcx, rsi 
add di, 82 
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], eax 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
lea rdx, qword ptr [rdx + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], -60 
movzx si, dl 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], dl 
and rdx, 0b1111111111111 # instrumentation
cmovnl dx, word ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], cl 
and rdx, 0b1111111111111 # instrumentation
cmp al, byte ptr [r14 + rdx] 
cmovl di, cx 
xchg al, cl 
not ax 
add eax, -163853904 
setnl al 
and rcx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rcx], edx 
lea eax, qword ptr [rdx + rax + 35512] 
imul dl 
and rdi, 0b1111111111111 # instrumentation
adc si, word ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], si 
mov rax, 6441586098832726715 
and rbx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rbx] 
xor cl, al 
and rax, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rax], 3 
xchg si, ax 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
and rsi, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rsi], al 
cwd  
imul cl 
add al, -69 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovp rcx, qword ptr [r14 + rdi] 
setbe bl 
and rdx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rdx], 65 
and rdi, 0b1111111111000 # instrumentation
and rdi, 0b111 # instrumentation
lock btr qword ptr [r14 + rdi], rdi 
and rcx, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rcx], 7 
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
