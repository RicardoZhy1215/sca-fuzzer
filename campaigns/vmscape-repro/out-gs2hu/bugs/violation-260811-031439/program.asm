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
xadd eax, esi 
lea eax, qword ptr [rbx + rdi] 
or al, 71 
imul dx 
adc dl, dl 
mul eax 
xor cl, -4 
and rax, 0b1111111111111 # instrumentation
cmovz edi, dword ptr [r14 + rax] 
neg edi 
and rax, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rax], di 
movsx rsi, dil 
setnle dil 
not rcx 
setnz al 
lea dx, qword ptr [rsi + rsi] 
and rbx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rbx], 3 
lea rbx, qword ptr [rip +.function_victim_safe] # noremove
jmp rbx # noremove
.exit_victim_1:
.function_victim_safe:
.bb_victim_safe.entry:
add bl, 61 # instrumentation
setz cl 
and rdi, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rdi], -69 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdx 
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], ax 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 0b1000000000000000000000000000000 # instrumentation
bsf edx, dword ptr [r14 + rcx] 
mov bx, 5713 
and rcx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rcx], eax 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], -23 
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi] 
not bl 
and rdx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rdx], di 
and bl, -124 
adc dil, -89 
xchg di, ax 
and rcx, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rax], -41 
neg rdx 
test bl, 85 
and rdx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rdx], 1 
add cl, 127 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdx] 
lea eax, qword ptr [rcx] 
and rax, 0b1111111111111 # instrumentation
movsx edi, byte ptr [r14 + rax] 
and rcx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rcx], -11 
cmovb edi, edx 
and rdx, 0b1111111111111 # instrumentation
dec word ptr [r14 + rdx] 
lea di, qword ptr [rdi] 
cmpxchg edi, edi 
and rdx, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock bts word ptr [r14 + rbx], 1 
or al, -14 
cmp cx, -78 
cmovs ebx, ecx 
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
xor bl, dl 
cmp cl, cl 
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], 12 
stc  
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf bx, word ptr [r14 + rsi] 
or ax, 79 
lea edx, qword ptr [rax] 
and rcx, 0b1111111111111 # instrumentation
movsx edi, byte ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], 65 
and edx, ecx 
lea rdi, qword ptr [rcx + rsi] 
add bl, al 
sbb rdx, rdx 
and rdi, 0b1111111111111 # instrumentation
cmovl ebx, dword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rcx], -602799891 
and rax, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rax] 
lea rax, qword ptr [rip +.function_attacker_gadget] # noremove
jmp rax # noremove
.exit_attacker_train:
.function_attacker_gadget:
.bb_attacker_gadget.entry:
lea edi, qword ptr [rax + rsi + 51672] 
and rbx, 0b1111111111111 # instrumentation
add esi, dword ptr [r14 + rbx] 
setbe al 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rbx 
sbb dil, sil 
and rdx, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rdx], 4 
and rcx, 0b1111111111111 # instrumentation
cmovnb edi, dword ptr [r14 + rcx] 
cmovnz rsi, rbx 
sub dil, dl 
and rdx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rdx], 31 
add dl, bl 
add al, -64 
dec ebx 
xchg esi, ebx 
test al, 116 
or eax, 1170786375 
and rax, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rax], edx 
and rdi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rdi], cl 
and rdi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rdi], 4 
and ax, 127 
cmovb rdx, rax 
or bl, 0b1000 # instrumentation
and bl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv bl 
and rbx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rbx], eax 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], 102 
and rax, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bts qword ptr [r14 + rax], rsi 
and rbx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
add word ptr [r14 + rcx], -86 
and rdi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rbx], -35 
seto sil 
cmovb rax, rdx 
lea di, qword ptr [rdi + rax] 
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
