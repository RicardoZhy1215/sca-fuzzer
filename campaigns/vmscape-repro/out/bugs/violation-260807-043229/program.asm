.intel_syntax noprefix
.section .data.main
.function_main_0:
.bb_main_0.entry:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]
.exit_main_0:
.function_main_1:
.bb_main_1.entry:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]
.exit_main_1:
.function_main_2:
.bb_main_2.entry:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]
.exit_main_3:
.function_main_4:
.bb_main_4.entry:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop  
.exit_main_4:
.section .data.vmvictim
.function_vmvictim_0:
.bb_vmvictim_0.entry:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
or rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rcx], bl 
xadd edx, eax 
and rbx, 0b1111111111000 # instrumentation
and dx, 0b111 # instrumentation
lock bts word ptr [r14 + rbx], dx 
add al, 113 # instrumentation
cmovbe edi, edi 
and rbx, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rbx] 
and dil, -7 
and rbx, 0b1111111111111 # instrumentation
cmovnz dx, word ptr [r14 + rbx] 
lea rdx, qword ptr [rcx + rax] 
and rsi, 0b1111111111111 # instrumentation
cmovb cx, word ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rsi], 1 
btr edx, edx 
sahf  
or rax, -1404502726 
and rdx, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rdx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
movsx dx, cl 
and rax, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], cl 
and rcx, 0b1111111111111 # instrumentation
cmovo rdi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmovp rdi, qword ptr [r14 + rdi] 
cmovnbe dx, bx 
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], cl 
adc ax, 14872 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx 
and rsi, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rsi] 
test rdx, rsi 
cmovo rdi, rdi 
add edx, -80 
and rbx, 0b1111111111111 # instrumentation
test word ptr [r14 + rbx], dx 
and rdi, 0b1111111111111 # instrumentation
cmp bl, byte ptr [r14 + rdi] 
btr esi, 135 
bts rdx, rcx 
and rcx, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
movzx di, byte ptr [r14 + rcx] 
cmpxchg al, bl 
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], -74 
and rdx, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rax], 103 
and rdi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdi], edi 
and rcx, 0b1111111111111 # instrumentation
cmovo si, word ptr [r14 + rcx] 
test dil, -107 
stc  
movsx edi, si 
add dl, -51 
cmpxchg sil, dl 
cmovo rdi, rbx 
lfence  
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vmvictim_1:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_0:
.function_vm_1:
.bb_vm_1.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rsi, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rsi], dx 
and rdx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rdx], 123 
setnle bl 
add esi, edx 
sub rdx, -127 
and rbx, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
sbb ebx, dword ptr [r14 + rsi] 
imul cx, ax, -60 
nop esi 
sbb al, 61 
and rsi, 0b1111111111111 # instrumentation
cmovnl di, word ptr [r14 + rsi] 
bts esi, edx 
and rax, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
btr qword ptr [r14 + rax], rsi 
add bl, -123 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovno edx, dword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovnbe eax, dword ptr [r14 + rbx] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea si, qword ptr [rdi + rax] 
and rax, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rax], ax 
and rbx, 0b1111111111111 # instrumentation
cmovp rdx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdx], -76 
and rsi, 0b1111111111111 # instrumentation
add al, -45 # instrumentation
seto dl 
xor bl, cl 
sub cx, dx 
movsx edx, dl 
and rax, 0b1111111111111 # instrumentation
sub cl, byte ptr [r14 + rax] 
seto dl 
lea eax, qword ptr [rcx + rax] 
lea ax, qword ptr [rdx + rsi + 31799] 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], dl 
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], 65 
cwd  
imul si, di 
and rbx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rbx], 59 
cmovnl edx, edi 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdi 
and rcx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rcx], -39 
and rdx, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rdx], 3 
mov al, dl 
imul rsi 
and rbx, 0b1111111111111 # instrumentation
cmovnb ax, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdi] 
and rdi, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rdi], 5 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], 95 
test dl, 97 
cmovb di, dx 
setbe bl 
setnz dl 
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_3:
.function_vm_2:
.bb_vm_2.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_2:
.section .data.main
.test_case_exit:nop
