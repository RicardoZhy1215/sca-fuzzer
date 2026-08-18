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
and rcx, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rcx], ax 
cmp cl, al 
mov dl, bl 
cmovnle si, di 
cmovnb rdi, rdi 
and rcx, 0b1111111111000 # instrumentation
and eax, 0b111 # instrumentation
lock btc dword ptr [r14 + rcx], eax 
lea esi, qword ptr [rbx + rcx + 33143] 
lea rcx, qword ptr [rsi + rax + 15104] 
and rcx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rcx] 
xor bl, -17 
not ax 
and rsi, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rsi], 89 
and rdi, 0b1111111111111 # instrumentation
or bx, word ptr [r14 + rdi] 
movsx rsi, sil 
and rsi, 0b1111111111111 # instrumentation
add word ptr [r14 + rsi], -85 
sub si, dx 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111111 # instrumentation
sub cx, word ptr [r14 + rdx] 
lea edx, qword ptr [rsi] 
and rcx, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rcx], 2 
add dl, -128 # instrumentation
cmovle rdi, rax 
and rdi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rax] 
setnz bl 
and rbx, 0b1111111111111 # instrumentation
movzx eax, word ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rbx], bl 
imul eax 
and rcx, 0b1111111111111 # instrumentation
sbb dil, byte ptr [r14 + rcx] 
cmovs cx, bx 
and rcx, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
bt word ptr [r14 + rcx], bx 
add al, 69 
and rdi, 0b1111111111111 # instrumentation
adc cl, byte ptr [r14 + rdi] 
mov al, bl 
and rdx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rdx] 
mov al, cl 
adc rcx, 5 
and rax, 0b1111111111111 # instrumentation
and rcx, 0b111 # instrumentation
bts qword ptr [r14 + rax], rcx 
and rax, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmovle ax, word ptr [r14 + rdx] 
bts edx, ebx 
and rcx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
and di, word ptr [r14 + rcx] 
cmovo bx, si 
imul ecx, esi 
cmp cl, bl 
and rbx, 0b1111111111111 # instrumentation
cmovnz edx, dword ptr [r14 + rbx] 
setle bl 
and rsi, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], -14962 
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
add bl, -51 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovnz edi, edx 
or rdx, rsi 
xadd cl, cl 
cmovle di, di 
cmovnle si, dx 
and rdi, 0b1111111111111 # instrumentation
test word ptr [r14 + rdi], di 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx 
and rbx, 0b1111111111111 # instrumentation
cmovnb rsi, qword ptr [r14 + rbx] 
cmovs rdx, rdi 
and rsi, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 9 
or bl, -114 
or rax, -2095926181 
lea esi, qword ptr [rsi + rax] 
movsx bx, cl 
lea rcx, qword ptr [rax + rcx + 6705] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], -49 
xor sil, dil 
movsx eax, cl 
xchg dl, bl 
and rdi, 0b1111111111111 # instrumentation
and word ptr [r14 + rdi], -102 
bts di, di 
add dl, 49 # instrumentation
cmovnp rbx, rsi 
setnle al 
btc edi, edi 
add cl, 78 # instrumentation
movsx rdi, ax 
movsx edx, bl 
and rsi, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
not word ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
setle dl 
and rbx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rbx] 
or cl, bl 
and rdx, 0b1111111111111 # instrumentation
cmovnl ecx, dword ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rax], ecx 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax 
mov rsi, rdi 
movzx ebx, dx 
and rbx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rbx] 
setz cl 
and rbx, 0b1111111111111 # instrumentation
movsx ecx, byte ptr [r14 + rbx] 
cmovz ebx, eax 
add bl, al 
bts edx, 143 
and rdx, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdx] 
or dil, 6 
and rcx, 0b1111111111111 # instrumentation
adc sil, byte ptr [r14 + rcx] 
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
