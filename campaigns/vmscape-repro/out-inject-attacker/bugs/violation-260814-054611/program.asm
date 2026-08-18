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
.macro.set_h2g_target.vm.function_vm_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]
.exit_main_1:
.function_main_2:
.bb_main_2.entry:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.switch_h2g.vmvictim.2: nop qword ptr [rax + 0xff]
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.3: nop qword ptr [rax + 0xff]
.exit_main_3:
.function_main_4:
.bb_main_4.entry:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop  
.exit_main_4:
.section .data.vmvictim
.function_vmvictim_1:
.bb_vmvictim_1.entry:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdi 
and rbx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rbx], rax 
sbb bl, 119 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rdi], cl 
lea cx, qword ptr [rdi + rbx + 39013] 
lea rbx, qword ptr [rcx] 
add cl, -36 
lea bx, qword ptr [rsi + rdi + 34266] 
and rsi, 0b1111111111111 # instrumentation
add bl, -49 # instrumentation
movsx edx, sil 
bswap rbx 
setl al 
add eax, -1911586005 
and rsi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rsi], cl 
add rbx, rax 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
dec dl 
cmovp rbx, rcx 
test al, 74 
and rdi, 0b1111111111111 # instrumentation
neg word ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], -65 
or rdx, rcx 
and rdi, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
btc dword ptr [r14 + rbx], edx 
and rax, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rax], -54 
and rsi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rsi], ebx 
and rdx, 0b1111111111111 # instrumentation
and rsi, qword ptr [r14 + rdx] 
sahf  
adc cl, sil 
mov ax, 21195 
and rsi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rsi], edi 
and rax, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rax], -125 
and rsi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rsi], eax 
cmovnz edi, esi 
and rsi, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rsi], -73 
and rax, 0b1111111111000 # instrumentation
and rsi, 0b111 # instrumentation
lock btc qword ptr [r14 + rax], rsi 
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], bl 
and rcx, 0b1111111111111 # instrumentation
cmovp rdi, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
or dx, word ptr [r14 + rdx] 
lea ax, qword ptr [rbx + rcx + 31757] 
and rsi, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rsi], edx 
and rbx, 0b1111111111111 # instrumentation
adc cl, byte ptr [r14 + rbx] 
lea bx, qword ptr [rbx + rdx] 
and rax, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rax], rsi 
and rdi, 0b1111111111111 # instrumentation
sbb ecx, dword ptr [r14 + rdi] 
lea rdx, qword ptr [rbx + rbx + 18232] 
sub eax, 1355010483 
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence  
.exit_vmvictim_3:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_0:
.function_vm_train:
.bb_vm_train.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
add al, 96 
bt rdi, 27 
add cl, 57 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovo ebx, dword ptr [r14 + rdi] 
sub esi, edx 
and rsi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rsi], edx 
cmovnle eax, ebx 
adc dil, 102 
and rsi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rsi], al 
and rdi, 0b1111111111000 # instrumentation
and rax, 0b111 # instrumentation
lock btc qword ptr [r14 + rdi], rax 
and rsi, 0b1111111111111 # instrumentation
add word ptr [r14 + rsi], -48 
and rbx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rbx], -68 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsr rsi, qword ptr [r14 + rdi] 
cmp al, -78 
and rbx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rbx], si 
adc al, -90 
and rsi, 0b1111111111111 # instrumentation
inc word ptr [r14 + rsi] 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
and rcx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rcx], dx 
and rdx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rdx], dx 
cmovno rbx, rbx 
adc dil, 59 
btc dx, 95 
and rsi, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rsi], edi 
movsx ebx, dil 
cmp cl, dl 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 0b1000000000000000000000000000000 # instrumentation
bsf edx, dword ptr [r14 + rcx] 
xadd ebx, ebx 
xchg ecx, eax 
and rsi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rsi], edx 
and rbx, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmovnz si, word ptr [r14 + rsi] 
setnle dil 
or esi, 0b1000000000000000000000000000000 # instrumentation
bsf ebx, esi 
lfence  # noremove
.macro.switch_g2h.main.vm_1: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_3:
.function_vm_measure:
.bb_vm_measure.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_measure:
.section .data.main
.test_case_exit:nop
