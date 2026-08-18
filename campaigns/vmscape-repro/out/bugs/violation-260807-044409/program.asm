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
add cl, -37 # instrumentation
lea rdi, qword ptr [rsi + rdx + 19168] 
lea ecx, qword ptr [rdi] 
and rcx, 0b1111111111111 # instrumentation
cmovb cx, word ptr [r14 + rcx] 
cmpxchg cl, al 
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr rcx, qword ptr [r14 + rbx] 
lea esi, qword ptr [rsi + rsi + 837] 
and al, -4 
and rbx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rbx], bl 
xor al, -40 
lea ax, qword ptr [rsi + rbx + 33933] 
and rsi, 0b1111111111111 # instrumentation
cmovnz ebx, dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rdi], esi 
and rax, 0b1111111111111 # instrumentation
cmovnbe si, word ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], cl 
lea ecx, qword ptr [rax + rcx] 
bts edi, eax 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rdx] 
cmovs si, bx 
neg cl 
xchg rsi, rcx 
and rdx, 0b1111111111111 # instrumentation
adc al, byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rdx] 
test bl, 101 
adc bl, 60 
inc rbx 
or rdi, rdi 
btc edi, 252 
or cl, -81 
and rax, 0b1111111111111 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
movzx dx, dl 
cmp al, -7 
and rbx, 0b1111111111111 # instrumentation
mov cx, word ptr [r14 + rbx] 
cmovo rdi, rdx 
setl al 
sub al, -126 
and cl, -68 
xor bx, si 
lea di, qword ptr [rsi] 
sbb sil, -121 
and rdi, 0b1111111111111 # instrumentation
cmovz dx, word ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rax], bx 
sbb rbx, 42 
lea esi, qword ptr [rdx + rdx + 62235] 
add ecx, 111 
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
add dl, -40 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setz dl 
sub eax, -1214467584 
setnp al 
and rdi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdi], 42 
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 89 
cmovle edx, ebx 
and rdi, 0b1111111111111 # instrumentation
neg word ptr [r14 + rdi] 
xor bl, al 
sub bl, cl 
sbb dl, -127 
cmpxchg si, ax 
bt rbx, 102 
lea rdi, qword ptr [rcx] 
xor rsi, -101 
add bl, dl 
lea bx, qword ptr [rcx + rcx + 49265] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea rax, qword ptr [rsi + rbx + 38317] 
and rdi, 0b1111111111111 # instrumentation
inc word ptr [r14 + rdi] 
lea cx, qword ptr [rdx] 
or edi, 15 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rcx 
and rax, 0b1111111111111 # instrumentation
add word ptr [r14 + rax], -70 
and rsi, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rax], al 
lea esi, qword ptr [rax + rcx] 
xchg edi, eax 
setz al 
and rdx, 0b1111111111111 # instrumentation
cmovle cx, word ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
bt word ptr [r14 + rsi], cx 
and rcx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rcx], -39 
and rbx, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rbx] 
lea rdx, qword ptr [rsi + rdi + 54918] 
movsx edx, dl 
xor al, al 
setb cl 
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], rax 
or esi, 122 
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], sil 
test cl, -106 
cmovle rcx, rdx 
and rcx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rcx], rsi 
and rdi, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
and byte ptr [r14 + rax], cl 
and rdx, 0b1111111111111 # instrumentation
cmovz rsi, qword ptr [r14 + rdx] 
xadd bl, al 
test rsi, rcx 
and rdi, 0b1111111111111 # instrumentation
cmovb rdi, qword ptr [r14 + rdi] 
add cl, dl 
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
