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
btc di, cx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi 
lea edx, qword ptr [rax + rbx + 52889] 
and rcx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rcx] 
add ecx, 20 
neg dl 
or al, cl 
sbb bl, bl 
movzx rdx, sil 
xchg ecx, eax 
and rsi, 0b1111111111111 # instrumentation
cmovnb cx, word ptr [r14 + rsi] 
xor bl, 109 
setb cl 
add esi, eax 
and rbx, 0b1111111111111 # instrumentation
cmovz ebx, dword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, -77 # instrumentation
and rdi, 0b1111111111111 # instrumentation
and esi, 0b111 # instrumentation
bt dword ptr [r14 + rdi], esi 
and rdi, 0b1111111111111 # instrumentation
cmovnbe di, word ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
and rax, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rax], dl 
and rdx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rdx] 
xor rax, 1764865508 
mov dl, cl 
mov dl, cl 
and rcx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rcx], cl 
and rcx, 0b1111111111111 # instrumentation
cmovo cx, word ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
movzx ecx, cl 
and rbx, 0b1111111111111 # instrumentation
cmovb cx, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdi], dl 
and rbx, 0b1111111111111 # instrumentation
movsx ecx, word ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], 7 
cmovnp rax, rdx 
and rdi, 0b1111111111111 # instrumentation
sub cl, byte ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rcx], al 
and rax, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rax], 2 
add dl, 18 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovo edi, dword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
bt word ptr [r14 + rbx], cx 
xor dx, 51 
movsx di, al 
and rdx, 0b1111111111111 # instrumentation
cmovp dx, word ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
xor sil, byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rcx], esi 
cdq  
and rbx, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
btc dword ptr [r14 + rbx], edx 
add dl, -117 # instrumentation
cmovnle si, ax 
and rsi, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rsi] 
lea ecx, qword ptr [rdi + rcx + 3062] 
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
mul cl 
add bl, 83 # instrumentation
setz dl 
xor bl, dl 
and rdx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdx] 
adc ax, -20088 
sbb rdi, -33 
sub rax, -1220817954 
xchg rcx, rax 
and cl, 116 
add si, cx 
xadd dl, sil 
and al, al 
and rdx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdx], 99 
lea rdx, qword ptr [rsi + rcx] 
lea rdx, qword ptr [rcx] 
xchg al, dl 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], -113 
and rdi, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rcx], 126 
add rax, rsi 
lea dx, qword ptr [rsi + rbx] 
and rdx, 0b1111111111000 # instrumentation
and rax, 0b111 # instrumentation
lock btr qword ptr [r14 + rdx], rax 
xor dl, 76 
and rdi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmp cx, word ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdx], al 
xor cl, cl 
lea bx, qword ptr [rcx + rdx] 
and rdi, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdi], 21 
movzx dx, sil 
and rax, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
test word ptr [r14 + rcx], bx 
btc edi, eax 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax 
xor ax, bx 
and rax, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rdi] 
movsx rsi, dx 
adc bl, bl 
cmovo ebx, ebx 
cmp sil, 123 
cmp dl, dl 
and rcx, 0b1111111111111 # instrumentation
cmovl edi, dword ptr [r14 + rcx] 
or bl, 124 
movsx di, sil 
imul ecx, edx, 94 
and rax, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rax], ebx 
and rcx, 0b1111111111111 # instrumentation
sub rdx, qword ptr [r14 + rcx] 
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
