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
add cl, -27 # instrumentation
setnl al 
cmovnz di, bx 
and rdi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
and cl, byte ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
or rsi, qword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
sub word ptr [r14 + rcx], cx 
dec ebx 
xor rax, 1655073670 
mov di, 11270 
and rax, 1735534126 
sub cl, cl 
and rcx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rcx] 
sbb dl, cl 
mov al, al 
dec rdi 
lea rcx, qword ptr [rax + rdi + 15981] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
movzx rbx, di 
movsx ebx, dl 
and rdi, 0b1111111111000 # instrumentation
and bx, 0b111 # instrumentation
lock btr word ptr [r14 + rdi], bx 
add bl, 37 # instrumentation
setnl dil 
lea ebx, qword ptr [rdx + rcx] 
and al, -44 
xchg ecx, eax 
adc dl, bl 
and rax, 0b1111111111111 # instrumentation
and dword ptr [r14 + rax], edi 
and rbx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rsi], -23 
xor dl, -82 
nop bx 
cmp bl, sil 
bts ax, 19 
movsx dx, al 
and rcx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rcx], cl 
and rsi, 0b1111111111111 # instrumentation
not qword ptr [r14 + rsi] 
cmovns rcx, rsi 
cmpxchg bx, dx 
sub dl, 62 
and rdx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdx], edx 
and rdi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdi], sil 
and rsi, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
btc word ptr [r14 + rsi], si 
and rcx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rcx], -121 
and rdi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdi], rdx 
and rsi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rsi], rcx 
and rdx, 0b1111111111111 # instrumentation
movzx ecx, byte ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rbx], -80 
and rbx, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
cmovno rbx, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdi], cl 
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
and rdx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rdx], 40 
and rsi, 0b1111111111000 # instrumentation
lock neg word ptr [r14 + rsi] 
btr dx, 254 
and rbx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rbx], 104 
and rcx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rcx] 
mov eax, -828991345 
or rax, 0b1000000000000000000000000000000 # instrumentation
bsr rax, rax 
sub al, bl 
and rax, 0b1111111111111 # instrumentation
cmovnz edi, dword ptr [r14 + rax] 
btr rcx, 219 
xor al, -11 
cmovns edi, edx 
and rcx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rcx] 
and rbx, 0b1111111111000 # instrumentation
and rax, 0b111 # instrumentation
lock btr qword ptr [r14 + rbx], rax 
add al, 91 
and rax, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rax], -104 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add bl, -64 # instrumentation
sbb bx, cx 
and rdi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdi], -9 
and rax, 0b1111111111111 # instrumentation
xor ecx, dword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
and si, 0b111 # instrumentation
lock btc word ptr [r14 + rsi], si 
and rax, 0b1111111111111 # instrumentation
movzx eax, word ptr [r14 + rax] 
or bx, -1 
lea bx, qword ptr [rbx + rsi] 
setbe bl 
add eax, 858273418 
nop rax 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rsi 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi 
xchg cx, di 
and rbx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rbx], al 
and rdi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rdi], edx 
and rbx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmovs esi, dword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
cmovle ax, word ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
not word ptr [r14 + rbx] 
cmp dl, al 
and rdi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdi], edi 
and rcx, 0b1111111111111 # instrumentation
cmovnbe edi, dword ptr [r14 + rcx] 
mov dx, -5449 
dec al 
and rdi, 0b1111111111111 # instrumentation
movzx si, byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rsi], 95 
lea bx, qword ptr [rdi + rax + 58154] 
neg sil 
add cl, cl 
cmovl cx, cx 
and rdi, 0b1111111111111 # instrumentation
adc dl, byte ptr [r14 + rdi] 
lea si, qword ptr [rbx] 
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
