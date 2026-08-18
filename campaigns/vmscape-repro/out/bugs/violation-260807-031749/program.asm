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
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], 37 
cmovns rsi, rdi 
and dil, -65 
and rbx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rbx], cl 
dec cl 
and rbx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rbx], cl 
adc ax, -24262 
cmovp rsi, rcx 
bt ecx, 50 
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsr esi, dword ptr [r14 + rsi] 
sub rsi, -7 
bt bx, 234 
cmpxchg cl, cl 
and rdx, 0b1111111111111 # instrumentation
cmovbe edi, dword ptr [r14 + rdx] 
mov al, al 
cmovnl cx, cx 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], 59 
cmp cl, -92 
and rax, 0b1111111111111 # instrumentation
or cx, word ptr [r14 + rax] 
setl dil 
or dil, 4 
and rsi, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rsi], 39 
add bl, -100 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovns cx, word ptr [r14 + rdx] 
mov rdx, -2897881419469751162 
and rcx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rcx], ax 
sbb eax, -1132773779 
or rax, 0b1000000000000000000000000000000 # instrumentation
bsf rsi, rax 
lea rax, qword ptr [rcx] 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], 47 
and rdi, 0b1111111111111 # instrumentation
cmovnp rcx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovnp rcx, qword ptr [r14 + rsi] 
btc rax, rdi 
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rcx 
lahf  
add rax, -1878134397 
inc ebx 
test bl, dl 
and rdi, 0b1111111111111 # instrumentation
cmovnz esi, dword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
bts word ptr [r14 + rdx], cx 
and eax, -1862275461 
and cl, bl 
and rdi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdi], ecx 
adc rdx, rcx 
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 0b1000 # instrumentation
and byte ptr [r14 + rbx], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rbx] 
xor si, bx 
cmc  
and rax, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rax], 74 
bts esi, 6 
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
add dl, -52 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
sbb rdx, -106 
and rsi, 0b1111111111111 # instrumentation
cmovnb cx, word ptr [r14 + rsi] 
btr eax, 119 
btc ax, ax 
add cl, -26 # instrumentation
cmovnb edx, esi 
cmovle si, cx 
movsx esi, bl 
sub al, cl 
and al, cl 
and rsi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rsi], si 
cmovp rax, rbx 
cmovno rcx, rcx 
cmovnl si, bx 
or cl, al 
cmp eax, -1056486592 
cmovno ecx, ebx 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
or dil, -66 
and rbx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rbx], 3 
and rax, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rax] 
mov rax, -3529409620798285377 
and rcx, 0b1111111111111 # instrumentation
imul ax, word ptr [r14 + rcx], -105 
and rax, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rax], -38 
bts eax, 185 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
test ax, 30087 
and dl, dl 
and rbx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rbx] 
xchg di, ax 
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], ecx 
mov dl, bl 
xadd si, cx 
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsr rdx, qword ptr [r14 + rax] 
add dl, 50 # instrumentation
setnle al 
movzx si, dl 
sbb cl, al 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], rdx 
cmp bl, 81 
and dl, sil 
cmovnle bx, di 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rcx], bl 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rsi] 
add bl, 31 # instrumentation
sbb bl, dl 
cmc  
and rdx, 0b1111111111111 # instrumentation
cmovb rcx, qword ptr [r14 + rdx] 
sub eax, -1085847755 
and rsi, 0b1111111111111 # instrumentation
and rax, 0b111 # instrumentation
bt qword ptr [r14 + rsi], rax 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], cl 
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
