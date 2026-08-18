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
and rsi, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rsi], 0 
add bl, -15 # instrumentation
setle al 
and rax, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rax], ax 
mov si, -4452 
adc dx, -38 
test di, 32051 
and rax, 0b1111111111111 # instrumentation
cmovnle si, word ptr [r14 + rax] 
add al, al 
sbb cl, 100 
and al, 55 
setnle cl 
lea esi, qword ptr [rdi] 
lea rax, qword ptr [rdx + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rcx] 
lea bx, qword ptr [rsi + rsi + 12746] 
and rbx, 0b1111111111111 # instrumentation
and bl, byte ptr [r14 + rbx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rsi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rsi], dl 
and rbx, 0b1111111111111 # instrumentation
adc rdx, qword ptr [r14 + rbx] 
add cl, cl 
sbb si, cx 
and rsi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rsi] 
add dl, -117 # instrumentation
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
add dil, byte ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rsi], ax 
bts ecx, edi 
and rax, 0b1111111111111 # instrumentation
test dword ptr [r14 + rax], edi 
and rcx, 0b1111111111111 # instrumentation
movzx eax, byte ptr [r14 + rcx] 
lea ax, qword ptr [rbx + rbx + 43096] 
cmovnl ecx, ecx 
sub edx, ecx 
cmovnb edx, eax 
and rdi, 0b1111111111111 # instrumentation
xor rdx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rcx] 
setnb bl 
neg rbx 
and rbx, 0b1111111111111 # instrumentation
and cl, byte ptr [r14 + rbx] 
lea rdi, qword ptr [rsi + rax + 50851] 
and rsi, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rsi], rax 
imul di, ax 
and rax, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
btr dword ptr [r14 + rax], edi 
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], 30 
and rdi, 0b1111111111111 # instrumentation
movsx ebx, byte ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovnle di, word ptr [r14 + rbx] 
btc ecx, 30 
xor ebx, edi 
not bl 
and rcx, 0b1111111111111 # instrumentation
cmovp eax, dword ptr [r14 + rcx] 
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
add cl, 53 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setnz dl 
and rdi, 0b1111111111111 # instrumentation
and dx, word ptr [r14 + rdi] 
test bl, sil 
movsx si, sil 
and rdi, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
sbb si, word ptr [r14 + rdx] 
cmp dil, 4 
or dil, 64 
and rcx, 0b1111111111111 # instrumentation
btc word ptr [r14 + rcx], 7 
add dl, -128 # instrumentation
lea rbx, qword ptr [rbx + rcx + 4016] 
seto bl 
test dil, bl 
seto cl 
mov rax, 2330751130696381081 
test al, bl 
lea eax, qword ptr [rsi + rcx] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
sub ax, -1947 
and rdx, 0b1111111111111 # instrumentation
cmp dil, byte ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rax] 
imul di, cx 
add bl, -78 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovp edi, dword ptr [r14 + rcx] 
setnbe bl 
sub cl, al 
imul cl 
and rdi, 0b1111111111000 # instrumentation
lock not qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rcx], 0b1000000000000000000000000000000 # instrumentation
bsf rdx, qword ptr [r14 + rcx] 
cmovnz bx, di 
test cl, dl 
and rdi, 0b1111111111111 # instrumentation
cmovle edi, dword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
or cx, word ptr [r14 + rdx] 
movzx ebx, cl 
setnle al 
and rdi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rdi], 44 
lea rcx, qword ptr [rbx] 
or ebx, 1 # instrumentation
and edx, ebx # instrumentation
shr edx, 1 # instrumentation
div ebx 
xor bx, si 
and rdi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdi], 95 
xchg edx, eax 
and rdi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdi], dil 
mov cl, dl 
and rax, 0b1111111111111 # instrumentation
cmovnz edi, dword ptr [r14 + rax] 
lea bx, qword ptr [rax + rbx] 
sub si, -19 
and rsi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rsi], 64 
add cl, -102 
and rsi, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rsi] 
lea ax, qword ptr [rcx + rcx + 18592] 
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
