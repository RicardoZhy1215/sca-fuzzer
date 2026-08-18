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
add dl, -119 # instrumentation
btr rbx, rcx 
and rbx, 0b1111111111111 # instrumentation
cmovbe di, word ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rdx], eax 
add al, 87 # instrumentation
lahf  
and rdx, 0b1111111111111 # instrumentation
cmovnle si, word ptr [r14 + rdx] 
lea rcx, qword ptr [rax + rbx] 
sub cl, -77 
sbb rax, -1542197651 
and edi, 9 
or rdx, -7 
and rax, 0b1111111111111 # instrumentation
and cl, byte ptr [r14 + rax] 
inc sil 
cmp sil, 97 
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], 112 
setnz dl 
and rsi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rsi], -12 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, bl 
adc eax, -59926656 
and rbx, 0b1111111111111 # instrumentation
and eax, dword ptr [r14 + rbx] 
adc dl, al 
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi] 
add al, -105 # instrumentation
setz dil 
and rbx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rbx], bl 
mov al, bl 
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx] 
add ax, 21793 
or dl, cl 
or rcx, 0b1000000000000000000000000000000 # instrumentation
bsf rdx, rcx 
add bl, 15 # instrumentation
movzx rcx, sil 
inc dx 
and rdi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdi], 21 
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], -4 
sub dl, 108 
sub rax, rsi 
and rbx, 0b1111111111111 # instrumentation
imul rdx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rcx], al 
and rdx, 0b1111111111111 # instrumentation
dec word ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
and rbx, qword ptr [r14 + rax] 
or al, al 
and rsi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rsi], al 
xor esi, edi 
and rdx, 0b1111111111111 # instrumentation
cmovnb eax, dword ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rsi], cl 
or eax, 0b1000000000000000000000000000000 # instrumentation
bsr ebx, eax 
and rcx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rcx], 26 
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
and rbx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rbx], al 
xor eax, -260312008 
and rbx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rbx] 
lea rsi, qword ptr [rdx + rbx + 49954] 
and rbx, 0b1111111111111 # instrumentation
and ax, 0b111 # instrumentation
btc word ptr [r14 + rbx], ax 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], 112 
and rax, 0b1111111111111 # instrumentation
and byte ptr [r14 + rax], cl 
lea edi, qword ptr [rdx + rax] 
and rdi, 0b1111111111111 # instrumentation
sbb cl, byte ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
adc ecx, dword ptr [r14 + rdi] 
lea rdx, qword ptr [rdi + rdx + 37949] 
imul al 
sbb dil, al 
cmp eax, 400700004 
setl dil 
nop rcx 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add dl, 115 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnl ebx, dword ptr [r14 + rcx] 
sub dl, 25 
and rcx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rcx] 
adc si, -107 
and rdx, 0b1111111111111 # instrumentation
or dil, byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], al 
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsr ebx, ecx 
add cl, bl 
and rdx, 0b1111111111111 # instrumentation
cmovnbe eax, dword ptr [r14 + rdx] 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], -26 
cmp eax, 1050399035 
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], al 
cmovp rdx, rax 
cld  
and rdx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rdx], sil 
setnl al 
mul di 
dec dl 
and rsi, 0b1111111111111 # instrumentation
cmovle esi, dword ptr [r14 + rsi] 
add dil, bl 
and rdi, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rdi], 5 
add dl, -55 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovle ax, word ptr [r14 + rdi] 
lea esi, qword ptr [rcx] 
cmovbe ecx, eax 
and rbx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rbx], al 
cmovnle ax, ax 
neg rbx 
add si, bx 
and rdi, 0b1111111111111 # instrumentation
xor bx, word ptr [r14 + rdi] 
cmovnbe rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
add al, 127 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnle bx, word ptr [r14 + rax] 
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
