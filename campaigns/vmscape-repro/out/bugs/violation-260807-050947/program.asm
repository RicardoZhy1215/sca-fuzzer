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
add bl, -66 # instrumentation
cmovl esi, ecx 
cmovbe rsi, rbx 
cmovb rcx, rcx 
xchg bl, al 
and rdi, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdi] 
and al, dl 
and rbx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rbx], di 
and dx, dx 
lea rax, qword ptr [rcx] 
and rbx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rbx], dx 
xchg cl, cl 
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsf ebx, dword ptr [r14 + rbx] 
add cl, 55 # instrumentation
setnl bl 
neg ecx 
sub cl, cl 
cmovbe eax, eax 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, -120 # instrumentation
cmovle rcx, rax 
and rdx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdx], -125 
nop bx 
and rdx, 0b1111111111111 # instrumentation
imul edx, dword ptr [r14 + rdx], 70 
xor bl, -83 
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 5 
or ax, -26294 
and rax, 0b1111111111111 # instrumentation
cmovnp rcx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovnp si, word ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rcx], 13 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rcx], bx 
and rcx, 0b1111111111111 # instrumentation
cmovnle bx, word ptr [r14 + rcx] 
cmc  
add si, -110 
test bl, cl 
sub bl, cl 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx 
and rdx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rdx], cx 
lea cx, qword ptr [rsi + rsi + 20946] 
or al, 1 # instrumentation
mov ax, 1 # instrumentation
div al 
and rdi, rdi 
and rbx, 0b1111111111111 # instrumentation
add ebx, dword ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rdx], -94 
and rbx, 0b1111111111000 # instrumentation
and rcx, 0b111 # instrumentation
lock bts qword ptr [r14 + rbx], rcx 
and rbx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rbx], 123 
and rsi, 0b1111111111000 # instrumentation
and rdi, 0b111 # instrumentation
lock btr qword ptr [r14 + rsi], rdi 
or bx, 1 # instrumentation
and dx, bx # instrumentation
shr dx, 1 # instrumentation
div bx 
and rbx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rbx], cl 
and rsi, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
btr dword ptr [r14 + rsi], edi 
or dl, cl 
setnz bl 
and rdi, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rdi], rcx 
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
add al, 101 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setnp dil 
bt rdx, 13 
sub rax, -561215638 
xadd cl, bl 
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], dil 
cmp ax, -3963 
setl al 
movzx esi, al 
cmovnb di, di 
mov ebx, 1775918560 
setb cl 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], -67 
and rcx, 0b1111111111111 # instrumentation
cmovp rsi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rbx] 
dec edx 
cmovb ebx, edi 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rbx], rax 
movzx rdx, al 
btc esi, 164 
and rsi, 0b1111111111111 # instrumentation
and rax, 0b111 # instrumentation
bt qword ptr [r14 + rsi], rax 
inc al 
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], 33 
and rax, 0b1111111111111 # instrumentation
adc rbx, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rsi], al 
xor al, 19 
or ecx, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv ecx 
add dl, -125 # instrumentation
cmovnp bx, cx 
lea ebx, qword ptr [rax + rcx + 63711] 
and rcx, 0b1111111111111 # instrumentation
and word ptr [r14 + rcx], -48 
setnle cl 
dec ebx 
and rax, 0b1111111111111 # instrumentation
xor rax, qword ptr [r14 + rax] 
movsx rcx, ax 
and rsi, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rsi] 
sbb cl, 64 
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx] 
add al, -100 # instrumentation
mov al, sil 
and rcx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdi], sil 
cmovz rbx, rdx 
and rbx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rbx], -40 
cwde  
adc rax, 49 
lea rax, qword ptr [rsi + rbx] 
cmovb ax, ax 
cmovl edi, ecx 
and rbx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rbx], dl 
sets dil 
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
