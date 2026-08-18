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
add bl, -122 # instrumentation
adc rsi, rdi 
sbb dl, -117 
test al, al 
movzx rcx, bl 
cmovns rcx, rdx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi 
lea rax, qword ptr [rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovnp ecx, dword ptr [r14 + rsi] 
sub rdi, rdi 
cmovz bx, dx 
sbb eax, -420217322 
not cx 
stc  
and rbx, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rbx] 
btr rsi, rdi 
cwde  
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
sub ax, 102 
and rdi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rdi], ax 
xchg cl, dl 
cmp rax, -405327609 
and rax, 0b1111111111111 # instrumentation
imul word ptr [r14 + rax] 
setnb cl 
and rcx, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rcx] 
or cx, cx 
lea rdi, qword ptr [rdx + rax + 11408] 
and rcx, 0b1111111111111 # instrumentation
adc esi, dword ptr [r14 + rcx] 
xadd rdi, rcx 
and rdx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdx], -122 
sbb ax, di 
cmovs rdx, rdx 
bts rsi, 226 
and rdx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdx], sil 
test sil, dil 
and rsi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rsi], eax 
or rdi, rcx 
imul di 
lea ebx, qword ptr [rbx] 
and rcx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rbx], -48 
and rdx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdx], edx 
and rcx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rcx] 
xor ax, 10495 
setb dl 
cmpxchg bl, bl 
and rsi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rsi], -52 
sub rax, -727446025 
cmpxchg cl, al 
movzx cx, cl 
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
cmp al, -10 
lea esi, qword ptr [rdi] 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rcx] 
or dil, 109 
and rcx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rcx], -8 
cmovl rcx, rax 
bts rbx, rcx 
setb dil 
xor bl, 104 
cmpxchg ax, si 
or al, 1 # instrumentation
mov ax, 1 # instrumentation
div al 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx 
and rdx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rdx], ebx 
xadd dl, dl 
sbb bx, 85 
adc bl, 16 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add dl, al 
and rdx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rdx], di 
and rax, 0b1111111111111 # instrumentation
and cx, word ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rdi] 
setl cl 
and rdi, 0b1111111111111 # instrumentation
cmp cx, word ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rsi], 39 
add rdi, rax 
lea rbx, qword ptr [rdi + rsi + 57964] 
and rbx, 0b1111111111111 # instrumentation
cmovb rcx, qword ptr [r14 + rbx] 
lea bx, qword ptr [rcx + rdx + 7264] 
cmovb rcx, rdx 
bswap rdx 
and rax, 0b1111111111111 # instrumentation
cmovp di, word ptr [r14 + rax] 
not si 
setnbe dl 
xadd al, dl 
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], -14 
add al, dl 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx 
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], 1677401355 
and rcx, 0b1111111111111 # instrumentation
add ebx, dword ptr [r14 + rcx] 
setnz dil 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rdi], cx 
cmovz rcx, rdi 
movsx dx, bl 
and rbx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rbx], esi 
and rbx, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
btc word ptr [r14 + rbx], bx 
lea rax, qword ptr [rcx + rsi] 
and rdx, 0b1111111111111 # instrumentation
sbb ebx, dword ptr [r14 + rdx] 
and rcx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rcx], cl 
and rax, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rax] 
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
