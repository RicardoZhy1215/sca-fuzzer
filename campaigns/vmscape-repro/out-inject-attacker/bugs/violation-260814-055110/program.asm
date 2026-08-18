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
and rbx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rbx] 
cmpxchg cl, bl 
lea rdi, qword ptr [rdi + rsi + 41814] 
sub rax, -1946334910 
imul eax, eax, 21 
add al, 121 # instrumentation
cmovle eax, esi 
nop  
cmp eax, -1488023058 
and rsi, 0b1111111111000 # instrumentation
lock not qword ptr [r14 + rsi] 
bt edx, 84 
bts cx, si 
cmp sil, cl 
or dl, al 
cmovz rax, rax 
and di, di 
add eax, -380580706 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], -80 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], -48 
imul ebx 
mul dl 
bt rdi, 4 
add ax, 7275 
and rdx, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rdx] 
test cl, 111 
xor al, 84 
and rdx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdx], 75 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rcx] 
xor cl, al 
setns al 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsr rsi, qword ptr [r14 + rax] 
add dl, -121 # instrumentation
cmovl dx, cx 
and cx, si 
add cl, -50 
and rax, 0b1111111111111 # instrumentation
imul word ptr [r14 + rax] 
add bl, -4 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rcx] 
cmovs rcx, rsi 
and rdi, 0b1111111111111 # instrumentation
cmovo rsi, qword ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rax], ax 
and rsi, 0b1111111111111 # instrumentation
cmovs esi, dword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
adc dil, byte ptr [r14 + rax] 
xchg rsi, rbx 
and rbx, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
bt word ptr [r14 + rax], 6 
add sil, -78 
and rdx, 0b1111111111111 # instrumentation
cmp ax, word ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmp si, word ptr [r14 + rbx] 
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
add bl, 46 # instrumentation
cmovnz di, ax 
btr rbx, rax 
xor rax, 1225976609 
bt edi, edi 
and rdx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdx], 27 
cmovnl ecx, edi 
xadd edx, eax 
setnp cl 
cmovnb rdx, rbx 
and edi, 97 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
movsx rbx, bx 
or ebx, -97 
add al, 91 # instrumentation
cmovnl si, si 
lea esi, qword ptr [rdi + rsi] 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], cl 
cmovnle edi, edx 
cmovp di, dx 
and rdx, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
bt dword ptr [r14 + rdx], edx 
not rbx 
and rbx, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
bts word ptr [r14 + rbx], bx 
add cl, 2 # instrumentation
lea rdx, qword ptr [rcx + rsi] 
seto al 
lea rdx, qword ptr [rcx] 
and rdx, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bt qword ptr [r14 + rdx], rsi 
and rsi, 0b1111111111111 # instrumentation
cmovnb di, word ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rsi], bl 
imul al 
and rbx, 0b1111111111111 # instrumentation
btc word ptr [r14 + rbx], 7 
and rsi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rsi] 
imul rdx, rcx 
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
