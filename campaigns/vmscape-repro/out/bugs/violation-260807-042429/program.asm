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
and cl, byte ptr [r14 + rsi] 
sub dil, 20 
btr si, cx 
add bl, -3 # instrumentation
cmovns bx, cx 
xor al, 57 
adc rax, 587212538 
bts eax, 173 
add sil, 10 
cmovp cx, ax 
cmovnbe rdx, rsi 
and rax, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rax], di 
cmovl ebx, ecx 
and rdi, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rdi] 
sbb bl, -103 
cmpxchg cl, al 
lea rcx, qword ptr [rbx + rbx + 34089] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, 33 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovnb ebx, dword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmovns ax, word ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rax], edi 
sub eax, ebx 
and rbx, 0b1111111111111 # instrumentation
cmovo bx, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
movsx esi, word ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rcx], 2 
and rcx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rcx] 
movsx rcx, al 
cmovnle rax, rbx 
and rdi, 0b1111111111111 # instrumentation
cmovnz rdx, qword ptr [r14 + rdi] 
mov dil, -53 
and rsi, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rsi], 61 
xor bl, -127 
cmp cl, cl 
cmovns si, bx 
and rbx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rbx], al 
add eax, -225234550 
bswap eax 
lea esi, qword ptr [rcx] 
test si, -14131 
add dl, 91 
xadd esi, esi 
mov dl, dl 
cmovnle rbx, rdi 
cmovno edi, ecx 
bt rsi, rax 
sbb dil, 105 
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], 124 
and rax, 0b1111111111111 # instrumentation
cmovnp rsi, qword ptr [r14 + rax] 
or rcx, 0b1000000000000000000000000000000 # instrumentation
bsf rdi, rcx 
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
add cl, 122 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovbe rcx, rdi 
test ebx, -1663050872 
sbb bl, 80 
xor cl, 10 
cmovbe esi, esi 
and rcx, 0b1111111111111 # instrumentation
sub dx, word ptr [r14 + rcx] 
setno sil 
mov rbx, rdi 
std  
lea rbx, qword ptr [rbx + rbx + 49920] 
cmovs rdi, rsi 
and rax, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rax], 1 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsf rdx, qword ptr [r14 + rdi] 
dec eax 
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl 
add cl, 59 # instrumentation
adc dl, sil 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdx, 0b1111111111111 # instrumentation
add word ptr [r14 + rdx], 118 
sbb rdi, -83 
xor bl, cl 
and rdx, 0b1111111111111 # instrumentation
sbb cl, byte ptr [r14 + rdx] 
adc al, -61 
stc  
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], 48 
and rcx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rcx], -81 
setnbe cl 
and rcx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rcx], bl 
and rsi, 0b1111111111111 # instrumentation
add ebx, dword ptr [r14 + rsi] 
movsx rbx, bx 
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], dl 
and rcx, 0b1111111111111 # instrumentation
add dx, word ptr [r14 + rcx] 
xor cl, al 
and rdi, 0b1111111111111 # instrumentation
movsx esi, word ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], -10 
sub eax, 1245256202 
and rcx, 0b1111111111111 # instrumentation
and ebx, 0b111 # instrumentation
btc dword ptr [r14 + rcx], ebx 
dec bl 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rax] 
add cl, 62 # instrumentation
sbb rax, 1343064293 
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx] 
mov dl, bl 
and rdx, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rdx], 4 
or esi, 73 
xor dl, cl 
and rsi, 0b1111111111111 # instrumentation
or eax, dword ptr [r14 + rsi] 
xor ax, -1821 
and rdi, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdi] 
lea edi, qword ptr [rcx + rax] 
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
