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
.patch_placeholder:
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
and rbx, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rbx] 
lea di, qword ptr [rsi] 
and rax, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rax], -15 
lea si, qword ptr [rdx + rcx + 12555] 
sbb bl, al 
cmp rdx, rdi 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdi], ecx 
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 2 
not bl 
add al, cl 
and rdi, 0b1111111111111 # instrumentation
and word ptr [r14 + rdi], -16 
and rax, 0b1111111111111 # instrumentation
cmovno ax, word ptr [r14 + rax] 
setnle cl 
and rbx, 0b1111111111111 # instrumentation
cmovnbe ecx, dword ptr [r14 + rbx] 
lea rbx, qword ptr [rsi + rax + 28813] 
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], -76 
and rsi, 0b1111111111111 # instrumentation
cmovp eax, dword ptr [r14 + rsi] 
movsx rbx, bx 
and rcx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rcx], bl 
and rdi, 0b1111111111111 # instrumentation
movzx rcx, word ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], 10 
adc rbx, 71 
lea rbx, qword ptr [rsi + rsi + 13240] 
and rbx, 0b1111111111111 # instrumentation
cmovnl di, word ptr [r14 + rbx] 
mov al, bl 
cmpxchg rax, rdx 
and rcx, 0b1111111111111 # instrumentation
cmovnle esi, dword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rcx], al 
seto al 
sbb rbx, rsi 
and rdi, 0b1111111111111 # instrumentation
movsx cx, byte ptr [r14 + rdi] 
xor eax, 1643943753 
sbb al, al 
movsx edi, cx 
and rsi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rsi], 1 
inc eax 
or di, 0b1000000000000000 # instrumentation
bsf cx, di 
and rdi, 0b1111111111111 # instrumentation
cmp bx, word ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
and rax, qword ptr [r14 + rcx] 
and eax, -35 
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], al 
sbb rdi, rax 
sbb al, -91 
setnle sil 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsr eax, dword ptr [r14 + rax] 
movsx rdx, ax 
and rdx, 0b1111111111111 # instrumentation
imul bx, word ptr [r14 + rdx], -52 
and rcx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rcx], -21 
and rsi, 0b1111111111111 # instrumentation
cmovo edx, dword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmovb rcx, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rsi], 1 
dec cl 
and rbx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rbx], -96 
and rcx, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rcx], 63 
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], 48 
and rdx, 0b1111111111111 # instrumentation
imul rcx, qword ptr [r14 + rdx] 
sbb al, -87 
cmp dl, dl 
and rsi, 0b1111111111111 # instrumentation
cmp di, word ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rax], 4 
and rbx, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rbx], -45 
and ax, -45 
cwde  
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vmvictim_0:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_0:
.function_vm_1:
.bb_vm_1.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
add bl, -97 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rbx, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rdi], 5 
and rdx, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rdx], 3 
test bl, sil 
xor dil, 46 
setnz sil 
cmovz rdx, rdx 
and rdi, 0b1111111111111 # instrumentation
add edi, dword ptr [r14 + rdi] 
nop rbx 
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], dil 
xor edi, edx 
and rax, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rax], bx 
add rax, rsi 
lea rbx, qword ptr [rdi + rdi] 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], rsi 
and rdx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdx], rbx 
and rax, 0b1111111111111 # instrumentation
cmovno bx, word ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmovnbe ax, word ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rax] 
lea rax, qword ptr [rdi] 
and rdx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rdx], edi 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rdi], cl 
bt cx, bx 
and rax, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rax], rdi 
sbb bl, -67 
dec bl 
and rbx, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rbx], 46 
and rax, 0b1111111111111 # instrumentation
movzx bx, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi 
and rcx, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rbx], 2 
add cl, 125 # instrumentation
and rax, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
adc bl, byte ptr [r14 + rdx] 
xor rsi, 80 
and rdx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdx], cl 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi 
and rbx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rbx] 
mov al, dl 
and rdx, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rsi], 73 
and rdx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rdx] 
sbb ax, -13491 
and rdi, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdi] 
and ebx, edx 
and rdx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rdx], -76 
and rsi, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rsi] 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rax], edi 
and rcx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rcx], esi 
and rbx, 0b1111111111111 # instrumentation
xor rdi, qword ptr [r14 + rbx] 
lea rax, qword ptr [rdx + rdx + 29082] 
cmovbe si, cx 
cmp al, bl 
xor bl, al 
adc bl, bl 
setz al 
and rsi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rsi], rdx 
and rdi, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rdi] 
setb al 
and rbx, 0b1111111111111 # instrumentation
sub word ptr [r14 + rbx], 23 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rbx 
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_1:
.function_vm_2:
.bb_vm_2.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_2:
.section .data.main
.test_case_exit:nop
