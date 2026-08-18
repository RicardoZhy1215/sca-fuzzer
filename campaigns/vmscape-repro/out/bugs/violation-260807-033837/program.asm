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
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], -27 
lea di, qword ptr [rbx + rsi + 48059] 
bswap ecx 
cmp edx, 64 
add bl, cl 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], -96 
or di, 0b1000000000000000 # instrumentation
bsf dx, di 
btr rdx, rdx 
adc cl, bl 
movsx rcx, dl 
mov eax, 1982015301 
and rdi, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rdi], di 
add al, 99 
btr bx, di 
or bl, cl 
lea rdi, qword ptr [rcx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rcx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rcx] 
xor al, dl 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
and rbx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rbx], bx 
xor dl, -63 
and rcx, 0b1111111111111 # instrumentation
cmovnle cx, word ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rbx], 1 
add rdx, rdx 
and rax, 0b1111111111111 # instrumentation
mul word ptr [r14 + rax] 
add dl, -32 # instrumentation
and rcx, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rcx] 
cmovs rdi, rbx 
and rdx, 0b1111111111111 # instrumentation
and rbx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rbx], 28 
cmovno si, ax 
test eax, 1375112154 
adc eax, 927891714 
and rdx, 0b1111111111111 # instrumentation
cmovbe cx, word ptr [r14 + rdx] 
lea rdi, qword ptr [rdx + rdi + 28691] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
cmovno bx, ax 
mov rbx, rdi 
and rsi, 0b1111111111111 # instrumentation
xor rax, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
test dword ptr [r14 + rax], edx 
and rdi, 0b1111111111111 # instrumentation
cmovno esi, dword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovbe edx, dword ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
imul bx, word ptr [r14 + rax] 
cmovno di, ax 
and rcx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rcx], cx 
sbb rax, 615509617 
setb al 
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
imul eax, edi, 30 
add cl, -66 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovns di, word ptr [r14 + rbx] 
xor al, 27 
cmovle ebx, edi 
cmovnp edx, ecx 
adc si, 77 
lea ecx, qword ptr [rdi] 
or dx, 62 
movzx eax, si 
lea dx, qword ptr [rsi + rax + 17571] 
imul dil 
inc dl 
and rdi, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rdi] 
bts cx, 216 
add cl, -6 # instrumentation
setno dil 
cmovnbe rax, rsi 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add bl, 110 # instrumentation
cwd  
and rbx, 0b1111111111111 # instrumentation
cmovns eax, dword ptr [r14 + rbx] 
lea rbx, qword ptr [rdx + rdi + 27745] 
and rax, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rax], -81 
neg bl 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 0b1000000000000000 # instrumentation
bsr di, word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
adc ax, word ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rbx], 34 
and rdx, 0b1111111111111 # instrumentation
cmovs bx, word ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmovnb esi, dword ptr [r14 + rdx] 
setnl bl 
xor edi, 43 
adc al, dl 
lea rax, qword ptr [rsi] 
lea di, qword ptr [rsi + rcx] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], 66 
lea rsi, qword ptr [rdi + rbx + 58999] 
and rdx, 0b1111111111111 # instrumentation
xor edx, dword ptr [r14 + rdx] 
bts ecx, edx 
add sil, al 
mov rsi, 6362711967353493974 
test dl, bl 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rsi 
cmovnz rdx, rdi 
and rax, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rax], di 
and rbx, 0b1111111111111 # instrumentation
cmovz ax, word ptr [r14 + rbx] 
mul rcx 
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], dl 
and rdx, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rdx], 5 
and rax, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rax], rdi 
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
