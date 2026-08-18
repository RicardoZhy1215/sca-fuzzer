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
and dl, 37 
lea rcx, qword ptr [rax + rbx + 23576] 
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], 82 
cmovnb ecx, edx 
xor cl, bl 
and ebx, edx 
and rcx, 0b1111111111111 # instrumentation
cmovno rdi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rdi] 
lea cx, qword ptr [rax] 
and rbx, 0b1111111111111 # instrumentation
xor rax, qword ptr [r14 + rbx] 
setns cl 
and rcx, 0b1111111111111 # instrumentation
cmovz cx, word ptr [r14 + rcx] 
cmovl edi, ecx 
and rdi, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rdi] 
lea rsi, qword ptr [rbx] 
mov rbx, rsi 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rsi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rsi], esi 
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr esi, dword ptr [r14 + rbx] 
add al, 62 # instrumentation
and rax, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rax] 
cmovno cx, ax 
cmovnz edx, ecx 
and rsi, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rsi], 98 
and al, al 
and rax, 0b1111111111111 # instrumentation
cmovnp rax, qword ptr [r14 + rax] 
lea bx, qword ptr [rbx + rdi + 63927] 
sbb dil, 111 
adc al, cl 
adc sil, 111 
lea rdi, qword ptr [rsi] 
xor al, -30 
and rsi, 0b1111111111111 # instrumentation
mov ecx, dword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
add cl, 8 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovl eax, dword ptr [r14 + rbx] 
xadd dl, bl 
add bl, -88 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], 78 
sbb al, -28 
cmovno ecx, esi 
adc al, dl 
and rsi, 0b1111111111111 # instrumentation
movzx ebx, byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmovp di, word ptr [r14 + rdx] 
setbe bl 
imul ebx 
and edi, 88 
or sil, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv sil 
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], dl 
cmovnbe rsi, rbx 
cmovbe edx, edi 
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
mov sil, cl 
and rax, 0b1111111111111 # instrumentation
xor cl, byte ptr [r14 + rax] 
sub al, 32 
xadd edx, eax 
and rdi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdi], sil 
mov cl, dil 
cmp bl, cl 
or si, dx 
and rax, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rax], 93 
or si, 0b1000000000000000 # instrumentation
bsf dx, si 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi 
mov si, ax 
lea cx, qword ptr [rcx] 
and rdx, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rdx], 0 
lea ax, qword ptr [rdx] 
lea ax, qword ptr [rax + rdx + 1628] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, -32 # instrumentation
cmovnbe rax, rcx 
and rbx, 0b1111111111111 # instrumentation
and rsi, 0b1111111111111 # instrumentation
test word ptr [r14 + rsi], ax 
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rax] 
sbb esi, -81 
not cl 
and rbx, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rbx] 
adc eax, 1690630518 
and rbx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rbx], edi 
and rax, 0b1111111111111 # instrumentation
not word ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
imul rdx, qword ptr [r14 + rdi], -100 
and rdi, 0b1111111111111 # instrumentation
cmovno rbx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmp al, byte ptr [r14 + rdx] 
xor rdi, rdi 
and rdi, 0b1111111111111 # instrumentation
cmovp eax, dword ptr [r14 + rdi] 
lea esi, qword ptr [rbx] 
setz sil 
and rdx, 0b1111111111111 # instrumentation
cmovno cx, word ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rdx] 
lea rcx, qword ptr [rax] 
setz cl 
and rdi, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rdi] 
cdq  
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rsi] 
cmp al, 14 
and rax, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rax], dx 
or si, 0b1000000000000000 # instrumentation
bsr si, si 
add al, 45 # instrumentation
adc rax, 61 
lea rdx, qword ptr [rdx + rbx] 
or edi, 1 # instrumentation
and edx, edi # instrumentation
shr edx, 1 # instrumentation
div edi 
lea rcx, qword ptr [rsi + rax] 
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
