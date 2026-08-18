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
add cl, -68 # instrumentation
lea esi, qword ptr [rdx + rdx + 40073]
adc eax, -468221008
and rcx, 0b1111111111111 # instrumentation
cmpxchg word ptr [r14 + rcx], di
and rdi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdi], ax
xor di, cx
and rsi, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rsi]
setle dil
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx]
lea di, qword ptr [rax + rsi + 30381]
lea rdi, qword ptr [rbx + rcx]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax
test dl, cl
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], al
movsx ax, bl
and rbx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rax]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
lea bx, qword ptr [rbx]
or eax, 956944323
and rax, 0b1111111111111 # instrumentation
cmovnbe cx, word ptr [r14 + rax]
xchg bx, bx
and dl, 53
or dl, dl
and rdi, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
and rdi, qword ptr [r14 + rcx]
add bl, dl
or di, -128
and rcx, 0b1111111111111 # instrumentation
add dil, byte ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rsi]
mov dx, dx
sub al, dl
and rdx, 0b1111111111111 # instrumentation
sub rsi, qword ptr [r14 + rdx]
and ax, -21184
xadd dx, bx
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
cmpxchg dx, dx
and eax, ebx
setnle al
and rsi, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rsi], edx
add al, -104
cmpxchg bl, cl
cmp di, -108
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl
and rdi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rdi], 3
xadd dl, al
and rsi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsr rcx, qword ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rdx]
imul cx
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
add cl, 48 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovns ecx, eax
or al, dl
sets dl
test dl, 113
adc eax, 106
and rsi, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rsi]
mov cl, -52
and rax, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rax]
imul ebx
lea rdi, qword ptr [rcx + rdx]
cmp cl, dl
setz cl
and rcx, 0b1111111111111 # instrumentation
imul ebx, dword ptr [r14 + rcx], 11
add dl, -16 # instrumentation
cdq
sets cl
and rdx, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rdx]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea di, qword ptr [rsi]
cmp al, al
lea si, qword ptr [rcx + rsi + 18524]
and rax, 0b1111111111111 # instrumentation
not byte ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmp bx, word ptr [r14 + rcx]
sub ax, -25800
and rdi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rdi], dl
and rcx, 0b1111111111111 # instrumentation
cmp edi, dword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], -38
add esi, 75
movzx rbx, bx
cmovb eax, ebx
or ax, -16336
adc rbx, rcx
lea rdx, qword ptr [rax + rdx]
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 0b1000 # instrumentation
and byte ptr [r14 + rbx], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rbx]
add al, -121 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovp ebx, dword ptr [r14 + rdx]
movsx bx, dil
and rax, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rdi], rdi
and rbx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rbx], al
and rdi, 0b1111111111111 # instrumentation
cmovno ecx, dword ptr [r14 + rdi]
xadd dl, bl
inc ecx
and rsi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf si, word ptr [r14 + rsi]
add al, -81 # instrumentation
and rsi, 0b1111111111111 # instrumentation
not byte ptr [r14 + rsi]
lea rdx, qword ptr [rdx + rsi]
sbb rdi, 98
and rdi, 0b1111111111111 # instrumentation
cmovno ebx, dword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf eax, dword ptr [r14 + rax]
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
