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
and rbx, 0b1111111111111 # instrumentation
neg word ptr [r14 + rbx]
cmovnp bx, cx
test bl, dl
setnb dil
xor ax, 13522
xor cl, 29
and rdi, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rdi]
btr edi, ecx
and rbx, 0b1111111111111 # instrumentation
sbb rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rbx], -97
adc al, 36
and rdx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdx], ebx
xor ax, 8263
add ax, 22011
xor sil, -54
and rsi, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rsi], dx
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rbx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rbx]
mov rcx, rdx
and rax, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rax]
and si, -51
cmovle cx, di
and rdx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rdx], 0
add al, -33 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovp di, word ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdx], cl
setl dl
adc cl, bl
and rdi, 0b1111111111111 # instrumentation
cmovnbe rdx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
nop eax
xor al, bl
and rsi, 0b1111111111111 # instrumentation
cmp di, word ptr [r14 + rsi]
mov rdi, rdx
cbw
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rbx], dl
and rcx, 0b1111111111111 # instrumentation
add si, word ptr [r14 + rcx]
cmovle dx, ax
and rdi, 0b1111111111111 # instrumentation
add word ptr [r14 + rdi], ax
adc ax, 31677
and rcx, 0b1111111111111 # instrumentation
cmovnle bx, word ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rdi
xor ax, -4391
and rsi, 0b1111111111111 # instrumentation
cmovnz di, word ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
sub ebx, dword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmovnbe cx, word ptr [r14 + rsi]
adc rbx, rsi
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], sil
xor rax, 910747537
and rcx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rcx], -1794728388
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
add dl, -20 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
mov edi, 74087488
cmovo rcx, rax
and cl, bl
and rbx, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rdi], 29
lea eax, qword ptr [rdi + rsi]
adc sil, 21
add al, -127
imul cx, bx
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rcx]
add bl, 104 # instrumentation
setbe cl
adc sil, -112
bts si, 79
not cl
and rax, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rax], 6
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
btc rdi, rdi
add cl, -7 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovb edi, dword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmovnp ebx, dword ptr [r14 + rdx]
mov cx, -15625
and rdi, 0b1111111111111 # instrumentation
cmovns di, word ptr [r14 + rdi]
or cl, al
and rcx, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rcx]
mov dl, cl
not ecx
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], -14
and rcx, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rcx]
adc sil, -44
add rax, -971848644
and rax, 0b1111111111111 # instrumentation
cmovb dx, word ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rax]
or rax, rdx
cmovnz edx, ebx
cmovp rax, rcx
or sil, 123
lea rax, qword ptr [rax]
and rbx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rbx], eax
and rdi, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rdi], edx
add sil, 30
lea si, qword ptr [rcx + rcx]
and rdi, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rcx], si
and rcx, 0b1111111111111 # instrumentation
cmovb dx, word ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rsi], al
and rbx, 0b1111111111111 # instrumentation
cmovnbe ebx, dword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
adc di, word ptr [r14 + rsi]
xor ebx, edx
and rax, 0b1111111111111 # instrumentation
and qword ptr [r14 + rax], rdx
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
