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
add dl, 5 # instrumentation
cmovb esi, edx
and rcx, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rcx]
and al, bl
imul bx, ax, -103
add al, 10 # instrumentation
cmovns rdi, rsi
or dl, dl
and rdx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdx], eax
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], al
setl dil
or bl, bl
and rdi, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rdi], 6
and rbx, 0b1111111111111 # instrumentation
sub cl, byte ptr [r14 + rbx]
bts ax, 8
add dl, -40 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rcx]
lea eax, qword ptr [rcx]
imul sil
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
xor al, 13
and rax, 0b1111111111111 # instrumentation
sub ebx, dword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rcx], edi
sub al, 55
and rdx, 0b1111111111111 # instrumentation
and rdi, 0b111 # instrumentation
btc qword ptr [r14 + rdx], rdi
xor cl, cl
and rdi, 0b1111111111111 # instrumentation
and rdx, 0b1111111111111 # instrumentation
xor sil, byte ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdx], dl
and rax, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], dx
adc ebx, edi
cmovl cx, bx
and rax, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
imul ax, word ptr [r14 + rcx]
add cl, 27 # instrumentation
setnbe al
add eax, -1772542436
and rdi, 0b1111111111111 # instrumentation
add bl, byte ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
adc rcx, qword ptr [r14 + rdi]
lea ecx, qword ptr [rdx + rcx]
and rdi, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
and word ptr [r14 + rax], 30
cmovns bx, bx
cmovb rdx, rcx
and rdx, 0b1111111111111 # instrumentation
sbb ax, word ptr [r14 + rdx]
btr di, di
and rcx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rcx], ecx
and rcx, 0b1111111111111 # instrumentation
cmovno rsi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
or cl, byte ptr [r14 + rbx]
lea ebx, qword ptr [rbx + rsi + 27059]
and rdx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rdx]
sub al, bl
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
test byte ptr [r14 + rbx], -36
and rcx, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rcx]
cmovl rdi, rdx
lea edi, qword ptr [rcx + rdi + 30541]
and rdi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rdi], ax
test bl, 49
cmp eax, 857810652
cmovp cx, dx
lea cx, qword ptr [rbx + rdi + 35448]
and rax, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rax], cl
cmovs rax, rsi
sbb dl, dl
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], 24
cmovp esi, ebx
lea rdi, qword ptr [rdx + rbx]
btr cx, 196
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
sub al, cl
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], 53
and rdi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdi], 12
dec rbx
btr di, dx
and cx, 0
mov di, ax
bswap rbx
inc ebx
xchg eax, eax
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], 2
lea ecx, qword ptr [rax + rdx + 28358]
or si, 0b1000000000000000 # instrumentation
bsr dx, si
mov sil, -127
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], -55
and rsi, 0b1111111111111 # instrumentation
xor eax, dword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
or eax, -2142325427
and bl, 111
and rcx, 0b1111111111111 # instrumentation
xor rax, qword ptr [r14 + rcx]
btc ax, 168
imul rdi, rdi, -111
xor ax, 22410
and rbx, 0b1111111111111 # instrumentation
cmovle edx, dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rdx]
lea bx, qword ptr [rdi]
bswap ebx
and rcx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rcx], bl
and rcx, 0b1111111111111 # instrumentation
cmovle edi, dword ptr [r14 + rcx]
lea rsi, qword ptr [rbx + rdi + 40413]
and rsi, 0b1111111111111 # instrumentation
cmovb si, word ptr [r14 + rsi]
btr di, 39
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
