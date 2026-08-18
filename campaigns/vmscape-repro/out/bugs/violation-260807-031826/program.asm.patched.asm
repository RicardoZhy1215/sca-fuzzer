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
add dl, -77 # instrumentation
seto sil
and rsi, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rsi]
cmovb cx, dx
cmpxchg bl, dl
adc rax, 1408383961
sbb ebx, 40
mov cl, bl
cmovle cx, bx
movzx esi, al
setl al
and rcx, 0b1111111111111 # instrumentation
cmovnle ebx, dword ptr [r14 + rcx]
lea edx, qword ptr [rbx]
and rax, -2143775289
xadd al, sil
cmovbe rdi, rax
and rdi, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rdi], 4
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add edi, 39
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rcx], 5
and rax, -77
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 1 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx]
stc
or ebx, 1 # instrumentation
and edx, ebx # instrumentation
shr edx, 1 # instrumentation
div ebx
and rdx, 0b1111111111111 # instrumentation
movzx rsi, word ptr [r14 + rdx]
cmp al, sil
lea ecx, qword ptr [rsi + rdx]
and rbx, 0b1111111111111 # instrumentation
mov bx, word ptr [r14 + rbx]
and cl, -82
and rbx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rdi], 126
and rdi, 0b1111111111111 # instrumentation
adc si, word ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rcx], 3
and rbx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rbx], bl
cmovnb ebx, esi
and rdx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdx], si
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], cl
or rsi, rsi
xadd cl, cl
and rsi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rsi], dx
mov dl, al
and rsi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rsi]
lea esi, qword ptr [rcx + rsi + 8313]
mul ecx
adc dl, al
imul ecx, eax
add al, cl
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
btc edx, ecx
add dl, -95 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnp cx, word ptr [r14 + rax]
lea rbx, qword ptr [rdx + rdx]
cmovp ax, cx
or ax, 0b1000000000000000 # instrumentation
bsr dx, ax
xor dx, si
and rdi, 0b1111111111111 # instrumentation
cmovz dx, word ptr [r14 + rdi]
lea di, qword ptr [rax + rbx]
mov cl, 106
add cx, -71
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi]
cmovnb edx, edi
mov cl, al
and rdx, 0b1111111111111 # instrumentation
adc rdi, qword ptr [r14 + rdx]
test rax, -103537838
cmovns rsi, rax
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add bl, 95 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdi], dil
sbb rax, -2021403108
and rdx, -36
movsx si, dil
and rdi, 0b1111111111111 # instrumentation
adc rcx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovo di, word ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], -94
cmovl rdx, rax
inc si
and rax, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rax], 93
lea edx, qword ptr [rbx]
mov bx, 9623
not al
and rbx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdi], 42
and rsi, 0b1111111111111 # instrumentation
xor dil, byte ptr [r14 + rsi]
xor dl, bl
lea ecx, qword ptr [rax + rdx]
xor cl, dl
and rax, 0b1111111111111 # instrumentation
or rdi, qword ptr [r14 + rax]
or al, dl
stc
and rdx, 0b1111111111111 # instrumentation
not word ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rcx]
and al, 118
setnp dl
and rax, 0b1111111111111 # instrumentation
cmovp ebx, dword ptr [r14 + rax]
cmp dl, cl
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
and rcx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rdi], 1
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
