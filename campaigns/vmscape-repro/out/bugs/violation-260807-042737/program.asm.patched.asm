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
and ebx, 0b111 # instrumentation
bt dword ptr [r14 + rsi], ebx
add al, -38 # instrumentation
cmovno rdi, rdx
or edi, eax
or edi, -126
adc cx, ax
add di, -124
setl sil
sub cl, dl
btr cx, 86
and rsi, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rsi]
lea rsi, qword ptr [rsi]
setnbe dl
mov dl, cl
and rbx, 0b1111111111111 # instrumentation
adc edx, dword ptr [r14 + rbx]
add cx, 10
and rax, 0b1111111111000 # instrumentation
and cx, 0b111 # instrumentation
lock btr word ptr [r14 + rax], cx
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, 35 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rsi]
cmovbe di, ax
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rcx]
cmp rax, -242557696
lea di, qword ptr [rcx]
and rsi, 0b1111111111111 # instrumentation
movzx cx, byte ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovl cx, word ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovnbe ecx, dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
add dl, -97 # instrumentation
mov cl, al
setnle dl
and rax, 0b1111111111111 # instrumentation
mov word ptr [r14 + rax], -20893
and rbx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rbx], edi
lea rdi, qword ptr [rdi + rdx]
bt edi, ebx
cmovnz ax, ax
and rsi, 0b1111111111111 # instrumentation
adc al, byte ptr [r14 + rsi]
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
and rax, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
btr word ptr [r14 + rdx], 3
lea edx, qword ptr [rdi + rsi + 35029]
xchg ax, dx
add dl, -37
and rax, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rax], 6
add bl, -91 # instrumentation
setle bl
and rbx, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rbx]
cmp cl, cl
and al, -73
and rdx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdx], cl
and rdi, 0b1111111111111 # instrumentation
and ax, 0b111 # instrumentation
btc word ptr [r14 + rdi], ax
and rdi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdi], bl
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
add bl, -71 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
xchg di, ax
lea cx, qword ptr [rbx]
cmovnp di, bx
cmovnbe ecx, edx
test dl, 125
and rcx, 0b1111111111111 # instrumentation
and dword ptr [r14 + rcx], 84
and rcx, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rcx], 0
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], -326270850
cwd
and rcx, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rcx], 7
or rax, 0b1000000000000000000000000000000 # instrumentation
bsf rdx, rax
and rax, 0b1111111111111 # instrumentation
xor cx, word ptr [r14 + rax]
adc ax, -26883
adc ax, -490
cmp dl, bl
adc bl, dl
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, 105 # instrumentation
and rsi, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rsi]
movzx edx, al
and rax, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rax], -122
and rcx, 0b1111111111111 # instrumentation
cmovnbe ebx, dword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
add cl, -116 # instrumentation
and rax, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rax], edx
and rsi, 0b1111111111111 # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi]
inc bx
and rdx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rdx], rax
not si
bts edi, edx
cmp rbx, 90
cmovnz rcx, rdx
cmovs rax, rbx
setz cl
and rdi, 0b1111111111111 # instrumentation
sub ax, word ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
sub cx, word ptr [r14 + rdi]
cmovs si, cx
and rdx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rbx], sil
inc cl
and rbx, 0b1111111111111 # instrumentation
cmp bx, word ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovnz di, word ptr [r14 + rsi]
inc edx
and rcx, 0b1111111111111 # instrumentation
movsx ecx, byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
test word ptr [r14 + rcx], 23584
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 0b1000 # instrumentation
and byte ptr [r14 + rdi], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rdi]
add dl, 79 # instrumentation
sbb al, -21
and rax, 0b1111111111111 # instrumentation
cmovz cx, word ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
and word ptr [r14 + rbx], -82
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
