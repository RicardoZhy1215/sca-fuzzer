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
add dl, 25 # instrumentation
setno bl
mov edx, edx
cmp bl, bl
setb al
and rdi, 0b1111111111111 # instrumentation
cmovbe ecx, dword ptr [r14 + rdi]
add dl, cl
cmovnp esi, ecx
setl al
sbb ecx, 10
lea ebx, qword ptr [rsi]
cld
lea cx, qword ptr [rdx + rsi]
lea ax, qword ptr [rsi + rax]
xor edx, -41
cmovz eax, esi
lea cx, qword ptr [rdi + rax + 1711]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111000 # instrumentation
and edi, 0b111 # instrumentation
lock bts dword ptr [r14 + rdx], edi
add bl, -78 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rsi], rbx
bt edx, ebx
xchg ebx, eax
and rdx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rdx], ecx
lea di, qword ptr [rcx + rbx]
lea si, qword ptr [rsi]
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rax], cx
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
xor dil, byte ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovnl edi, dword ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rdx], bx
adc bl, 106
sbb cl, -103
btr di, 170
imul edi, eax, -22
dec dl
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl
and cl, dl
sub cl, dl
imul bl
movzx eax, sil
and rcx, 0b1111111111111 # instrumentation
and cx, word ptr [r14 + rcx]
btr edx, esi
lea rsi, qword ptr [rcx + rdi]
add bl, 61
and rcx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rsi
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
xor bl, 48
movzx rbx, dil
and rdx, 0b1111111111111 # instrumentation
cmovnbe rax, qword ptr [r14 + rdx]
stc
bts rbx, rdi
and rbx, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rbx], 28
std
lea edx, qword ptr [rsi + rdx]
sbb sil, al
bt ax, si
adc ax, -5818
xor bl, -87
sbb cx, bx
nop rdi
adc al, 48
and rdi, 0b1111111111111 # instrumentation
adc rbx, qword ptr [r14 + rdi]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rcx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rcx]
lea rcx, qword ptr [rdi]
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], 57
lea ecx, qword ptr [rdi + rcx + 49509]
and rsi, 0b1111111111111 # instrumentation
btr word ptr [r14 + rsi], 0
and rcx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rcx], cl
and rsi, 0b1111111111111 # instrumentation
and dx, word ptr [r14 + rsi]
adc cx, si
and rdx, 0b1111111111111 # instrumentation
mov edx, dword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rcx]
imul rdi
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], 80
and rax, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rax], edx
and rdi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rdi], esi
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], rsi
xor rdx, rax
and rdx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rcx], 0b1000000000000000000000000000000 # instrumentation
bsr rsi, qword ptr [r14 + rcx]
add ax, 25050
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000000000000000 # instrumentation
bsr dx, word ptr [r14 + rdi]
add cl, -23 # instrumentation
and rax, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rax]
setnz dl
imul dx, bx
and rbx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rbx], 74
and rdx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdx], al
and rdi, 0b1111111111111 # instrumentation
sub bx, word ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovb edi, dword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], dil
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], cl
and rcx, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdi], esi
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
