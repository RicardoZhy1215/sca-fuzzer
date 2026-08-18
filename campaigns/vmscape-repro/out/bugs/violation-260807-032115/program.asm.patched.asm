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
add dl, -45 # instrumentation
sbb dl, dl
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rdx], edi
or bl, bl
bt bx, 97
sub di, di
test cl, cl
xchg rcx, rdx
adc di, di
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], 43
cmp dl, 104
and rdx, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rdx], di
and edx, -70
adc ax, -26217
add cl, 104
bswap rbx
cmc
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, 56 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovno eax, dword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rsi], ecx
and rcx, 0b1111111111111 # instrumentation
cmovno edx, dword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdx]
sub al, bl
movsx rbx, bx
cdq
sub cl, al
setle al
and rsi, 0b1111111111111 # instrumentation
and dl, byte ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rcx
dec al
add bl, al
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], -92
setl sil
cmovnl rdi, rbx
and rsi, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rsi], dl
and rbx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rbx], bl
and rdx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdx], -125
and rdi, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdi], dil
adc rdi, rsi
and rsi, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rsi]
adc bl, dl
and rdi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdi], esi
adc rax, -512664969
and rbx, 0b1111111111111 # instrumentation
dec word ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], rax
and rdx, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdx], esi
and rsi, 0b1111111111111 # instrumentation
or si, word ptr [r14 + rsi]
xchg edi, ecx
and rbx, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rbx]
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
add cl, 12 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setnz dl
imul ebx, esi, 66
add cl, -94 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovz rax, qword ptr [r14 + rcx]
or dil, 115
lea bx, qword ptr [rsi + rax + 38047]
cmpxchg dl, cl
and rax, 0b1111111111111 # instrumentation
and qword ptr [r14 + rax], rdi
and rbx, 0b1111111111111 # instrumentation
xor rdi, qword ptr [r14 + rbx]
setns dl
cmovnle rax, rbx
cmp ax, -1038
cmovnbe rsi, rdi
and ax, dx
and rax, 0b1111111111111 # instrumentation
test word ptr [r14 + rax], 2514
xor rdx, rcx
cmp esi, 8
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add bl, 10 # instrumentation
cmovnp eax, edi
cmovnz rdi, rdx
and rbx, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rbx], -52
cld
imul cl
add bl, 107 # instrumentation
mov ebx, 1182491401
and rdx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rdx], dl
and rcx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rcx]
cld
or rax, 1901826585
lea esi, qword ptr [rdx + rax]
and rax, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
bt dword ptr [r14 + rax], edx
and rdx, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rdx], esi
cmovnb rsi, rsi
and rdx, 0b1111111111111 # instrumentation
and rdx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rbx]
or ax, 1 # instrumentation
and dx, ax # instrumentation
shr dx, 1 # instrumentation
div ax
and rsi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rsi], edx
and rsi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rsi], bl
and rcx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rcx], 0b1000000000000000000000000000000 # instrumentation
bsf rbx, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdx], 214351885
cmc
or cx, 0b1000000000000000 # instrumentation
bsr bx, cx
add al, -22 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovs esi, dword ptr [r14 + rsi]
adc al, 126
and rcx, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rcx]
add al, -88 # instrumentation
lea rdi, qword ptr [rbx]
cmovnl rdx, rbx
sbb bl, dl
sbb cl, al
and rbx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rbx], edx
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
