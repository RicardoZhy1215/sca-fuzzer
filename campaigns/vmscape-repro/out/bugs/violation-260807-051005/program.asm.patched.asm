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
test rdx, -2103517689
cmovnle di, ax
sub bl, 56
lea eax, qword ptr [rcx + rdx + 41940]
and rcx, 0b1111111111111 # instrumentation
or di, word ptr [r14 + rcx]
lea bx, qword ptr [rbx + rax]
sub al, -82
and rcx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rcx]
sub rdx, 11
adc dl, bl
cmovnb rsi, rax
imul di, cx
add al, 67 # instrumentation
lea ax, qword ptr [rax + rdx + 27140]
and rax, 0b1111111111111 # instrumentation
cmovnp rsi, qword ptr [r14 + rax]
cmovnp rdi, rcx
cmp al, 86
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111111 # instrumentation
xor bx, word ptr [r14 + rdx]
and dl, bl
lea rdi, qword ptr [rax + rcx]
mov dl, -20
btr eax, esi
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx]
add cl, -22 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdx]
sub cl, dl
and rsi, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rsi], -18
and rdx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdx]
cmovz cx, di
dec sil
setle bl
setb cl
xchg esi, edi
lea ecx, qword ptr [rax]
and rsi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
and cl, byte ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rdx], bx
lea bx, qword ptr [rdx + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovnp bx, word ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rsi], -110
and rcx, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
bt dword ptr [r14 + rcx], edi
and rdx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdx], rcx
test rbx, rsi
cmp cl, cl
and rcx, 0b1111111111111 # instrumentation
sbb bl, byte ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rbx], -128
and rbx, 0b1111111111111 # instrumentation
cmovnle eax, dword ptr [r14 + rbx]
sbb bx, si
and rcx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rcx], cl
and rdx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rdx], 123
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
cmp rax, 379066161
and rcx, 0b1111111111111 # instrumentation
cmovnbe ax, word ptr [r14 + rcx]
or bl, al
sub cl, dl
and rsi, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rsi], 1
lea edx, qword ptr [rdi + rdi + 51808]
and rdx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdx], cl
xor rbx, rcx
lea edi, qword ptr [rdi]
adc bl, al
sub rax, -68135907
and rdi, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rdi]
imul dx
cdq
lea esi, qword ptr [rcx + rsi]
and rdi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdi]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rsi], ax
btr cx, bx
and rax, 0b1111111111111 # instrumentation
and byte ptr [r14 + rax], bl
cmp rdi, -117
xchg bx, ax
cmovle esi, ecx
inc rax
and rdx, 0b1111111111111 # instrumentation
cmovp bx, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
nop word ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rcx], -14
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rdx], eax
and rdi, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rdi], 1
and rax, 0b1111111111111 # instrumentation
not qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
xor esi, dword ptr [r14 + rdi]
xor dx, cx
and rbx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rbx], di
clc
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], -67
sub sil, 79
and rdi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdi], sil
lea cx, qword ptr [rax + rbx]
bts bx, si
lea si, qword ptr [rsi]
and rcx, 0b1111111111111 # instrumentation
cmp edi, dword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmovp rdi, qword ptr [r14 + rdi]
cmovz rdi, rcx
and rcx, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rcx], 17
inc esi
lea edx, qword ptr [rdi + rcx]
and edx, 91
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
