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
add al, -102 # instrumentation
cmovno rbx, rdi
cmp bx, 21
cmovnp rdi, rdi
and rax, 0b1111111111111 # instrumentation
not dword ptr [r14 + rax]
add al, cl
and rdx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdx], bl
cmovns di, si
sbb bl, dl
and rdi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdi]
sbb dl, bl
setnbe dl
add dl, 77
add eax, edx
cmovnb ecx, edi
movzx rdx, dx
and rbx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rbx], ax
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
xadd ebx, eax
xor cl, bl
sets al
or si, dx
sbb dl, 93
and rsi, 0b1111111111111 # instrumentation
movzx edi, byte ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rcx]
std
and ax, 7587
and rcx, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rcx], ebx
and rdi, 0b1111111111111 # instrumentation
movzx rsi, word ptr [r14 + rdi]
test rax, 940549158
or ecx, 1 # instrumentation
and edx, ecx # instrumentation
shr edx, 1 # instrumentation
div ecx
add dl, 48 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], 1899714279
and rax, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rax], -36
and rcx, 0b1111111111111 # instrumentation
cmovb dx, word ptr [r14 + rcx]
cmovl ax, cx
setno sil
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx]
add cl, cl
cmovl rax, rbx
std
xchg cl, al
and rsi, 0b1111111111111 # instrumentation
adc cx, word ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rcx], bl
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], 69
setp dil
and rsi, 0b1111111111111 # instrumentation
imul edi, dword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
sub al, 106
and rsi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rsi], -50
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
add al, 87 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovp rcx, rdi
and al, 79
cmovs bx, cx
and rax, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rax]
add dx, bx
cmovl rax, rsi
and rsi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdi
neg dl
bswap rcx
mov cl, al
and rcx, rdi
and rdi, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rdi], 0
add dl, 7 # instrumentation
setno dil
setnb al
test rax, -1516581129
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, -106 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rdi], 116
and sil, dil
or rbx, 0b1000000000000000000000000000000 # instrumentation
bsf rdx, rbx
add al, 39 # instrumentation
lea esi, qword ptr [rdx]
and rdx, 0b1111111111111 # instrumentation
cmovbe edi, dword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rsi], 2
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], 57
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], rdx
and rdx, 0b1111111111111 # instrumentation
cmovo ax, word ptr [r14 + rdx]
movsx ecx, dx
and rbx, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rbx], di
xadd cl, cl
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rsi]
sub cl, cl
and rdi, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rdi]
sub al, -27
clc
sbb dl, bl
test rsi, rcx
and rdx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rdx]
and rdx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rdx], 7
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsf ecx, dword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rdi], 78
test cl, 51
and rbx, 0b1111111111111 # instrumentation
cmovle ecx, dword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rcx]
adc bx, si
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000000000000000 # instrumentation
bsf bx, word ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 1 # instrumentation
and dx, word ptr [r14 + rbx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rbx]
add al, 3 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovle esi, dword ptr [r14 + rbx]
or al, cl
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
