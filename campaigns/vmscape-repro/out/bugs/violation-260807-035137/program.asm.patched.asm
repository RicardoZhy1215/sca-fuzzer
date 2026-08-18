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
test bl, -94
lea rcx, qword ptr [rdx + rdx + 31296]
imul bl
add cl, -96 # instrumentation
cmovnbe rcx, rdi
and rdi, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rdi], bl
lea cx, qword ptr [rax + rcx + 51250]
sub ecx, ecx
adc rsi, rbx
and rsi, 0b1111111111111 # instrumentation
or al, byte ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdi], eax
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 1 # instrumentation
and edx, dword ptr [r14 + rbx] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rbx]
and di, cx
xadd edx, eax
sbb eax, ebx
cmovnb rdi, rcx
setb al
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
cmp bx, si
btc rax, rdx
and rbx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rdi], ax
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], 4
and rdi, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rdi]
cmpxchg rdx, rcx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rdx
sbb rax, 1566238720
and al, dl
and rax, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rax], -128
and rax, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rax], 6
xchg si, si
or rbx, -80
sub dl, 82
mov al, 75
not al
and rdx, 0b1111111111111 # instrumentation
not dword ptr [r14 + rdx]
setb bl
and rsi, 0b1111111111111 # instrumentation
and rsi, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
movsx rcx, word ptr [r14 + rsi]
neg esi
and rcx, 0b1111111111111 # instrumentation
and rbx, 0b111 # instrumentation
bts qword ptr [r14 + rcx], rbx
sbb bl, 90
and rax, 0b1111111111111 # instrumentation
cmovbe ecx, dword ptr [r14 + rax]
sub eax, -1085029147
and rsi, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rsi]
or cx, 0b1000000000000000 # instrumentation
bsf si, cx
lea ecx, qword ptr [rbx]
and rcx, 0b1111111111111 # instrumentation
cmovnz edi, dword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rcx], dil
mov dil, -127
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
sub rdi, -9
btr di, di
xor cl, -67
movsx edx, bl
sub esi, edx
or cx, 0b1000000000000000 # instrumentation
bsf ax, cx
test dil, 44
cmovz rbx, rax
and rbx, 0b1111111111111 # instrumentation
cmovnbe di, word ptr [r14 + rbx]
lea ecx, qword ptr [rbx]
movzx rsi, cx
sub cl, sil
bt rsi, 50
cmp rax, -1570052073
sub cl, -123
add al, dl
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx]
or al, cl
and rcx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rcx], -66
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], 63
lea di, qword ptr [rdi]
and rdx, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
and ax, word ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovnle esi, dword ptr [r14 + rsi]
movsx cx, cl
and rbx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rbx], 5
and rbx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
bt dword ptr [r14 + rcx], edx
add bl, -61 # instrumentation
and rdx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rdx]
movzx edi, cl
and rdx, 0b1111111111111 # instrumentation
cmovo si, word ptr [r14 + rdx]
btr rdx, 192
sbb ax, -31001
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rcx]
mov bl, dl
and rsi, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rsi], -22
and rcx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rcx], 6
and rdi, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rdi], 4
add dl, 85 # instrumentation
setno dil
lea cx, qword ptr [rax]
sub al, -2
and rsi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rsi], 52
and rdx, 0b1111111111111 # instrumentation
cmovp dx, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rax], 0
movzx rsi, bl
movsx rdi, bx
mul rax
add al, -30 # instrumentation
and rdi, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdi]
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
