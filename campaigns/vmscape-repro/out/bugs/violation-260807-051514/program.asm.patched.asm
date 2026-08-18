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
and rcx, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rbx], 1
xadd esi, ecx
and rbx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rbx], bl
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], -27
nop esi
setnp al
xor edi, -45
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx]
add bl, -96 # instrumentation
setl bl
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], bx
sub rax, -113
lea rbx, qword ptr [rdx + rdi + 5788]
lea rax, qword ptr [rdx]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
add bl, cl
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, 60 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovle si, word ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx]
xchg ecx, edx
and rsi, 0b1111111111111 # instrumentation
or rbx, qword ptr [r14 + rsi]
sbb ax, cx
and rbx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rbx], rax
btc bx, dx
add bl, 42 # instrumentation
cmovo bx, cx
and rsi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rsi], -121
xor dl, cl
cwde
or ecx, 1 # instrumentation
and edx, ecx # instrumentation
shr edx, 1 # instrumentation
div ecx
cmpxchg al, al
and rsi, 0b1111111111111 # instrumentation
add word ptr [r14 + rsi], ax
cmpxchg cx, bx
and rcx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rcx], 7
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 1 # instrumentation
and edx, dword ptr [r14 + rax] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rax]
add bl, 117 # instrumentation
and rcx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rcx]
add al, -118
and rdx, 0b1111111111111 # instrumentation
cmovz rdx, qword ptr [r14 + rdx]
movzx ebx, cx
sbb al, -40
and rdx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rdx]
nop edi
and rdx, 0b1111111111111 # instrumentation
xor esi, dword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
add bl, 114 # instrumentation
lea ebx, qword ptr [rdx + rax]
and rdi, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdi], -23
and rsi, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rsi], di
add cl, dil
movzx dx, bl
and rdx, 0b1111111111111 # instrumentation
test word ptr [r14 + rdx], 31707
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
cmp sil, dl
xor bl, al
bt di, cx
add dl, 109 # instrumentation
movzx ecx, al
and rbx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rbx]
add cl, 88 # instrumentation
lea esi, qword ptr [rdi + rdx + 29849]
setnle cl
sbb cl, dl
btc rdi, rcx
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rdi
and rax, 0b1111111111111 # instrumentation
cmovnz dx, word ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
and word ptr [r14 + rax], dx
bt eax, ecx
movzx rsi, cl
mov dx, -1490
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea ebx, qword ptr [rax + rdi + 31344]
and rdx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdx], ax
sub bl, 22
movsx rcx, di
and rsi, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rsi], dl
imul di, cx, 32
and rcx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rcx], 5
inc dl
and rbx, 0b1111111111111 # instrumentation
or dl, byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovo cx, word ptr [r14 + rax]
lea cx, qword ptr [rsi + rdx + 42713]
and rsi, 0b1111111111111 # instrumentation
cmovz edx, dword ptr [r14 + rsi]
mul cx
adc rdx, -2
and rbx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rbx], al
sub dl, cl
and rsi, 0b1111111111111 # instrumentation
adc word ptr [r14 + rsi], 52
and dl, dl
cmp dl, bl
and rcx, 0b1111111111111 # instrumentation
cmovo ecx, dword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rdi], 7
and rsi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rsi], 79
and rcx, 0b1111111111111 # instrumentation
bts word ptr [r14 + rcx], 4
lea rcx, qword ptr [rdi + rdx]
and cx, 31
xor dl, al
and rdi, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdi], -59
and rdi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdi], -86
and rax, 0b1111111111111 # instrumentation
add dword ptr [r14 + rax], 44
and rax, 0b1111111111111 # instrumentation
adc rdx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rdi], 6
and rdx, 0b1111111111000 # instrumentation
lock not qword ptr [r14 + rdx]
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
