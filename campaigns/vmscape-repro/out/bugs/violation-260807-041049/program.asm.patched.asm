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
add dl, -38 # instrumentation
inc dx
setnle dl
and rax, 0b1111111111111 # instrumentation
movsx edx, word ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rax], ecx
and rdx, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rdx]
cmpxchg dx, cx
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsr eax, dword ptr [r14 + rdi]
add dl, 82 # instrumentation
lea cx, qword ptr [rcx + rsi]
setl dl
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rcx]
or al, -35
imul dx
xor rax, 142895502
imul esi, edx, -28
and rax, 0b1111111111111 # instrumentation
or cx, word ptr [r14 + rax]
lea ecx, qword ptr [rcx + rbx + 26808]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rcx, 0b1111111111111 # instrumentation
xor dl, byte ptr [r14 + rcx]
cmovnb esi, ecx
xchg di, ax
and rdi, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdi], -104
cmovb rsi, rdx
test rcx, -1513557164
test rdx, rbx
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], dl
lea di, qword ptr [rdx + rsi]
or esi, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv esi
add cl, -13 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovbe eax, dword ptr [r14 + rbx]
btc rsi, rax
lea esi, qword ptr [rax + rsi]
and rdi, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
btc word ptr [r14 + rsi], 1
add cl, -12 # instrumentation
cmovp rbx, rcx
cmovnbe ecx, edx
and rcx, 0b1111111111111 # instrumentation
and ax, word ptr [r14 + rcx]
and ax, 25567
sub al, -26
dec ax
lea ecx, qword ptr [rdx + rsi]
and rcx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rcx], dl
and rsi, 0b1111111111111 # instrumentation
cmovp rdx, qword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
and rsi, 0b111 # instrumentation
lock btr qword ptr [r14 + rsi], rsi
add bl, 7 # instrumentation
and rax, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmovo dx, word ptr [r14 + rdx]
btr edi, esi
setbe dl
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
lea rbx, qword ptr [rdx + rcx + 64244]
and rcx, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rcx]
cmp bl, -115
cmovnp cx, ax
lea di, qword ptr [rsi]
sub dil, 51
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 83
btr edx, 101
add bl, -3 # instrumentation
setnp al
and rdx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdx], dl
btc rdi, 152
lea rbx, qword ptr [rdi + rcx + 50211]
and rbx, 0b1111111111111 # instrumentation
movsx edx, word ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdi], ecx
and rdi, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rdi], rsi
or dx, 1 # instrumentation
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
bt dx, 28
and rbx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rbx]
adc eax, -2004720141
and esi, ecx
and rsi, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rax], cl
and rdi, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdi], 39
cmp dl, cl
or si, 1 # instrumentation
and dx, si # instrumentation
shr dx, 1 # instrumentation
div si
add dl, -127 # instrumentation
setnp cl
and rdx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdx], -13002
and rcx, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rcx]
or ecx, ebx
bts esi, 240
adc al, -63
and rax, 0b1111111111111 # instrumentation
cmovp bx, word ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], di
setnb al
and rcx, 0b1111111111111 # instrumentation
or si, -104
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr rsi, qword ptr [r14 + rbx]
add di, 76
bts ax, di
and rax, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rax], rbx
and rsi, 0b1111111111111 # instrumentation
sub word ptr [r14 + rsi], 100
and rsi, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rsi], rax
movzx rcx, al
cmovnl cx, di
and rsi, 0b1111111111000 # instrumentation
and rdi, 0b111 # instrumentation
lock btr qword ptr [r14 + rsi], rdi
movsx esi, bx
add eax, -118985343
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], -125
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
