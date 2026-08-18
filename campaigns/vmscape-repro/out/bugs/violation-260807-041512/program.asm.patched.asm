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
and rsi, -36
sub rax, -1926913819
and eax, -1699563500
movsx rcx, dil
xor rdi, rbx
and rcx, 0b1111111111111 # instrumentation
and ebx, dword ptr [r14 + rcx]
or al, al
nop cx
setnle cl
and rdi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdi], al
bt bx, si
add bl, 81 # instrumentation
cmovle rbx, rbx
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsr rbx, qword ptr [r14 + rax]
imul esi
mul bl
and rbx, 0b1111111111111 # instrumentation
and cl, byte ptr [r14 + rbx]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
test ax, -16479
and rbx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rbx]
test dl, sil
movsx rax, al
xor sil, 57
adc al, al
setnbe dl
and rdx, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rdx]
movsx esi, sil
and rdx, 0b1111111111111 # instrumentation
add ax, word ptr [r14 + rdx]
nop eax
and rbx, 0b1111111111111 # instrumentation
xor si, word ptr [r14 + rbx]
cmovz cx, cx
lea rdi, qword ptr [rdi + rbx + 36071]
and rcx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rcx]
setnp al
and rdx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdx], sil
and rax, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rax]
lea rdx, qword ptr [rcx + rdx + 27973]
lea rcx, qword ptr [rcx]
cmpxchg bl, al
dec dx
cwd
cmovle cx, dx
and rsi, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdi], edx
and rax, 0b1111111111000 # instrumentation
and rdi, 0b111 # instrumentation
lock btc qword ptr [r14 + rax], rdi
and rax, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rax]
cmovnz rcx, rax
and rcx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rcx], 12
cmovnbe rax, rdx
lea rcx, qword ptr [rdx]
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
btc rdx, rbx
lea esi, qword ptr [rcx]
cmp al, -47
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx
and rax, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rax], 53
and rdi, 0b1111111111111 # instrumentation
sbb cl, byte ptr [r14 + rdi]
seto dl
add ecx, 109
and rdi, 0b1111111111111 # instrumentation
and rcx, 0b111 # instrumentation
btc qword ptr [r14 + rdi], rcx
xchg rcx, rax
sub eax, -1624875801
cmovnp ecx, edx
add al, -111
test edi, -462503075
bt di, 213
adc bl, 96
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
btc word ptr [r14 + rbx], 0
add dl, 122 # instrumentation
and rsi, 0b1111111111111 # instrumentation
not qword ptr [r14 + rsi]
setp cl
lea eax, qword ptr [rdx + rax]
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], dl
and rcx, 0b1111111111111 # instrumentation
cmovnz si, word ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rbx]
mov cl, -122
cmpxchg dl, bl
add bl, -84
and rbx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rbx], bl
xadd rax, rsi
not bl
lea dx, qword ptr [rsi + rcx]
and rax, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rax]
setb bl
and rcx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rcx], -116
setnle bl
test cl, -104
movsx edi, cl
and rdx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rdx], -50
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 0
and rdi, 0b1111111111111 # instrumentation
movsx ax, byte ptr [r14 + rdi]
xor dl, cl
sub cl, -76
add bl, 105 # instrumentation
setbe bl
and rdi, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rbx]
cmovp edx, ebx
and rax, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rax], dl
setnz al
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
