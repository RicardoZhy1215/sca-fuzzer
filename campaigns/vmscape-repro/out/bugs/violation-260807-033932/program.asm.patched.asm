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
sub bl, al
or rax, -1319722939
add al, 22
or ax, -17821
sbb rbx, rsi
mul bl
cmp dl, -66
imul ax
and rdx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsr rdx, qword ptr [r14 + rdx]
btc si, bx
and ax, 28043
setp dil
mul cl
add bl, -11 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov ebx, dword ptr [r14 + rbx]
not dl
cmovz edx, edx
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rax], bl
test al, 115
btr si, 62
add al, -34 # instrumentation
setnl al
btr ecx, esi
and rax, 0b1111111111111 # instrumentation
cmovnbe cx, word ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rcx], dx
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], rsi
and rax, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rax], 53
and rsi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rsi], 104
add cl, dl
lea di, qword ptr [rdx]
cmovnl dx, si
and rcx, 0b1111111111000 # instrumentation
and rcx, 0b111 # instrumentation
lock btr qword ptr [r14 + rcx], rcx
cmp al, 15
lea si, qword ptr [rax + rsi + 35230]
cmovnp si, si
and rax, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rax]
add cl, 54 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovbe ecx, dword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rsi]
xchg eax, eax
and rax, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], -54
xor al, dil
test esi, edx
adc rax, 678504208
setz dl
test ebx, edi
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rsi], bx
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr rax, qword ptr [r14 + rbx]
add bl, 53 # instrumentation
cmovo edi, ebx
movsx ax, al
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
btr ebx, esi
add al, 45 # instrumentation
cmovp di, di
cmovnle edi, edi
and rcx, 0b1111111111111 # instrumentation
and ebx, dword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovle ecx, dword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovnl eax, dword ptr [r14 + rax]
cmpxchg bl, dl
or al, 1 # instrumentation
mov ax, 1 # instrumentation
div al
add bl, 94 # instrumentation
setbe dil
test si, -23451
movzx ax, dil
sets dil
and rbx, 0b1111111111111 # instrumentation
sub word ptr [r14 + rbx], 2
and rcx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rcx]
cmovnle bx, ax
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 7
and rsi, 0b1111111111111 # instrumentation
or edx, dword ptr [r14 + rsi]
or eax, -2042143865
sbb rax, rbx
sub rcx, rsi
and rbx, 0b1111111111111 # instrumentation
sub word ptr [r14 + rbx], bx
not edi
lea ecx, qword ptr [rdi + rdi + 9906]
and rdx, 0b1111111111111 # instrumentation
cmovle eax, dword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsr rdi, qword ptr [r14 + rsi]
lea rdi, qword ptr [rax + rdi]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rbx
and rdx, 0b1111111111111 # instrumentation
xor rdi, qword ptr [r14 + rdx]
imul dil
sbb dl, dl
and rdx, 0b1111111111111 # instrumentation
and esi, dword ptr [r14 + rdx]
add edi, -75
cmovnb ebx, edx
cmovle si, dx
and rsi, 0b1111111111111 # instrumentation
cmovz bx, word ptr [r14 + rsi]
btc rdi, 15
and rdi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdi], dl
or ax, 19081
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr esi, dword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rcx], bl
or cl, dl
and rax, 0b1111111111111 # instrumentation
cmovb edx, dword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmovb rax, qword ptr [r14 + rcx]
cmovo rcx, rax
and rdx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdx], -31
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000 # instrumentation
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rcx]
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
