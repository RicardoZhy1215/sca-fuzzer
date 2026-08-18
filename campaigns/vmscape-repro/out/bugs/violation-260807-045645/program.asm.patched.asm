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
add bl, -63 # instrumentation
cmovns rbx, rdi
and rdi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rdi]
lea bx, qword ptr [rax]
or al, 1 # instrumentation
mov ax, 1 # instrumentation
div al
and rbx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rbx], edi
and rdx, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rdx], al
dec dil
and rbx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rbx]
seto bl
mov al, -6
and rbx, 0b1111111111111 # instrumentation
cmovno rcx, qword ptr [r14 + rbx]
bt cx, dx
movzx edi, dl
test dx, 16406
and rsi, 0b1111111111111 # instrumentation
movzx di, byte ptr [r14 + rsi]
sbb di, -94
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, 84 # instrumentation
bswap rdx
and rbx, 0b1111111111111 # instrumentation
cmovnz ax, word ptr [r14 + rbx]
xchg cx, ax
lea si, qword ptr [rbx + rsi + 3228]
and rdi, 0b1111111111111 # instrumentation
movzx di, byte ptr [r14 + rdi]
add sil, 41
neg dil
and rdx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rdx], si
and rdx, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rdx], al
sbb eax, edi
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi
and rbx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rbx], cl
and rax, 0b1111111111111 # instrumentation
adc rcx, qword ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
cmovnb rbx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
bts word ptr [r14 + rcx], 5
sub dx, -89
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], ebx
and rbx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rbx], -53
lea bx, qword ptr [rdx + rax + 47233]
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], -117
sub rbx, 14
xor sil, sil
and rax, -1497972336
and rcx, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
btc dword ptr [r14 + rcx], edi
adc al, cl
and rdx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdx], bl
sub al, 93
and rcx, 0b1111111111111 # instrumentation
or dl, byte ptr [r14 + rcx]
mul dx
and rax, 0b1111111111111 # instrumentation
sbb di, word ptr [r14 + rax]
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
bt ecx, 41
movsx rdi, sil
test rdi, rdx
and al, -74
and rbx, 0b1111111111111 # instrumentation
cmovns di, word ptr [r14 + rbx]
imul eax, edi, 70
add al, -112 # instrumentation
cmovnl rdx, rsi
bt rdx, rsi
cmp eax, -516361046
adc al, -99
adc cx, cx
lea rbx, qword ptr [rsi]
lea di, qword ptr [rdx + rax + 1383]
xor rcx, rax
cmovnbe di, ax
and bl, -9
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add al, -123 # instrumentation
and rcx, 0b1111111111111 # instrumentation
movzx ax, byte ptr [r14 + rcx]
lea rdx, qword ptr [rcx + rdx]
sbb rax, -2121605163
and rax, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rax], 125
and rdi, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rdi]
add al, -22 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rdi]
bt edx, 108
mov dl, dl
and sil, 126
and rcx, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rcx]
clc
neg rbx
and rdi, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 0b1000 # instrumentation
and byte ptr [r14 + rbx], 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv byte ptr [r14 + rbx]
cmp al, -75
and rcx, 0b1111111111111 # instrumentation
cmovl edi, dword ptr [r14 + rcx]
cmp al, bl
lea cx, qword ptr [rdi + rdi + 3588]
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], eax
lea bx, qword ptr [rdx + rdx]
and rdx, 0b1111111111111 # instrumentation
mov ebx, dword ptr [r14 + rdx]
cmovnp bx, bx
sbb dl, 35
and rcx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
adc edx, dword ptr [r14 + rcx]
bt rsi, 66
and rsi, 0b1111111111111 # instrumentation
xor dl, byte ptr [r14 + rsi]
setnp bl
and rsi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rsi]
xadd edx, eax
and rcx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rcx]
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
