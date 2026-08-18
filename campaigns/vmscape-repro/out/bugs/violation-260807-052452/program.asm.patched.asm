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
add al, 59 # instrumentation
lea rdx, qword ptr [rdi + rsi + 3782]
and rax, 0b1111111111111 # instrumentation
mov word ptr [r14 + rax], -30986
and rdx, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rax], -81
add ecx, 63
nop eax
setp dl
adc dl, bl
cmp rax, -21
mov di, ax
and rdi, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdx], 118
cmovo ax, ax
cmovnp eax, ecx
mul dl
and rdi, 0b1111111111111 # instrumentation
add cx, word ptr [r14 + rdi]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add al, 55 # instrumentation
and rsi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rsi], -104
and rax, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rax], cx
and rcx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
add bl, byte ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rdx], bl
and rdx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdx], 99
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
lea cx, qword ptr [rsi + rsi]
lea edx, qword ptr [rsi]
and rax, 0b1111111111111 # instrumentation
sub ecx, dword ptr [r14 + rax]
mul bl
and rsi, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rsi], dil
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsr ecx, dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
btr word ptr [r14 + rbx], cx
and rdi, 0b1111111111111 # instrumentation
xor ax, word ptr [r14 + rdi]
or ax, 1 # instrumentation
and dx, ax # instrumentation
shr dx, 1 # instrumentation
div ax
and rbx, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rbx], 1
and rax, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rax], -51
and rcx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rcx], 122
and rcx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rcx], 1
and rcx, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rcx], 5
or cx, 0b1000000000000000 # instrumentation
bsr di, cx
xor dl, dl
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], -9
and rdi, 0b1111111111111 # instrumentation
xor rcx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rbx], 2
lea edx, qword ptr [rdx]
and rcx, 0b1111111111111 # instrumentation
add al, byte ptr [r14 + rcx]
dec rbx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax
and rsi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rsi], 3
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
add bl, -99 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
movsx rsi, si
adc cx, dx
lea rdi, qword ptr [rdx + rax + 21410]
btr eax, 1
or al, 94
bswap edx
lea ax, qword ptr [rcx + rdi]
and rax, 0b1111111111111 # instrumentation
or cl, byte ptr [r14 + rax]
cmovnb rdx, rbx
lea bx, qword ptr [rcx + rbx + 28250]
lea ecx, qword ptr [rbx + rcx]
add dl, -87
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], al
bt ecx, 76
cmp al, -75
movzx ebx, ax
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, 82 # instrumentation
sbb bx, si
inc cl
sub eax, 81
and rsi, 0b1111111111111 # instrumentation
adc rax, qword ptr [r14 + rsi]
cmovp cx, dx
add al, dl
or eax, -103136922
and rdx, 0b1111111111111 # instrumentation
sub ebx, dword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
and rcx, 0b111 # instrumentation
btr qword ptr [r14 + rdi], rcx
sbb rsi, rcx
adc al, -121
cld
inc ax
and rdi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdi], 27
xadd bl, cl
lea rcx, qword ptr [rax]
sbb ax, -60
and rsi, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rsi]
setp cl
and rbx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
imul cx, word ptr [r14 + rdi]
adc rdx, rsi
add dl, cl
and rax, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rax], 119
lea esi, qword ptr [rsi + rdi]
setnz cl
test al, -110
and rax, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rax], -52
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], dil
bswap edx
and rcx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
and ecx, dword ptr [r14 + rcx]
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
