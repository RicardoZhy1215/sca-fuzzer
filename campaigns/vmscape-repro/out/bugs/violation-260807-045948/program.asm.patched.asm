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
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 0b1000 # instrumentation
and byte ptr [r14 + rdx], 0b11111000 # instrumentation
cmp rax, 126
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], di
not eax
or ax, 30629
btc eax, ebx
movsx esi, dx
xor esi, edx
and al, -8
sub rax, -85
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx
and rsi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rsi], -31
or bl, 17
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000000000000000 # instrumentation
bsr bx, word ptr [r14 + rdi]
cmp al, -8
add al, 38
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, -109 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rdi]
cmovo si, dx
bts si, cx
sbb bx, -53
mov di, cx
seto dl
dec ecx
and rdx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdx], -49
and rax, 0b1111111111111 # instrumentation
and rdx, 0b111 # instrumentation
bt qword ptr [r14 + rax], rdx
adc rdx, 101
lea bx, qword ptr [rbx + rsi]
and rax, 0b1111111111111 # instrumentation
add word ptr [r14 + rax], -72
setp dil
and rdx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdx]
add si, bx
and rbx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rbx], dil
mov dx, 1310
nop rsi
and rdx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rdx]
add al, -107 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovnz rax, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdx], rax
xor esi, 108
sub dl, bl
and dl, al
and rax, 0b1111111111111 # instrumentation
cmovnb rax, qword ptr [r14 + rax]
xor dl, 122
and rdx, 0b1111111111111 # instrumentation
cmovnp rcx, qword ptr [r14 + rdx]
btc rdi, 24
and rsi, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rsi], 10
cmovl edi, edx
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
xor sil, dl
xor dl, 78
cmovnbe ax, si
and rax, 833126011
bts si, cx
and rax, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rax]
lea eax, qword ptr [rdx]
and bl, cl
sub bx, 20
and rdi, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdi], al
sbb bl, -55
setl sil
cmpxchg dil, al
and rax, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rax]
test al, -87
and rdx, 0b1111111111111 # instrumentation
sbb rsi, qword ptr [r14 + rdx]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
movsx esi, byte ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmp eax, dword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmovnz eax, dword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rsi]
sbb ax, 24777
and rax, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rax], eax
add dl, al
and rdx, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rdx], 3
not al
mul rax
and rdx, 0b1111111111111 # instrumentation
or dx, word ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi]
or rdi, 0b1000000000000000000000000000000 # instrumentation
bsr rdx, rdi
add cl, -11 # instrumentation
lea bx, qword ptr [rax]
and rsi, 0b1111111111111 # instrumentation
cmovbe esi, dword ptr [r14 + rsi]
and ax, 17892
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rbx], dx
and rax, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rax], dil
lea di, qword ptr [rsi]
and rbx, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rbx], 5
and rax, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rax], -36
sub sil, -50
lea rax, qword ptr [rsi + rax]
and rdi, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rdi], -47
and rdx, 0b1111111111111 # instrumentation
adc rbx, qword ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
movsx edi, byte ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovo rdi, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rcx]
add al, -18
and rax, 0b1111111111111 # instrumentation
and rax, 0b111 # instrumentation
btc qword ptr [r14 + rax], rax
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
