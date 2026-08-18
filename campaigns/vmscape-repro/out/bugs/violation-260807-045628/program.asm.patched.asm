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
mov bl, bl
sub rax, 1047909650
add bl, 72 # instrumentation
lea rax, qword ptr [rbx + rbx]
sets sil
cwd
and rdi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdi]
cmovle rcx, rax
btc rsi, 44
or rax, -93
sub ebx, 42
and rdx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdx], rcx
and rsi, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rsi]
setnl dil
and rbx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rbx], al
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, 45 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovno cx, word ptr [r14 + rsi]
imul rbx, rcx, -128
and rbx, 0b1111111111111 # instrumentation
imul rax, qword ptr [r14 + rbx]
cmp dx, bx
and rdx, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rdx]
cmovnbe rcx, rbx
not dl
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 50
sbb bl, dl
adc dil, 4
and rcx, 0b1111111111111 # instrumentation
movzx ax, byte ptr [r14 + rcx]
not rcx
xor al, -27
and rsi, 0b1111111111111 # instrumentation
sbb rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rbx]
movzx esi, al
xchg ebx, eax
sub bx, ax
and rax, 0b1111111111111 # instrumentation
bt word ptr [r14 + rax], 5
and rsi, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rsi]
cmp sil, -36
and rsi, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rsi]
lea ebx, qword ptr [rdx + rdx + 22251]
and rbx, 0b1111111111111 # instrumentation
btr word ptr [r14 + rbx], 1
lea esi, qword ptr [rdx]
and rax, 0b1111111111111 # instrumentation
mov ebx, dword ptr [r14 + rax]
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl
add dl, -61 # instrumentation
setp bl
and rbx, 0b1111111111111 # instrumentation
or edx, dword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
add cl, byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmovnbe rax, qword ptr [r14 + rcx]
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
add cl, -108 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
sbb eax, -264195205
cbw
and rax, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rax], dl
lahf
lea eax, qword ptr [rsi]
btr rdi, rcx
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl
and bl, bl
add rax, 650064320
mul bl
add al, -73 # instrumentation
cmovnp eax, ecx
setnl dil
cmp rsi, rsi
and rcx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rcx], edx
sub bl, -102
cmovnz ecx, ebx
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, 61 # instrumentation
and rcx, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rcx]
sbb cl, -28
xor bl, bl
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx]
cmovs si, cx
cmp eax, eax
bts ebx, 200
add bl, -104 # instrumentation
cmovnl rdi, rax
and rbx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rbx], 22
and rdx, 0b1111111111111 # instrumentation
cmovle ebx, dword ptr [r14 + rdx]
or cl, cl
cmovnz rsi, rdx
and rbx, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rbx], 4
add dl, -48 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovnb edi, dword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
movsx bx, byte ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovnl si, word ptr [r14 + rdx]
sbb al, dil
adc di, cx
mul si
and rdx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rdx], dx
sbb rdx, rax
cmp al, bl
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], cl
and rdi, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdi]
sub rcx, 102
lea ecx, qword ptr [rdx + rdx]
xadd dl, bl
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rbx], bl
and rax, 0b1111111111111 # instrumentation
cmovnle ebx, dword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], -29
bts rsi, 252
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
