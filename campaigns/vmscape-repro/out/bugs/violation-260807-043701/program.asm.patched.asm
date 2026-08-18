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
and rdi, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rdi], ecx
cmovns edx, ebx
sub dl, -101
movsx ebx, al
and rcx, 0b1111111111111 # instrumentation
and ecx, 0b111 # instrumentation
bt dword ptr [r14 + rcx], ecx
add ax, 31694
cmp cl, 61
and cl, al
sbb al, al
inc cl
cmovnp esi, ebx
imul rbx, rdi, -84
add dl, -20 # instrumentation
setl bl
cld
and rsi, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rsi], rbx
lea rbx, qword ptr [rax]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, 82 # instrumentation
mov ecx, -1485314662
and rax, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
sub dx, word ptr [r14 + rcx]
lea rdx, qword ptr [rcx]
and rcx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rcx], 84
and rdx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdx], bl
and rbx, 0b1111111111111 # instrumentation
and edx, dword ptr [r14 + rbx]
imul rbx, rdx, -69
and rdx, 0b1111111111111 # instrumentation
or eax, dword ptr [r14 + rdx]
xor cl, dl
sub edx, 68
btc rcx, 163
add cl, 120 # instrumentation
setnp al
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi]
or bl, -114
and rsi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rsi], ecx
and rdx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rdx], ebx
and rbx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rbx], cl
setz bl
and rcx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rcx], cl
and rdi, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsf rdi, qword ptr [r14 + rbx]
lea edi, qword ptr [rcx + rbx + 22968]
lea ebx, qword ptr [rsi + rsi + 40229]
cmpxchg dl, cl
imul rsi
and ecx, 11
and rax, 0b1111111111111 # instrumentation
cmovp rcx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rbx], 3
and rdi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdi], bl
and rdi, 0b1111111111111 # instrumentation
cmovs ecx, dword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi]
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
add al, -49 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
mov bl, cl
and rbx, 0b1111111111111 # instrumentation
sbb cx, word ptr [r14 + rbx]
xor ax, 16643
and rcx, 0b1111111111111 # instrumentation
cmovl edx, dword ptr [r14 + rcx]
sbb al, 125
and rcx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rcx]
xor di, cx
xchg al, dl
cmovnp rdi, rdi
nop si
or rcx, 0b1000000000000000000000000000000 # instrumentation
bsr rax, rcx
add dl, -114 # instrumentation
cmovnle rbx, rdi
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], 23
setnbe dl
and rcx, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rcx]
xchg edi, eax
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add dl, 30 # instrumentation
and rsi, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rsi]
adc al, 72
and rdx, 0b1111111111111 # instrumentation
and rcx, qword ptr [r14 + rdx]
xadd rbx, rbx
or al, 16
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi
add bl, -67
and rax, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rax], bl
seto cl
and rsi, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rsi]
sbb di, cx
sub dx, dx
test eax, 565185842
and rcx, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rcx], 2
and rsi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rsi], dl
and rcx, 0b1111111111111 # instrumentation
and bl, byte ptr [r14 + rcx]
sub esi, 6
sub al, -74
and rdi, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdi]
or ax, bx
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], 48
and rsi, 0b1111111111111 # instrumentation
cmovnbe rcx, qword ptr [r14 + rsi]
lea ebx, qword ptr [rcx]
lea cx, qword ptr [rax]
and rdi, 0b1111111111111 # instrumentation
and dword ptr [r14 + rdi], edx
mov ebx, eax
and rdi, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rdi], -62
and rcx, 0b1111111111111 # instrumentation
cmovs ecx, dword ptr [r14 + rcx]
adc eax, -583831156
or bx, 0b1000000000000000 # instrumentation
bsr bx, bx
sub sil, 102
and rbx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rbx], dl
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
