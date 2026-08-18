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
mul rbx
add al, -69
movsx esi, bx
and rcx, 0b1111111111111 # instrumentation
cmovp ebx, dword ptr [r14 + rcx]
movzx eax, bl
lea rax, qword ptr [rax + rsi + 60408]
imul eax
xchg al, cl
cmovb di, si
or edx, 0b1000000000000000000000000000000 # instrumentation
bsr edx, edx
or cl, bl
and rdx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rdx], 0
movsx rsi, bl
cmovz rax, rbx
xchg rdx, rsi
imul rcx, rsi, -43
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], rbx
and rdx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdx]
setl sil
and rdi, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rdi], 0
add cl, 82 # instrumentation
and rax, 0b1111111111000 # instrumentation
xchg word ptr [r14 + rax], di
and rax, 0b1111111111111 # instrumentation
cmovp bx, word ptr [r14 + rax]
imul rdx, rax
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rcx], rdx
and rax, 0b1111111111111 # instrumentation
cmovnb di, word ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
and eax, 837521129
and eax, -1152034946
setnb sil
and dl, bl
and rax, 0b1111111111111 # instrumentation
cmovz rax, qword ptr [r14 + rax]
cmovnb rbx, rdi
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], bl
mov dl, al
xchg cx, ax
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], ebx
and rdx, 0b1111111111111 # instrumentation
mul dword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rsi]
bts cx, si
and rcx, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rcx], ecx
and rcx, 0b1111111111111 # instrumentation
cmovs ax, word ptr [r14 + rcx]
and rax, rdx
cmovb rdx, rdx
cmovs esi, ebx
and rax, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmovnle ecx, dword ptr [r14 + rsi]
and rsi, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rsi], -124
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
and rbx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rbx], cl
sbb dl, dl
cmovl edi, esi
dec ecx
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi]
add al, -40 # instrumentation
setnb bl
imul rsi, rsi, -93
and rax, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rax], 127
cmovnp esi, esi
lea ecx, qword ptr [rax + rax + 59283]
not bl
test rax, -1616669156
or al, -89
btr rbx, 197
lea rdi, qword ptr [rdx]
and rax, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rax], 53
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add dl, -90 # instrumentation
lea rdi, qword ptr [rdi + rdi + 30217]
setb al
and rbx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rbx], 121
and rsi, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rsi], -32
adc dx, -2
cmovnl dx, bx
and rdi, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rdi], -83
and rsi, 0b1111111111111 # instrumentation
mov cl, byte ptr [r14 + rsi]
lea edx, qword ptr [rcx + rcx + 36260]
test edx, 999302291
cmovbe rcx, rdi
lea cx, qword ptr [rdi]
and rdx, 0b1111111111111 # instrumentation
cmovbe edi, dword ptr [r14 + rdx]
or edx, esi
lea rcx, qword ptr [rcx]
and rcx, 0b1111111111111 # instrumentation
and di, word ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rsi], 50
or bl, al
and rdx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdx], rax
add eax, -1249154414
xchg cl, al
and rbx, 0b1111111111111 # instrumentation
cmovb bx, word ptr [r14 + rbx]
add rsi, rdx
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], -52
cmovle rsi, rdi
add ax, dx
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], 115
cmovz ax, si
lea bx, qword ptr [rsi + rdx]
xor al, 55
stc
sub bl, dl
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
