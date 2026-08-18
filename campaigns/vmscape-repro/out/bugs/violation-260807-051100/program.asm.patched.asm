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
add al, -88 # instrumentation
setno dl
cmovz rbx, rbx
add edi, edi
adc bl, -45
xor ebx, esi
lea edi, qword ptr [rsi + rax]
and rbx, 0b1111111111111 # instrumentation
cmovnb edi, dword ptr [r14 + rbx]
cmovbe esi, eax
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf si, word ptr [r14 + rsi]
add cl, -114 # instrumentation
lea rsi, qword ptr [rcx]
setp bl
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf cx, word ptr [r14 + rsi]
add al, -118 # instrumentation
setp cl
cmp rax, -85
cmovns rax, rdi
and rax, 0b1111111111111 # instrumentation
cmovb cx, word ptr [r14 + rax]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rsi, 0b1111111111111 # instrumentation
test word ptr [r14 + rsi], cx
clc
cmovo dx, dx
and rax, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rax], 2
or bl, bl
and rdx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdx]
xadd edi, ebx
xchg bl, dl
or dl, cl
and rbx, 0b1111111111111 # instrumentation
neg word ptr [r14 + rbx]
cmovl rax, rdi
and rax, 0b1111111111111 # instrumentation
cmp cl, byte ptr [r14 + rax]
mul dl
cmp bx, bx
and rbx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovnb si, word ptr [r14 + rax]
sub eax, 100072811
and rbx, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rbx]
setns bl
and cl, al
or rcx, 0b1000000000000000000000000000000 # instrumentation
bsf rbx, rcx
xor dil, cl
and rsi, 0b1111111111111 # instrumentation
or eax, dword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], -41
and rsi, 0b1111111111111 # instrumentation
not dword ptr [r14 + rsi]
not rax
and rdx, 0b1111111111111 # instrumentation
btc word ptr [r14 + rdx], 2
and rax, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rax]
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl
and rsi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rsi], dl
lea dx, qword ptr [rsi + rdx + 34241]
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
add dl, 77 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rbx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rbx]
cmovb rcx, rdx
mov dl, al
movzx dx, bl
add bl, bl
and bl, 101
and rcx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rcx], al
setnp bl
or bl, 0b1000 # instrumentation
and bl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv bl
and rax, 0b1111111111111 # instrumentation
mov ecx, dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rbx], 6
and di, si
test al, 120
xor rdx, 61
cmovnle ax, dx
cmovnl eax, esi
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea esi, qword ptr [rcx]
movzx ecx, cl
and rsi, 0b1111111111111 # instrumentation
neg qword ptr [r14 + rsi]
lea ebx, qword ptr [rcx]
and rsi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rsi], 41
xor al, -104
add rdx, rax
setbe al
and rdi, 0b1111111111111 # instrumentation
add ax, word ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmovns dx, word ptr [r14 + rdx]
and al, 45
and rdi, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rdi]
xchg ax, ax
adc cx, cx
and rax, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rcx]
add bl, -86 # instrumentation
setle cl
lea dx, qword ptr [rsi + rbx]
std
and rbx, 0b1111111111111 # instrumentation
cmovb rax, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
and eax, 0b111 # instrumentation
bt dword ptr [r14 + rdi], eax
and rax, 0b1111111111111 # instrumentation
cmovnbe eax, dword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], rcx
and rcx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rcx], cl
and rax, 0b1111111111111 # instrumentation
cmovo rdi, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovnbe edx, dword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rdx]
movzx ax, dl
or dl, -111
and rcx, 0b1111111111111 # instrumentation
movsx rdx, word ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
and rdx, 0b111 # instrumentation
lock btc qword ptr [r14 + rax], rdx
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx
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
