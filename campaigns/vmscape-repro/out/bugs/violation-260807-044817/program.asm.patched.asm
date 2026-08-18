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
add dl, -97 # instrumentation
lea edi, qword ptr [rdi + rsi]
cmovz esi, esi
cmp dl, 108
test eax, 130611413
sub dl, dl
and rsi, 0b1111111111111 # instrumentation
xor dl, byte ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rdi], -35
setl cl
and rcx, 0b1111111111111 # instrumentation
cmovbe rsi, qword ptr [r14 + rcx]
adc al, bl
or ecx, 0b1000000000000000000000000000000 # instrumentation
bsf ebx, ecx
add dl, -73 # instrumentation
setle cl
setno dl
and ax, 17098
and rdi, 0b1111111111111 # instrumentation
not word ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
adc rax, qword ptr [r14 + rax]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, -85 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnbe edx, dword ptr [r14 + rdx]
imul cl
btc rdx, 25
add cl, -11 # instrumentation
not bl
and rdx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdx]
setno al
cmpxchg al, dl
and rdi, 0b1111111111111 # instrumentation
cmovnbe edx, dword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
test word ptr [r14 + rdx], cx
or bl, al
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rcx
and rdi, 0b1111111111111 # instrumentation
cmp eax, dword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rcx], dx
sbb al, -72
test eax, -344087725
and rsi, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
imul bx, word ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovb eax, dword ptr [r14 + rbx]
adc rdx, rax
or rax, -1974357316
xor al, bl
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
and ax, word ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rcx]
add bl, -115
mov dl, 69
cmovnl cx, bx
and rdx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rdx]
sub cx, dx
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], rcx
add bl, dl
lea rdx, qword ptr [rbx]
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
add dl, 14 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rbx, 0b1111111111111 # instrumentation
cmovb si, word ptr [r14 + rbx]
mov eax, -572252305
sbb bx, 81
xchg di, ax
and ax, -26601
and rdi, 0b1111111111111 # instrumentation
cmovl si, word ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
cmovnbe ecx, ebx
cmovnl edi, edx
xadd bl, al
cmovns ax, di
or cx, si
setnz cl
add bx, 120
lea ebx, qword ptr [rbx + rax + 64795]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rsi
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add dl, -82 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovo di, word ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
sub cx, word ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
xchg word ptr [r14 + rdx], cx
test rdi, -1570712924
lea rdi, qword ptr [rcx + rdx]
bt ecx, edi
and rdx, 0b1111111111111 # instrumentation
mov dx, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rax], sil
xor dl, bl
cmpxchg rdx, rcx
and rcx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rcx], 39
and rax, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rax], -127
and rcx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rcx], ebx
and rsi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rsi], cl
and rdx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdx], ax
cmp cl, 90
add rax, rdx
and rax, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rax], 5
and rcx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rcx], 1
setbe dl
cmovs di, bx
xor dl, al
and rsi, 0b1111111111111 # instrumentation
cmp edx, dword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], 116
mul si
sub ax, si
and rsi, 0b1111111111111 # instrumentation
movzx esi, byte ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
movzx edx, al
and rax, 0b1111111111111 # instrumentation
cmovb dx, word ptr [r14 + rax]
cmp ebx, 47
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
