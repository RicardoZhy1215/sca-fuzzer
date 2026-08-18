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
cmp esi, esi
xor eax, edi
lea bx, qword ptr [rdx]
and rax, 0b1111111111111 # instrumentation
cmovnbe rcx, qword ptr [r14 + rax]
cbw
or dil, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv dil
neg rdi
cdq
add al, 20
lea edx, qword ptr [rdx + rbx + 26674]
imul rbx, rax, -88
lea rbx, qword ptr [rcx + rdx]
and rbx, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rbx]
add dl, al
or cl, cl
nop rcx
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rdx], di
cmovnl rbx, rsi
xor al, cl
and rax, 0b1111111111111 # instrumentation
cmovz rbx, qword ptr [r14 + rax]
lea edi, qword ptr [rdx + rcx + 38103]
and rbx, 0b1111111111111 # instrumentation
cmovnz bx, word ptr [r14 + rbx]
or rdi, 0b1000000000000000000000000000000 # instrumentation
bsf rax, rdi
cmp cl, bl
and rdx, 0b1111111111111 # instrumentation
add edx, dword ptr [r14 + rdx]
and rdi, 0b1111111111000 # instrumentation
lock btc qword ptr [r14 + rdi], 6
add dl, 24 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovnb eax, dword ptr [r14 + rbx]
lea edi, qword ptr [rcx + rsi + 36239]
and rdx, 0b1111111111111 # instrumentation
cmovo edx, dword ptr [r14 + rdx]
lea si, qword ptr [rsi + rdi + 35513]
add dil, 40
xadd eax, edi
and rdi, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rdi], 0
and rbx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rbx]
or ax, 19552
adc ecx, edi
and rcx, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 1 # instrumentation
and dx, word ptr [r14 + rdi] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rdi]
bts di, ax
and rsi, 0b1111111111111 # instrumentation
test word ptr [r14 + rsi], 2212
and rdi, 0b1111111111111 # instrumentation
and edx, 0b111 # instrumentation
btc dword ptr [r14 + rdi], edx
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl
xchg sil, bl
cmp al, al
cmovnb dx, dx
lea cx, qword ptr [rcx]
and rbx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rbx], ebx
or edi, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv edi
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
and rcx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rcx], al
cmovz edx, edx
cmovnl esi, ecx
test cl, 65
setnp cl
and rbx, 0b1111111111111 # instrumentation
or cl, byte ptr [r14 + rbx]
xor cl, dl
cmp rax, -1952074611
bswap edx
cmp al, -100
and rax, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
btc word ptr [r14 + rax], bx
and rdx, 0b1111111111000 # instrumentation
and si, 0b111 # instrumentation
lock btc word ptr [r14 + rdx], si
test al, -62
lea ebx, qword ptr [rsi + rbx + 62754]
or dl, dl
setb al
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rsi, 0b1111111111111 # instrumentation
movsx rcx, word ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rax], dil
bswap rdi
and rax, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rax], 1458125964
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx
setnz al
and rdi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rdi]
cmp cl, cl
setp al
and rbx, 0b1111111111111 # instrumentation
cmovnle bx, word ptr [r14 + rbx]
lea eax, qword ptr [rsi]
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
setnb bl
mov dl, dl
and rcx, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rcx], esi
cwde
cmpxchg rcx, rdx
mov rax, rsi
mov rdx, 1783842677074233607
and bl, 123
sub ax, 24452
test cl, -119
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rcx], dx
adc al, cl
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], al
cmovl rcx, rdi
and rbx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rbx], di
lea edx, qword ptr [rsi]
and rdx, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 1 # instrumentation
and rcx, 0b1111111111000 # instrumentation
and rbx, 0b111 # instrumentation
lock bts qword ptr [r14 + rcx], rbx
sub rcx, -18
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
