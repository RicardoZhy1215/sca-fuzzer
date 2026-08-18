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
or bl, -18
lea edi, qword ptr [rbx + rdi + 14023]
and rbx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rbx], al
mul edi
and rdi, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
bts word ptr [r14 + rdi], di
sbb al, al
test ax, -17300
cmovnle rsi, rcx
sbb eax, 356071096
lea bx, qword ptr [rbx + rax + 6790]
cmovnp ecx, ebx
btc rax, rdx
or cl, cl
cmovp si, cx
and rdx, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rdx]
sahf
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, 44 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovz rcx, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rcx], cl
sbb dl, cl
and rax, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rax], sil
cmovle dx, bx
not si
and rcx, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
btr word ptr [r14 + rcx], dx
mov dl, -38
and rcx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rcx]
or bx, 0b1000000000000000 # instrumentation
bsf di, bx
and rdi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdi], dl
and rcx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rcx], di
and rbx, 0b1111111111111 # instrumentation
sbb bl, byte ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
cmovnle edi, dword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rcx
and rax, 0b1111111111111 # instrumentation
cmp ebx, dword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rdi], edi
xchg rsi, rax
add cl, al
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx
lea dx, qword ptr [rax + rdx]
imul si
xor al, 99
and rcx, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rcx]
cmovnz rcx, rax
and rcx, 0b1111111111111 # instrumentation
and al, byte ptr [r14 + rcx]
std
and rsi, 0b1111111111111 # instrumentation
cmovs esi, dword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mov word ptr [r14 + rax], di
and rbx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rbx], esi
and rbx, 0b1111111111111 # instrumentation
cmovs bx, word ptr [r14 + rbx]
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
test bl, -25
sub rax, -563859985
or ax, 83
or ebx, eax
sahf
xor al, 105
lea rdi, qword ptr [rsi]
lea ebx, qword ptr [rax + rdi + 52560]
sub cl, 104
and rax, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rax]
cmovnp cx, bx
imul cx
sbb sil, -48
and dl, -76
and rsi, 0b1111111111111 # instrumentation
add dil, byte ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdx], 7
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, 82 # instrumentation
movsx dx, dil
and rcx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rax]
setnbe bl
and rcx, 0b1111111111111 # instrumentation
cmovnle ax, word ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovnl dx, word ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rbx], dil
add dl, -71
cmovno rdx, rsi
and rcx, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rbx]
xadd rax, rbx
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], 87
sub rdi, 112
test sil, sil
sub cl, -59
and rdi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
cmovnle ecx, dword ptr [r14 + rax]
imul ax, di
and rdi, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
bts word ptr [r14 + rdi], cx
nop ebx
and rcx, 0b1111111111111 # instrumentation
xor si, word ptr [r14 + rcx]
cmovbe bx, cx
and rdx, 0b1111111111111 # instrumentation
mov edi, dword ptr [r14 + rdx]
lea eax, qword ptr [rdx + rbx]
xor si, 18
adc di, si
xadd dl, bl
and rdx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdx], dl
cmovno rcx, rsi
movzx ebx, cl
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
