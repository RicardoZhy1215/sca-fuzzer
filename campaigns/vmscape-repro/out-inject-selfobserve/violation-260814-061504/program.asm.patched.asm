.intel_syntax noprefix
.section .data.main
.function_main_0:
.bb_main_0.entry:
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]
.exit_main_0:
.function_main_1:
.bb_main_1.entry:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]
.exit_main_1:
.function_main_2:
.bb_main_2.entry:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.switch_h2g.vmvictim.1: nop qword ptr [rax + 0xff]
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vmvictim.function_vmvictim_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.2: nop qword ptr [rax + 0xff]
.exit_main_3:
.function_main_4:
.bb_main_4.entry:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop
.exit_main_4:
.section .data.vm
.function_vm_train:
.bb_vm_train.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
add dl, -122 # instrumentation
mov rcx, rdi
mov bx, di
cmovnz rcx, rbx
lea rbx, qword ptr [rbx + rax + 46081]
add bl, al
sbb bl, -88
cmovno rcx, rdx
mov cl, bl
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsf edi, dword ptr [r14 + rdx]
test eax, -1542051919
cmp bl, al
add al, al
and ax, -22660
dec al
and rdx, 0b1111111111111 # instrumentation
cmp sil, byte ptr [r14 + rdx]
or ax, 0b1000000000000000 # instrumentation
bsf si, ax
lea rax, qword ptr [rip +.function_vm_gadget] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_gadget:
.bb_vm_gadget.entry:
and rax, 0b1111111111111 # instrumentation
add word ptr [r14 + rax], 41
lea ecx, qword ptr [rdx + rdi]
lea rdi, qword ptr [rdx + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx]
adc al, -73
and rax, 0b1111111111111 # instrumentation
imul word ptr [r14 + rax]
sbb al, cl
and rdi, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rdi], 3
add bl, -61 # instrumentation
cmovnle esi, edi
sub al, -37
cmovbe bx, dx
lea ecx, qword ptr [rbx + rcx + 46673]
cmovb edx, esi
lea si, qword ptr [rsi + rcx]
and rbx, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rbx], ebx
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], al
and rsi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rsi], al
and rbx, 0b1111111111111 # instrumentation
cmovnp di, word ptr [r14 + rbx]
add cl, -116
add sil, al
cmovnle rsi, rdi
and rbx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rbx]
setb cl
or bl, 22
xchg ax, ax
test bl, bl
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
add al, 112 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], eax
setnle dl
lea ecx, qword ptr [rcx]
setle dl
cmovnz esi, edi
lfence  # noremove
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence
.exit_vm_gadget:
.section .data.vmvictim
.function_vmvictim_0:
.bb_vmvictim_0.entry:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vmvictim_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rsi, 72
lea ebx, qword ptr [rcx + rcx + 58210]
and rax, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
bt dword ptr [r14 + rax], edi
add cl, 117 # instrumentation
sbb ecx, 105
setnle bl
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], -102
setle sil
btc rdi, 222
lea rdi, qword ptr [rsi + rdx]
lea rax, qword ptr [rdi]
and rcx, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rcx], 1
xchg dl, al
and rdx, 0b1111111111111 # instrumentation
and rbx, 0b111 # instrumentation
btc qword ptr [r14 + rdx], rbx
add dl, dl
btr rsi, 80
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
and rdx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdx], 88
and rbx, 0b1111111111111 # instrumentation
cmovl edx, dword ptr [r14 + rbx]
and ax, 22169
lea rbx, qword ptr [rbx + rcx + 13287]
and rax, 0b1111111111111 # instrumentation
cmovl ax, word ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi]
adc al, bl
and rsi, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rsi], -27
and rax, 0b1111111111111 # instrumentation
cmovnb edx, dword ptr [r14 + rax]
bt esi, ebx
cmovz rcx, rbx
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], dl
and bl, al
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], al
and rbx, 0b1111111111111 # instrumentation
imul rsi, qword ptr [r14 + rbx]
xchg rbx, rdx
xadd eax, eax
mul sil
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], al
and rax, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rax], sil
lea rcx, qword ptr [rdi + rdi]
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], al
setnz bl
cmovnp ax, si
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rdi], edx
btr eax, 123
and rdi, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rdi]
add cx, cx
and rax, 0b1111111111111 # instrumentation
and byte ptr [r14 + rax], -96
test eax, 89958315
and rsi, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rsi], di
and rax, 0b1111111111111 # instrumentation
imul edx, dword ptr [r14 + rax], 95
lfence  # noremove
.macro.measurement_end.vmvictim_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence
.exit_vmvictim_3:
.function_vmvictim_2:
.bb_vmvictim_2.entry:
.macro.landing_h2g.vmvictim_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vmvictim_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence
.exit_vmvictim_2:
.section .data.main
.test_case_exit:nop
