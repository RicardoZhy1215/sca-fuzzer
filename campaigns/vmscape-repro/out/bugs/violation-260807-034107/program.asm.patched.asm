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
add cl, 62 # instrumentation
cmovnp rax, rsi
and ax, 10
cmp dl, cl
and rax, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rax], dl
xadd cl, dl
cmovz cx, dx
xchg ecx, eax
seto al
neg cl
and rax, 0b1111111111111 # instrumentation
movsx si, byte ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
not byte ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], -58
test sil, dl
cmovno bx, dx
cmovnz cx, bx
or si, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv si
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rax], dl
cmovnz rax, rax
cmovnb rsi, rsi
inc dl
sub sil, sil
xchg ebx, eax
xchg dl, dl
and rcx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rcx], -118
setp dl
and rsi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rsi], -45
and rcx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rcx], al
and rcx, 0b1111111111111 # instrumentation
sub rax, qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rsi]
setl al
nop ebx
and rbx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rbx], dl
test sil, 54
and rdx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdx], ebx
btc rdx, 141
and rax, 0b1111111111111 # instrumentation
bts word ptr [r14 + rax], 6
and rdi, 0b1111111111000 # instrumentation
and ax, 0b111 # instrumentation
lock bts word ptr [r14 + rdi], ax
sbb dl, dl
and rdx, 0b1111111111111 # instrumentation
mov esi, dword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rbx]
add bx, si
and rax, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rax], 0
add dl, -48 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovl ecx, dword ptr [r14 + rdx]
cmovnp cx, si
or al, -102
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], -17
adc al, cl
and rax, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rax], eax
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
and rdx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rdx], ax
or al, 31
and rdi, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
bt word ptr [r14 + rcx], 0
xchg rcx, rax
cmp edx, -102
cmc
adc rdx, rcx
lea esi, qword ptr [rcx + rbx + 14028]
bt rdi, 68
sub cl, -93
cmovnz ebx, eax
lea dx, qword ptr [rdi + rax]
or dil, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv dil
add dl, -59 # instrumentation
and rsi, 0b1111111111000 # instrumentation
lock dec word ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rcx]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
mul edx
cmovb rdx, rdx
and rbx, 0b1111111111111 # instrumentation
xor edi, dword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
and eax, 0b111 # instrumentation
lock btr dword ptr [r14 + rdx], eax
and rsi, 0b1111111111000 # instrumentation
and rdx, 0b111 # instrumentation
lock bts qword ptr [r14 + rsi], rdx
and rax, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rax], rax
cmp ax, 8992
setnl cl
cmovo rbx, rdx
test esi, -1712078691
sbb cl, 24
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 79
and rbx, 0b1111111111111 # instrumentation
not qword ptr [r14 + rbx]
setnbe bl
add al, 92
and rcx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rcx], -99
lea cx, qword ptr [rdx + rdi + 16824]
and rdi, 0b1111111111111 # instrumentation
sbb dl, byte ptr [r14 + rdi]
cmp bl, 104
cmp dl, -112
and rdi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rdi]
lea eax, qword ptr [rdi]
btc di, 52
add dl, dl
and rcx, 0b1111111111111 # instrumentation
sub di, word ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmovns eax, dword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmp ebx, dword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
mov cl, byte ptr [r14 + rdi]
xor cl, -29
and rax, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rax], 774134030
lea dx, qword ptr [rcx + rdx + 27574]
cmovno eax, edi
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
