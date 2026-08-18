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
lea ax, qword ptr [rcx + rcx]
sub al, -83
cwde
lea ecx, qword ptr [rdx + rcx + 48613]
add eax, -889906340
mov cx, cx
and rdi, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdi], -85
dec dil
sbb dl, dl
setnz cl
and al, al
add edx, esi
xor al, 118
cmovz bx, cx
lea ebx, qword ptr [rdx + rbx + 60903]
cmovle esi, ebx
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
xor esi, ebx
and rdi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdi], al
and rbx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rbx], -79
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], si
setnle cl
and rcx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rcx], -103
and rsi, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rsi], -76
and rdx, 0b1111111111111 # instrumentation
sbb dil, byte ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsf rdi, qword ptr [r14 + rdi]
add bl, 42 # instrumentation
cmovnbe eax, ecx
sub bl, bl
cmovz si, cx
and dl, cl
and rbx, 0b1111111111111 # instrumentation
xor dx, word ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
not qword ptr [r14 + rdi]
not sil
test rax, rdi
and rax, 0b1111111111111 # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmovns si, word ptr [r14 + rcx]
adc dl, -54
adc edx, 47
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rbx
and rbx, 0b1111111111111 # instrumentation
sbb ebx, dword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
sub dl, byte ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rax], sil
and rdx, 0b1111111111111 # instrumentation
neg word ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
test word ptr [r14 + rcx], -3605
and rcx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rbx], -8
and rbx, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rbx]
and rax, 0b1111111111111 # instrumentation
or ebx, dword ptr [r14 + rax]
movzx esi, dl
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
or sil, 1 # instrumentation
mov ax, 1 # instrumentation
div sil
and rsi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsr rdi, qword ptr [r14 + rsi]
cmp rsi, rdi
and rdx, -5
cmp sil, 122
or edi, 0b1000000000000000000000000000000 # instrumentation
bsr esi, edi
add al, -79 # instrumentation
adc eax, 47
sahf
bt rbx, 59
and rdx, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rdx]
adc rbx, rdx
imul cl
sub dl, -71
cmpxchg eax, esi
imul dx, cx, 85
xor ax, dx
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rsi, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rsi], edx
mul rbx
add cl, -128 # instrumentation
lea rsi, qword ptr [rdi + rcx]
and rbx, 0b1111111111111 # instrumentation
setnle byte ptr [r14 + rbx]
sub bl, 35
sub cl, 113
or di, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv di
add al, 40 # instrumentation
lea ax, qword ptr [rbx]
lea ax, qword ptr [rdi]
setnl cl
bt rdi, 30
and rbx, 0b1111111111000 # instrumentation
and di, 0b111 # instrumentation
lock btr word ptr [r14 + rbx], di
btc si, ax
and rax, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rax], -16
mul rax
lea ebx, qword ptr [rdx + rdi]
and rdi, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdi], rdx
and bl, al
and rdx, 0b1111111111111 # instrumentation
cmovnp rdi, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rbx], 15
lea edx, qword ptr [rsi + rbx + 31710]
cmovo dx, bx
and rax, 0b1111111111111 # instrumentation
xor cx, word ptr [r14 + rax]
xor al, -88
add al, 84 # instrumentation
lea rbx, qword ptr [rdx + rax]
lahf
and rbx, 0b1111111111111 # instrumentation
cmovno rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovle cx, word ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rcx]
lea rbx, qword ptr [rbx + rdx + 44096]
cmovl rsi, rcx
and rdx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rdx], 34
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
