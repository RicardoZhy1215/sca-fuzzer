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
.patch_placeholder:
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
add dl, 22 # instrumentation
and rdx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rdx], 23
not ax
and rsi, 0b1111111111111 # instrumentation
and rbx, 0b1111111111111 # instrumentation
sub rax, qword ptr [r14 + rbx]
movsx rax, cl
cmpxchg al, cl
xor rbx, rsi
and rax, 0b1111111111111 # instrumentation
cmovnle si, word ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], dl
add dl, dl
sub sil, dil
and rbx, 0b1111111111111 # instrumentation
sbb bx, word ptr [r14 + rbx]
cmovnbe edx, ebx
lea rdx, qword ptr [rax]
add sil, -99
lea esi, qword ptr [rsi]
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 104
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rdx], dx
and rcx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rcx], -13
lea rsi, qword ptr [rdx + rsi + 21390]
cmovns rax, rbx
sub si, ax
btr rsi, 82
cmpxchg rdi, rdx
cmovbe rsi, rcx
and rsi, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000000000000000 # instrumentation
bsr di, word ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rsi], 2
add dl, -128 # instrumentation
movzx eax, bl
and rcx, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rcx]
test ax, 18145
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], rdi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
and rsi, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rsi], edx
lea rdx, qword ptr [rcx]
setp cl
setnp cl
or ebx, 0b1000000000000000000000000000000 # instrumentation
bsf ecx, ebx
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdx
or bl, al
and rbx, 0b1111111111111 # instrumentation
cmovno rbx, qword ptr [r14 + rbx]
lea rax, qword ptr [rsi + rdi]
and rbx, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rbx], dx
and rdx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rdx], rcx
cbw
sbb ax, 20030
and rbx, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rbx]
and bl, cl
and rsi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rbx], ebx
xor dx, ax
and rdx, 0b1111111111000 # instrumentation
and rax, 0b111 # instrumentation
lock btr qword ptr [r14 + rdx], rax
and rax, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rax], sil
and rdi, 0b1111111111111 # instrumentation
cmovbe dx, word ptr [r14 + rdi]
adc al, -103
and rdi, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
bt dword ptr [r14 + rdi], edi
and rdx, 0b1111111111111 # instrumentation
sbb ax, word ptr [r14 + rdx]
sub bx, 108
and rsi, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rbx]
nop
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rbx
and rdx, 0b1111111111111 # instrumentation
cmovnl edi, dword ptr [r14 + rdx]
inc ax
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence
.exit_vmvictim_0:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
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
movzx rdx, dil
lea ecx, qword ptr [rdx + rsi + 6631]
lea ecx, qword ptr [rbx + rbx + 10774]
mul bl
and rcx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rcx], 2
add bl, 97 # instrumentation
cmovnle cx, cx
sub sil, -9
lea ax, qword ptr [rax + rcx + 28466]
test dl, -82
test dil, -103
and rdi, 0b1111111111111 # instrumentation
mov si, word ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdx], edi
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl
and rax, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rax], rdi
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], al
lea ax, qword ptr [rdx + rbx]
xchg edx, eax
and rbx, 0b1111111111111 # instrumentation
cmovno rcx, qword ptr [r14 + rbx]
or edi, 0b1000000000000000000000000000000 # instrumentation
bsf edi, edi
lea bx, qword ptr [rbx + rdx + 53035]
cmpxchg rbx, rbx
and rax, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rax], cl
cmp rbx, -91
and rdx, 0b1111111111111 # instrumentation
or cl, byte ptr [r14 + rdx]
cmp bl, bl
and rbx, 0b1111111111000 # instrumentation
and ebx, 0b111 # instrumentation
lock btr dword ptr [r14 + rbx], ebx
and rcx, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rcx], dx
and rdi, 0b1111111111111 # instrumentation
adc rsi, qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rcx], bx
cmovnbe esi, ecx
sets dl
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx]
mov bl, dil
movzx si, dl
and rax, 0b1111111111111 # instrumentation
cmovnbe ax, word ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rcx], esi
and rbx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rbx], -54
xor ebx, edi
clc
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rsi]
btc cx, dx
not ecx
and rbx, 0b1111111111000 # instrumentation
lock bts dword ptr [r14 + rbx], 7
add dl, 8 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnp edi, dword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 125
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], dl
or rax, -1022429208
and rbx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rbx], -10
lea di, qword ptr [rbx]
and rdi, 0b1111111111111 # instrumentation
sbb ebx, dword ptr [r14 + rdi]
sets bl
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rbx]
add cl, cl
setl dl
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl
and rdi, 0b1111111111111 # instrumentation
and di, word ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
and ax, 0b111 # instrumentation
lock btr word ptr [r14 + rdi], ax
movsx si, dl
adc ax, 58
and rsi, rax
and rax, 0b1111111111111 # instrumentation
setl byte ptr [r14 + rax]
adc cl, cl
lea esi, qword ptr [rdi]
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence
.exit_vm_1:
.function_vm_2:
.bb_vm_2.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence
.exit_vm_2:
.section .data.main
.test_case_exit:nop
