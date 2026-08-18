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
or cl, 31
setnbe bl
cmovbe rax, rsi
adc al, 81
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], cl
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], rsi
lea rdi, qword ptr [rcx + rdx + 19315]
and rdi, 0b1111111111111 # instrumentation
adc bl, byte ptr [r14 + rdi]
xor dl, bl
and rdi, 0b1111111111111 # instrumentation
cmovz rsi, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 0b1000000000000000 # instrumentation
bsr bx, word ptr [r14 + rbx]
add cl, -90 # instrumentation
mov cl, bl
sbb al, dil
adc ebx, ebx
lea edi, qword ptr [rcx]
mov esi, -2052538147
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], dl
and rsi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
movsx eax, word ptr [r14 + rdi]
and rsi, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rsi]
and dl, -52
and rcx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rcx]
cmovo ebx, eax
or eax, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv eax
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], bl
setb dl
and rdx, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
btr qword ptr [r14 + rdx], rsi
lea esi, qword ptr [rsi + rdx]
and rax, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rax]
lea rcx, qword ptr [rbx + rax]
lea rax, qword ptr [rcx]
and bl, -93
cmovns rdx, rdi
lea esi, qword ptr [rax + rax]
add eax, edi
sub dl, cl
test rax, -46517277
cmovno dx, si
and rdx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdx], esi
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], 1 # instrumentation
mov ax, 1 # instrumentation
div byte ptr [r14 + rdi]
or eax, 0b1000000000000000000000000000000 # instrumentation
bsf edx, eax
and rdi, 0b1111111111000 # instrumentation
and rsi, 0b111 # instrumentation
lock btc qword ptr [r14 + rdi], rsi
and rdi, 0b1111111111000 # instrumentation
and ecx, 0b111 # instrumentation
lock bts dword ptr [r14 + rdi], ecx
and rdx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdx], edi
and rdi, 0b1111111111000 # instrumentation
and rsi, 0b111 # instrumentation
lock btr qword ptr [r14 + rdi], rsi
and rsi, 0b1111111111000 # instrumentation
lock xadd dword ptr [r14 + rsi], edx
test si, -19324
and rdx, 0b1111111111111 # instrumentation
cmovle dx, word ptr [r14 + rdx]
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
add cl, -100 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovnz rcx, rbx
setz dil
add ax, 4284
cmovns rdi, rdx
btc eax, edi
stc
xor eax, 928505127
cmovb rsi, rbx
sub rsi, 75
and rdx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdx], cl
xor ebx, -53
not bx
and rcx, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
btr word ptr [r14 + rcx], bx
cmp cl, cl
bts esi, edx
and rsi, 0b1111111111111 # instrumentation
movzx rdi, word ptr [r14 + rsi]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea edx, qword ptr [rax + rcx]
sub ax, -20474
bts cx, 148
btc rdx, rdi
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
add dl, byte ptr [r14 + rdx]
dec rax
or dl, -50
sbb bl, al
and rdx, 0b1111111111111 # instrumentation
sbb dil, byte ptr [r14 + rdx]
setp cl
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi
xchg bl, al
lea ax, qword ptr [rcx + rcx]
cmp dx, 53
lea rsi, qword ptr [rdx]
imul rbx, rsi, -58
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rbx], dil
nop rcx
setnp cl
lea eax, qword ptr [rsi + rcx]
and rdx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rdx], rbx
lea ecx, qword ptr [rax + rcx]
and rcx, 0b1111111111111 # instrumentation
and rax, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdx], 1 # instrumentation
and rax, -860942512
and rsi, 0b1111111111000 # instrumentation
and bx, 0b111 # instrumentation
lock btr word ptr [r14 + rsi], bx
bt ebx, edi
add rsi, rdi
cmpxchg cl, bl
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
