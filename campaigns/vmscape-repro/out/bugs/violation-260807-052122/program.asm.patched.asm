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
add dl, -20 # instrumentation
not cx
cmovno rdx, rsi
adc ebx, -11
and rdi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdi], 86
and rdx, 0b1111111111111 # instrumentation
imul rbx, qword ptr [r14 + rdx], 76
xor bl, dl
and rax, 1403753617
and dx, -47
sub eax, 86
lea bx, qword ptr [rdi + rcx + 47702]
sub eax, ebx
and rsi, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rdi]
setp dl
and cx, 79
and rdi, 0b1111111111111 # instrumentation
cmovnl edx, dword ptr [r14 + rdi]
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add al, -62 # instrumentation
setnl cl
adc cx, bx
and rdx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdx], cx
lea ecx, qword ptr [rdx + rdx + 60247]
xadd al, al
bt esi, 176
and rsi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rsi], -76
and rdx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdx], -39
and rsi, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rsi], 4
add bl, 69 # instrumentation
movzx edx, cx
and rdi, 0b1111111111111 # instrumentation
cmovl ax, word ptr [r14 + rdi]
lea edi, qword ptr [rsi + rdx]
cmp bl, 5
and rbx, 0b1111111111111 # instrumentation
btr word ptr [r14 + rbx], 0
and rax, 0b1111111111111 # instrumentation
and qword ptr [r14 + rax], 81
cmovns si, si
and rax, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rax]
cmovnb edi, ebx
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rax], 99
cmp bl, sil
and rax, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
adc al, byte ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rdi], edi
setp bl
sub rsi, rdi
seto sil
and rax, 0b1111111111111 # instrumentation
imul ecx, dword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rbx], 116
or ecx, edx
and rsi, 0b1111111111111 # instrumentation
bts word ptr [r14 + rsi], 1
xor cl, dl
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
add al, dl
add cl, bl
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rsi]
mul bl
add bl, 105 # instrumentation
and rax, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
add edx, dword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
cmovnp bx, word ptr [r14 + rax]
mul dl
sbb rax, 189994064
cmovb eax, ecx
imul rax
test bl, 117
bswap esi
and rdx, 0b1111111111111 # instrumentation
and rdi, 0b1111111111111 # instrumentation
movsx eax, word ptr [r14 + rdi]
lea edi, qword ptr [rdi]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add bl, -112 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovns di, word ptr [r14 + rdi]
btc di, cx
add cl, -86 # instrumentation
and rdi, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rdi]
setle al
lea ebx, qword ptr [rax]
sub rax, -668799665
or dl, dl
cmovz rdi, rax
and rdx, 0b1111111111111 # instrumentation
sbb sil, byte ptr [r14 + rdx]
adc al, cl
and rax, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rax], -76
and rbx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rbx], -6
and rdi, 0b1111111111111 # instrumentation
cmovs dx, word ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rdi], ebx
and rbx, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rbx], dx
and rsi, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rsi], di
or cl, bl
and rbx, 0b1111111111111 # instrumentation
add ebx, dword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rdx]
cmp al, -92
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rdi], dl
and cl, -7
and rcx, 0b1111111111111 # instrumentation
movsx di, byte ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rax], 60
cmp rax, 1880063776
or bl, cl
and cl, -108
cmovle ax, si
cmovnle cx, di
and rdx, 0b1111111111111 # instrumentation
cmovnb esi, dword ptr [r14 + rdx]
sub ax, -13381
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
