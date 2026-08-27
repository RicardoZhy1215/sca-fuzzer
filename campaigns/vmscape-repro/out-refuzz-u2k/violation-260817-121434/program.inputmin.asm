.intel_syntax noprefix
.section .data.main
.function_main_0:
.macro.set_h2g_target.vm.function_vm_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_1: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.0: nop qword ptr [rax + 0xff]
.function_main_1:
.macro.landing_g2h.main_1: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_0: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vmvictim.0: nop qword ptr [rax + 0xff]
.function_main_2:
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
.function_main_3:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]
.function_main_4:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
.section .data.vmvictim
.function_vmvictim_0:
.macro.landing_h2g.vmvictim_0: nop qword ptr [rax + 0xff]
add dl, -2 # instrumentation
lea ecx, qword ptr [rdi + rcx + 23504]
setns cl
and rcx, 0b1111111111111 #
dec cx
and rbx, 0b1111111111111 # instrumentation
inc byte ptr [r14 + rbx]
# mem access: [12] 0x61f0-0x61f0 cl 7:48 | [62] 0x61f0-0x61f0 cl 7:48
and rbx, 0b1111111111000 # instrumentation
and rax, 0b111 # instrumentation
mov qword ptr [r14 + rbx], rax
# mem access: [12] 0x61f0 cl 7:48 | [62] 0x61f0 cl 7:48
mov rbx, 0b1111111111111 #
and rbx, 0b1111111111111 # instrumentation
movsx edi, byte ptr [r14 + rbx]
# mem access: [12] 0x6fff cl 63:63 | [62] 0x6fff cl 63:63
and rax, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rax]
# mem access: [12] 0x5000 cl 0:0 | [62] 0x5000 cl 0:0
and rcx, 0b1111111111111 #
mov rsi, 0b1111111111111 #
add al, 52 # instrumentation
mov eax, edi
mov rbx, 0b1111111111111 #
and rbx, 0b1111111111111 # instrumentation
mov rdx, 0b111 #
mov rax, 0b1111111111000 #
and rcx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rcx]
# mem access: [12] 0x6d00 cl 52:0 | [62] 0x6d00 cl 52:0
mov rax, 0b1111111111111 #
mov rbx, 0b1111111111111 #
add rdx, 0b1111111111111 #
mov rax, 0b1111111111111 #
add rdx, 0b1111111111111 #
mov rbx, 0b1111111111111 #
mov rcx, 0b1111111111111 #
and rdx, 0b1111111111111 # instrumentation
mov si, 0b111 #
mov rbx, 0b1111111111111 #
and rcx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rcx], 1 #
# mem access: [12] 0x6fff cl 63:63 | [62] 0x6fff cl 63:63
add dl, -58
and rdx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rdx]
# mem access: [12] 0x536f cl 13:47 | [62] 0x536f cl 13:47
mov rdx, 0b1111111111111 #
mov rdx, 0b1111111111111 #
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence
.section .data.vm
.function_vm_0:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence
.function_vm_1:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
# mem access: [12] 0xb000 cl 0:0 | [62] 0xb000 cl 0:0
mov rbx, qword ptr [r14 +0x2008] # noremove
# mem access: [12] 0xb008 cl 0:8 | [62] 0xb008 cl 0:8
mov rcx, qword ptr [r14 +0x2010] # noremove
# mem access: [12] 0xb010 cl 0:16 | [62] 0xb010 cl 0:16
mov rdx, qword ptr [r14 +0x2018] # noremove
# mem access: [12] 0xb018 cl 0:24 | [62] 0xb018 cl 0:24
mov rsi, qword ptr [r14 +0x2020] # noremove
# mem access: [12] 0xb020 cl 0:32 | [62] 0xb020 cl 0:32
mov rdi, qword ptr [r14 +0x2028] # noremove
# mem access: [12] 0xb028 cl 0:40 | [62] 0xb028 cl 0:40
mfence  # noremove
and rbx, 0b1111111111000 # instrumentation
cmpxchg dword ptr [r14 + rbx], edi
# mem access: [12] 0x9018-0x9018 cl 0:24 | [62] 0x9018-0x9018 cl 0:24
and rsi, 0b1111111111000 # instrumentation
cmpxchg byte ptr [r14 + rsi], cl
# mem access: [12] 0x9350-0x9350 cl 13:16 | [62] 0x9350-0x9350 cl 13:16
and rax, 0b1111111111000 # instrumentation
add word ptr [r14 + rax], -60
# mem access: [12] 0xafd0-0xafd0 cl 63:16 | [62] 0xafd0-0xafd0 cl 63:16
mov dl, dl
and rcx, 0b1111111111000 # instrumentation
mov qword ptr [r14 + rcx], rbx
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], sil
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], 112
mov cl, 69
add al, 47
add dx, bx
and rax, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rax], 1
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], dil
and rdx, 0b1111111111000 # instrumentation
and ax, 0b111 # instrumentation
btr word ptr [r14 + rdx], ax
and rdx, 0b1111111111000 # instrumentation
mov word ptr [r14 + rdx], 116
and rax, 0b1111111111111 # instrumentation
mov dl, byte ptr [r14 + rax]
and rax, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], 18
and rdi, 0b1111111111111 # instrumentation
mov ecx, dword ptr [r14 + rdi]
add rdi, 0b1111111111111 #
and rdi, 0b1111111111111 # instrumentation
add bl, byte ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
mov byte ptr [r14 + rbx], al
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence
.function_vm_2:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence
.section .data.main
.test_case_exit:nop
