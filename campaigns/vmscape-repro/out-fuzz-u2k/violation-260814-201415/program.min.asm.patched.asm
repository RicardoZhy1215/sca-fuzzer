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
add dl, 22 # instrumentation
and rdx, 0b1111111111111 # instrumentation
add word ptr [r14 + rdx], 23
# mem access: [0] 0x692c-0x692c cl 36:44 | [50] 0x692c-0x692c cl 36:44
inc ax
and rsi, 0b1111111111111 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rbx]
# mem access: [0] 0x5483 cl 18:3 | [50] 0x5483 cl 18:3
mov al, cl
mov rbx, rsi
and rax, 0b1111111111111 # instrumentation
mov si, word ptr [r14 + rax]
# mem access: [0] 0x56bf cl 26:63 | [50] 0x5bbf cl 46:63
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], dl
# mem access: [0] 0x6758-0x6758 cl 29:24 | [50] 0x5460-0x5460 cl 17:32
mov dl, dl
add sil, dil
and rbx, 0b1111111111111 # instrumentation
add bx, word ptr [r14 + rbx]
# mem access: [0] 0x6f2c cl 60:44 | [50] 0x6f2c cl 60:44
mov edx, ebx
lea rdx, qword ptr [rax]
mov sil, -99
lea esi, qword ptr [rsi]
and rbx, 0b1111111111111 #
and rdx, 0b1111111111000 # instrumentation
mov word ptr [r14 + rdx], dx
# mem access: [0] 0x56b8 cl 26:56 | [50] 0x5bb8 cl 46:56
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], -13
# mem access: [0] 0x57bf cl 30:63 | [50] 0x57bf cl 30:63
lea rsi, qword ptr [rdx + rsi + 21390]
mov rax, rbx
add si, ax
mov rsi, 82
cmpxchg rdi, rdx
mov rsi, rcx
and rsi, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rsi]
# mem access: [0] 0x57bf cl 30:63 | [50] 0x57bf cl 30:63
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000000000000000 # instrumentation
# mem access: [0] 0x6195-0x6195 cl 6:21 | [50] 0x6195-0x6195 cl 6:21
mov di, word ptr [r14 + rdi]
# mem access: [0] 0x6195 cl 6:21 | [50] 0x6195 cl 6:21
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2
# mem access: [0] 0x57bf cl 30:63 | [50] 0x57bf cl 30:63
add dl, -128 # instrumentation
movzx eax, bl
and rcx, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rcx]
# mem access: [0] 0x57bf cl 30:63 | [50] 0x57bf cl 30:63
add ax, 18145
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi
# mem access: [0] 0x5cc3 cl 51:3 | [50] 0x6850 cl 33:16
and rdx, 0b1111111111000 # instrumentation
add qword ptr [r14 + rdx], rsi
# mem access: [0] 0x5638-0x5638 cl 24:56 | [50] 0x5b38-0x5b38 cl 44:56
and rsi, 0b1111111111111 #
lea rdx, qword ptr [rcx]
setp cl
or ebx, 0b1000000000000000000000000000000 # instrumentation
add ecx, ebx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx
# mem access: [0] 0x57bf cl 30:63 | [50] 0x57bf cl 30:63
mov bl, al
add rbx, 0b1111111111111 #
lea rax, qword ptr [rsi + rdi]
and rbx, 0b1111111111111 # instrumentation
add word ptr [r14 + rbx], dx
# mem access: [0] 0x5ca3-0x5ca3 cl 50:35 | [50] 0x6830-0x6830 cl 32:48
and rdx, 0b1111111111000 # instrumentation
add qword ptr [r14 + rdx], rcx
# mem access: [0] 0x57b8-0x57b8 cl 30:56 | [50] 0x57b8-0x57b8 cl 30:56
cbw
mov ax, 20030
and rbx, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rbx]
# mem access: [0] 0x5ca3 cl 50:35 | [50] 0x6830 cl 32:48
mov bl, cl
and rsi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rsi]
# mem access: [0] 0x57bf-0x57bf cl 30:63 | [50] 0x57bf-0x57bf cl 30:63
add rbx, 0b1111111111111 #
and rdx, 0b1111111111000 # instrumentation
and rax, 0b111 #
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], sil
# mem access: [0] 0x5006 cl 0:6 | [50] 0x5006 cl 0:6
mov rdi, 0b1111111111111 #
and rdi, 0b1111111111111 # instrumentation
mov edi, 0b111 #
mov rsi, 0b1111111111111 #
mov rcx, 0b1111111111000 #
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
# mem access: [0] 0xb000 cl 0:0 | [50] 0xb000 cl 0:0
mov rbx, qword ptr [r14 +0x2008] # noremove
# mem access: [0] 0xb008 cl 0:8 | [50] 0xb008 cl 0:8
mov rcx, qword ptr [r14 +0x2010] # noremove
# mem access: [0] 0xb010 cl 0:16 | [50] 0xb010 cl 0:16
mov rdx, qword ptr [r14 +0x2018] # noremove
# mem access: [0] 0xb018 cl 0:24 | [50] 0xb018 cl 0:24
mov rsi, qword ptr [r14 +0x2020] # noremove
# mem access: [0] 0xb020 cl 0:32 | [50] 0xb020 cl 0:32
mov rdi, qword ptr [r14 +0x2028] # noremove
# mem access: [0] 0xb028 cl 0:40 | [50] 0xb028 cl 0:40
mfence  # noremove
movzx rdx, dil
lea ecx, qword ptr [rdx + rsi + 6631]
lea ecx, qword ptr [rbx + rbx + 10774]
and rcx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rcx], 2
# mem access: [0] 0xa140 cl 5:0 | [50] 0xa140 cl 5:0
add bl, 97 #
mov sil, -9
test dil, -103
and rdi, 0b1111111111111 # instrumentation
mov si, word ptr [r14 + rdi]
# mem access: [0] 0x96c3 cl 27:3 | [50] 0x96c3 cl 27:3
and rdx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rdx], edi
# mem access: [0] 0x90c3-0x90c3 cl 3:3 | [50] 0x90c3-0x90c3 cl 3:3
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
inc cl
and rax, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rax], rdi
# mem access: [0] 0x9001-0x9001 cl 0:1 | [50] 0x9001-0x9001 cl 0:1
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], al
# mem access: [0] 0x96c3 cl 27:3 | [50] 0x96c3 cl 27:3
lea ax, qword ptr [rdx + rbx]
xchg edx, eax
and rbx, 0b1111111111111 # instrumentation
cmovno rcx, qword ptr [r14 + rbx]
# mem access: [0] 0x93f6 cl 15:54 | [50] 0x93f6 cl 15:54
or edi, 0b1000000000000000000000000000000 # instrumentation
bsf edi, edi
lea bx, qword ptr [rbx + rdx + 53035]
cmpxchg rbx, rbx
and rax, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rax], cl
# mem access: [0] 0xa7d8-0xa7d8 cl 31:24 | [50] 0xa7d8-0xa7d8 cl 31:24
add rbx, -91
and rdx, 0b1111111111111 # instrumentation
add cl, byte ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
and ebx, 0b111 # instrumentation
lock btr dword ptr [r14 + rbx], ebx
and rcx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rcx], dx
and rdi, 0b1111111111111 #
mov rcx, 0b1111111111000 #
and rbx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rbx]
mov bl, dil
and rax, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000 # instrumentation
add byte ptr [r14 + rsi], 0b11111000 #
add cx, dx
mov rbx, 0b1111111111000 #
mov dl, 8 #
and rbx, 0b1111111111111 # instrumentation
mov word ptr [r14 + rbx], 125
add rcx, 0b1111111111111 #
mov rax, -1022429208
and rdi, 0b1111111111111 #
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi]
add bl, 1 #
and rdi, 0b1111111111111 # instrumentation
mov di, word ptr [r14 + rdi]
and rdi, 0b1111111111000 # instrumentation
add ax, 0b111 #
movsx si, dl
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
