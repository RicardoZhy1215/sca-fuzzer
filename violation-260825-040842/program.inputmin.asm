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
and rbx, 0b1111111111000 # instrumentation
neg byte ptr [r14 + rbx]
# mem access: [39] 0x5bc8-0x5bc8 cl 47:8 | [89] 0x5bc8-0x5bc8 cl 47:8
lea di, qword ptr [rsi]
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], -15
# mem access: [39] 0x5000-0x5000 cl 0:0 | [89] 0x5000-0x5000 cl 0:0
lea si, qword ptr [rdx + rcx + 12555]
sbb bl, al
cmp rdx, rdi
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 0b1000 # instrumentation
# mem access: [39] 0x53fa-0x53fa cl 15:58 | [89] 0x53fa-0x53fa cl 15:58
and byte ptr [r14 + rcx], 0b11111000 # instrumentation
# mem access: [39] 0x53fa-0x53fa cl 15:58 | [89] 0x53fa-0x53fa cl 15:58
and edx, 0b11 # instrumentation
div dword ptr [r14 + rcx]
# mem access: [39] 0x53fa cl 15:58 | [89] 0x53fa cl 15:58
and rdi, 0b1111111111000 # instrumentation
xor dword ptr [r14 + rdi], ecx
# mem access: [39] 0x5db0-0x5db0 cl 54:48 | [89] 0x5db0-0x5db0 cl 54:48
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2
# mem access: [39] 0x6dba cl 54:58 | [89] 0x6dba cl 54:58
not bl
mov al, cl
and rdi, 0b1111111111111 # instrumentation
and word ptr [r14 + rdi], -16
# mem access: [39] 0x5db0-0x5db0 cl 54:48 | [89] 0x5db0-0x5db0 cl 54:48
and rax, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rax]
# mem access: [39] 0x50fa cl 3:58 | [89] 0x50fa cl 3:58
setnle cl
and rbx, 0b1111111111111 # instrumentation
mov ecx, dword ptr [r14 + rbx]
# mem access: [39] 0x5b38 cl 44:56 | [89] 0x5b38 cl 44:56
lea rbx, qword ptr [rsi + rax + 28813]
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], -76
# mem access: [39] 0x5db0-0x5db0 cl 54:48 | [89] 0x5db0-0x5db0 cl 54:48
and rsi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rsi]
# mem access: [39] 0x6dba cl 54:58 | [89] 0x6dba cl 54:58
movsx rbx, bx
and rcx, 0b1111111111000 # instrumentation
mov byte ptr [r14 + rcx], bl
# mem access: [39] 0x5818 cl 32:24 | [89] 0x5818 cl 32:24
and rdi, 0b1111111111111 # instrumentation
movzx rcx, word ptr [r14 + rdi]
# mem access: [39] 0x5db0 cl 54:48 | [89] 0x5db0 cl 54:48
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], 10
# mem access: [39] 0x5db0 cl 54:48 | [89] 0x5db0 cl 54:48
mov rbx, 71
lea rbx, qword ptr [rsi + rsi + 13240]
and rbx, 0b1111111111111 # instrumentation
mov di, word ptr [r14 + rbx]
# mem access: [39] 0x5f2c cl 60:44 | [89] 0x5f2c cl 60:44
mov al, bl
mov rax, rdx
and rcx, 0b1111111111111 # instrumentation
mov esi, dword ptr [r14 + rcx]
# mem access: [39] 0x5d44 cl 53:4 | [89] 0x5d44 cl 53:4
and rcx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rcx], al
# mem access: [39] 0x5d44 cl 53:4 | [89] 0x5d44 cl 53:4
seto al
mov rbx, rsi
and rdi, 0b1111111111111 # instrumentation
movsx cx, byte ptr [r14 + rdi]
# mem access: [39] 0x6cfd cl 51:61 | [89] 0x6cfd cl 51:61
xor eax, 1643943753
add al, al
movsx edi, cx
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], 1
# mem access: [39] 0x5f54 cl 61:20 | [89] 0x5f54 cl 61:20
inc eax
or di, 0b1000000000000000 # instrumentation
bsf cx, di
and rdi, 0b1111111111111 # instrumentation
mov bx, word ptr [r14 + rdi]
# mem access: [39] 0x500f cl 0:15 | [89] 0x500f cl 0:15
and rcx, 0b1111111111111 # instrumentation
and rax, qword ptr [r14 + rcx]
# mem access: [39] 0x5000 cl 0:0 | [89] 0x5000 cl 0:0
and eax, -35
and rcx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rcx], al
# mem access: [39] 0x5000 cl 0:0 | [89] 0x5000 cl 0:0
sub rdi, rax
sbb al, -91
setnle sil
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
# mem access: [39] 0x505b-0x505b cl 1:27 | [89] 0x505b-0x505b cl 1:27
add eax, dword ptr [r14 + rax]
# mem access: [39] 0x505b cl 1:27 | [89] 0x505b cl 1:27
movsx rdx, ax
and rdx, 0b1111111111111 # instrumentation
imul bx, word ptr [r14 + rdx], -52
# mem access: [39] 0x6dbc cl 54:60 | [89] 0x6dbc cl 54:60
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], -21
# mem access: [39] 0x5000 cl 0:0 | [89] 0x5000 cl 0:0
and rsi, 0b1111111111111 # instrumentation
mov edx, dword ptr [r14 + rsi]
# mem access: [39] 0x5f01 cl 60:1 | [89] 0x5f01 cl 60:1
and rax, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rax]
# mem access: [39] 0x6dbc cl 54:60 | [89] 0x6dbc cl 54:60
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1
# mem access: [39] 0x5f01 cl 60:1 | [89] 0x5f01 cl 60:1
dec cl
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], -96
# mem access: [39] 0x5000 cl 0:0 | [89] 0x5000 cl 0:0
and rcx, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rcx], 63
# mem access: [39] 0x50ff cl 3:63 | [89] 0x50ff cl 3:63
and rcx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rcx], 48
# mem access: [39] 0x50ff cl 3:63 | [89] 0x50ff cl 3:63
and rdx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdx]
# mem access: [39] 0x5e94 cl 58:20 | [89] 0x5e94 cl 58:20
mov al, -87
mov dl, dl
mov rsi, 0b1111111111111 #
mov rax, 0b1111111111111 #
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
add bl, -97 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
# mem access: [39] 0xb000 cl 0:0 | [89] 0xb000 cl 0:0
mov rbx, qword ptr [r14 +0x2008] # noremove
# mem access: [39] 0xb008 cl 0:8 | [89] 0xb008 cl 0:8
mov rcx, qword ptr [r14 +0x2010] # noremove
# mem access: [39] 0xb010 cl 0:16 | [89] 0xb010 cl 0:16
mov rdx, qword ptr [r14 +0x2018] # noremove
# mem access: [39] 0xb018 cl 0:24 | [89] 0xb018 cl 0:24
mov rsi, qword ptr [r14 +0x2020] # noremove
# mem access: [39] 0xb020 cl 0:32 | [89] 0xb020 cl 0:32
mov rdi, qword ptr [r14 +0x2028] # noremove
# mem access: [39] 0xb028 cl 0:40 | [89] 0xb028 cl 0:40
mfence  # noremove
and rbx, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rbx]
# mem access: [39] 0xa552-0x1ff8 cl 21:18 | [89] 0xa552-0x1ff8 cl 21:18
and rdx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rdi], 5
and rdx, 0b1111111111000 # instrumentation
btc qword ptr [r14 + rdx], 3
mov bl, sil
xor dil, 46
setnz sil
cmovz rdx, rdx
and rdi, 0b1111111111111 # instrumentation
add edi, dword ptr [r14 + rdi]
and rax, 0b1111111111000 # instrumentation
add byte ptr [r14 + rax], dil
add edi, edx
and rax, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rax], bx
add rax, rsi
lea rbx, qword ptr [rdi + rdi]
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi
and rdx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rdx], rbx
and rax, 0b1111111111111 # instrumentation
mov bx, word ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmp rdx, qword ptr [r14 + rax]
lea rax, qword ptr [rdi]
and rdx, 0b1111111111000 # instrumentation
mov dword ptr [r14 + rdx], edi
and rdi, 0b1111111111000 # instrumentation
cmpxchg byte ptr [r14 + rdi], cl
bt cx, bx
and rax, 0b1111111111000 # instrumentation
add qword ptr [r14 + rax], rdi
mov bl, -67
dec bl
and rbx, 0b1111111111000 # instrumentation
add word ptr [r14 + rbx], 46
and rax, 0b1111111111111 # instrumentation
movzx bx, byte ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi
and rcx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rbx], 2
add cl, 125 # instrumentation
and rax, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rdx]
add rsi, 80
and rdx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdx], cl
mov rdx, 0b1111111111000 #
and rbx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rbx]
mov al, dl
and rdx, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
add word ptr [r14 + rsi], 73
and rdx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
setnbe byte ptr [r14 + rdi]
and rdx, 0b1111111111000 # instrumentation
add qword ptr [r14 + rdx], -76
and rsi, 0b1111111111000 # instrumentation
inc word ptr [r14 + rsi]
and rax, 0b1111111111000 # instrumentation
cmpxchg dword ptr [r14 + rax], edi
and rcx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rcx], esi
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
