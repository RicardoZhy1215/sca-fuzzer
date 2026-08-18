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
xor dil, 79
cmpxchg bx, ax
and rsi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rsi], 1701201268
adc al, -42
lea ecx, qword ptr [rsi + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovnb esi, dword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
neg word ptr [r14 + rcx]
cmovo ebx, eax
xadd bx, si
mul cx
btc rsi, 71
bts rdx, rcx
sub eax, 636495634
and rdx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdx], 104
xor rcx, rdi
adc ax, 3513
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add bl, 18 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sbb cx, word ptr [r14 + rsi]
test al, 114
movzx di, cl
sbb di, bx
and rcx, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
bts word ptr [r14 + rcx], dx
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], edi
or al, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv al
nop edx
sub eax, -755157190
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
add dl, 69 # instrumentation
lea rcx, qword ptr [rcx + rdi + 24284]
and rdx, 0b1111111111111 # instrumentation
cmovl ebx, dword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
and al, byte ptr [r14 + rbx]
setle bl
sets cl
setnl dl
setno dil
cmp al, dl
and rdx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rdx], -47
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rdi
xor rbx, 7
adc al, sil
cmpxchg cl, bl
lea rax, qword ptr [rsi + rdx]
inc cl
adc bl, cl
and rdi, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rdi]
lea dx, qword ptr [rbx + rbx + 43263]
and rcx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rcx], 103
and rax, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rax], eax
and rsi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rsi], eax
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
add bl, 102 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rcx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rcx]
cmovp si, bx
cbw
sbb al, cl
mov sil, sil
and rsi, 0b1111111111111 # instrumentation
cmp di, word ptr [r14 + rsi]
cmovnbe di, si
lea edx, qword ptr [rsi + rsi + 1995]
cmovns cx, di
movzx si, bl
bt rdi, rdi
mov ebx, edx
cmpxchg dl, cl
mov edx, -1874141179
cmovbe bx, dx
sbb dl, -86
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], 6
cwd
sub cl, -69
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rbx
lea bx, qword ptr [rcx]
cmovo esi, ebx
and rbx, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
bt word ptr [r14 + rbx], cx
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], -43
adc al, dl
mov dl, bl
and rdi, 0b1111111111111 # instrumentation
cmovnbe bx, word ptr [r14 + rdi]
and rdi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
cmovns ax, word ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rbx], ecx
cmovb rdx, rcx
lea rdx, qword ptr [rsi]
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], dx
adc ecx, eax
lea bx, qword ptr [rcx + rdx]
and rdi, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rdi], cx
and rbx, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rbx]
xchg ax, dx
clc
and rbx, 0b1111111111111 # instrumentation
add ax, word ptr [r14 + rbx]
cmp dl, al
xchg bx, di
and rbx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rbx], -32
sbb rax, 86
bt ecx, edx
and rdx, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], ecx
cmp al, 103
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
