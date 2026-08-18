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
and cl, bl
and rbx, 0b1111111111111 # instrumentation
cmovp ecx, dword ptr [r14 + rbx]
lea di, qword ptr [rdi + rax]
and rcx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rcx]
inc rdi
cmovp edi, edi
sbb dl, -125
xadd rcx, rbx
or cl, 18
cmovnbe eax, edi
adc al, dil
or cx, 1 # instrumentation
and dx, cx # instrumentation
shr dx, 1 # instrumentation
div cx
add al, 6 # instrumentation
setp bl
imul eax, ecx
and rdx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rdx], rax
cmp dil, sil
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rbx, 0b1111111111111 # instrumentation
imul rax, qword ptr [r14 + rbx]
lea rbx, qword ptr [rcx + rcx + 64472]
adc ax, 126
cmovnbe di, si
nop
cmovns rsi, rbx
and rax, 0b1111111111111 # instrumentation
or qword ptr [r14 + rax], rdi
and rbx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rbx], 103
setle bl
or cl, al
and rdx, 0b1111111111111 # instrumentation
add word ptr [r14 + rdx], -34
adc edi, esi
and eax, 1409503164
and rbx, 0b1111111111111 # instrumentation
cmovno esi, dword ptr [r14 + rbx]
imul dl
add dl, -57 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rdx]
movzx cx, dl
and rcx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rcx], bl
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsf rdi, qword ptr [r14 + rdi]
add al, -8 # instrumentation
lea ebx, qword ptr [rdi + rsi + 10620]
setnle al
inc rcx
and rbx, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rbx], 7
and rcx, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rcx]
or si, 0b1000000000000000 # instrumentation
bsf bx, si
add al, -25 # instrumentation
setb cl
and rdx, 0b1111111111111 # instrumentation
and ebx, 0b111 # instrumentation
bts dword ptr [r14 + rdx], ebx
and rax, 0b1111111111111 # instrumentation
and sil, byte ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovle ax, word ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
sub ebx, dword ptr [r14 + rbx]
movzx esi, cl
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
not ebx
and rsi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rsi], 3
btc si, 148
and rbx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rbx], dl
btr edx, 147
xor dil, 66
btc bx, ax
add di, dx
btc edi, edi
and rdx, 0b1111111111111 # instrumentation
not dword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], eax
or dil, 1 # instrumentation
mov ax, 1 # instrumentation
div dil
add al, 86 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovno rsi, qword ptr [r14 + rax]
cmovnz rbx, rcx
lea rdx, qword ptr [rdx]
cmovle rbx, rcx
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea esi, qword ptr [rdi + rcx + 34989]
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx
and ax, 22216
sub cl, dl
inc dx
cmovns bx, di
dec edx
lea di, qword ptr [rcx]
and rdi, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rdi]
cmovno ecx, ebx
xor dil, -43
adc cl, -12
and rbx, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rbx], 7
imul al
cmovnb ax, si
sbb rax, 523457562
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], -54
adc bl, bl
imul rcx
and rcx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rcx], -51
add rdx, 13
btr bx, 242
and rax, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rax], -75
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi
and rdi, 0b1111111111111 # instrumentation
imul dx, word ptr [r14 + rdi]
add al, -59 # instrumentation
cmovnz rsi, rdi
and rax, 0b1111111111111 # instrumentation
and di, word ptr [r14 + rax]
movsx ebx, ax
and rcx, 0b1111111111111 # instrumentation
or edi, dword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
cmovs cx, word ptr [r14 + rdx]
add al, -6
and rdx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdx], 396314021
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
