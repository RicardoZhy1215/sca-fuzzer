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
and rdx, 0b1111111111111 # instrumentation
cmp bx, word ptr [r14 + rdx]
adc rsi, rdi
lea ax, qword ptr [rcx]
and bl, dl
neg ecx
or si, dx
cmovns rbx, rdx
test eax, -684758606
xor cx, ax
and rsi, 0b1111111111000 # instrumentation
and ebx, 0b111 # instrumentation
lock bts dword ptr [r14 + rsi], ebx
or si, 58
and rax, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rax], dx
or dil, 47
and eax, 67
sbb dl, cl
and rax, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rax], -53
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
neg al
and rbx, 0b1111111111000 # instrumentation
lock xadd byte ptr [r14 + rbx], bl
and rax, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rax], rbx
and rax, 0b1111111111111 # instrumentation
sub edx, dword ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock not qword ptr [r14 + rbx]
sbb ax, -668
imul bl
imul rbx, rbx, -85
sub rbx, rcx
lea edx, qword ptr [rbx + rax]
cmpxchg bl, al
inc rbx
and rbx, 0b1111111111111 # instrumentation
cmovbe ebx, dword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rdi]
btr rdi, 58
bts rcx, rdi
btr rbx, rax
add bl, 48 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovnbe rdx, qword ptr [r14 + rbx]
movzx rcx, ax
and rdi, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
or sil, byte ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rax]
lea si, qword ptr [rbx]
adc dl, 48
add al, al
and rcx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rcx], edx
test esi, 193811659
and rcx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rcx], -88
and rcx, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
and si, 0b111 # instrumentation
lock btr word ptr [r14 + rax], si
add al, 62 # instrumentation
setno dl
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
cmpxchg esi, ebx
cmovbe si, ax
and rbx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rbx], -882651035
and rdx, 0b1111111111111 # instrumentation
cmovnp bx, word ptr [r14 + rdx]
setb bl
and rdx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdx], esi
bt eax, 80
test sil, -90
lea rdi, qword ptr [rcx + rcx + 31589]
cmovp rbx, rbx
lea rcx, qword ptr [rsi + rax + 37966]
cmovz rax, rcx
cmc
or edi, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv edi
cmp edi, edx
or bl, 0b1000 # instrumentation
and bl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv bl
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdi, 0b1111111111111 # instrumentation
and edi, dword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
cmovb ebx, dword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
test dword ptr [r14 + rsi], -518246797
xor esi, esi
and rcx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
sub dl, byte ptr [r14 + rcx]
xor ecx, -76
btc rsi, rbx
add cl, 41 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovno di, word ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 1 # instrumentation
and edx, dword ptr [r14 + rbx] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rbx]
add cl, -11 # instrumentation
adc dl, al
and rdi, 0b1111111111111 # instrumentation
and rcx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovo esi, dword ptr [r14 + rcx]
lea ax, qword ptr [rcx + rdi + 33718]
and rdx, 0b1111111111111 # instrumentation
cmovb cx, word ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
movsx rdx, word ptr [r14 + rbx]
cmovo rbx, rbx
and rsi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rsi]
and rdx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdx], 126
and rcx, 0b1111111111111 # instrumentation
cmovle ecx, dword ptr [r14 + rcx]
cmovnle ebx, eax
xor eax, 26
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax]
lea eax, qword ptr [rax + rcx]
xor ax, 15591
and rcx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rcx], rdx
and rsi, 0b1111111111111 # instrumentation
cmovb edx, dword ptr [r14 + rsi]
sbb cl, 22
and rax, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rax]
sub rdx, rbx
sets al
setnb al
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
