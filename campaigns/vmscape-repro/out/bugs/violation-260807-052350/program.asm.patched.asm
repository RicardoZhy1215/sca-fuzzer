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
add bl, 122 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovno eax, dword ptr [r14 + rbx]
or dl, dl
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
or ecx, ecx
bts esi, ecx
cmp ax, -13
and eax, edi
sbb dl, 106
cmovbe di, cx
lea rsi, qword ptr [rax + rcx + 41415]
or edx, 0b1000000000000000000000000000000 # instrumentation
bsr ecx, edx
xadd esi, ecx
and rdi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdi], sil
xadd dil, bl
sub rbx, rdi
cmovs rsi, rsi
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
movzx rax, dx
xor edx, -42
nop rsi
and rsi, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rsi], 109
and rax, 826683163
imul rbx
seto al
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], 81
btc ebx, 97
bts ecx, edi
lea ax, qword ptr [rax + rdx + 3655]
lea rsi, qword ptr [rax + rsi]
and rcx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rcx], rax
and rdi, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
sub bl, byte ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
cmovl edi, dword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmovz rdx, qword ptr [r14 + rax]
and rdi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdi], 10
mul sil
or dil, -37
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsf ecx, dword ptr [r14 + rsi]
sub rdx, 7
and rbx, 0b1111111111111 # instrumentation
cmovnl ax, word ptr [r14 + rbx]
cmovnle rdx, rcx
and rbx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rbx], 66
lea esi, qword ptr [rbx + rdx]
and rbx, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
adc dl, byte ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rcx], 3
and rdx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdx], dl
xor cl, -79
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
add bl, 10 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovs ecx, esi
and rdi, 0b1111111111111 # instrumentation
and bl, byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovo cx, word ptr [r14 + rcx]
cmovl cx, dx
cmpxchg cl, al
setl dl
cmovnl ax, cx
sbb bl, al
xor cl, bl
and rdx, 0b1111111111111 # instrumentation
and esi, 0b111 # instrumentation
btr dword ptr [r14 + rdx], esi
and rdx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdx], al
cmovnb edx, edi
cmovno cx, ax
cmovnbe bx, ax
test eax, 695709497
and rsi, 0b1111111111111 # instrumentation
sub sil, byte ptr [r14 + rsi]
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add dl, -19 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovnle ax, word ptr [r14 + rbx]
and rdi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rdi], edx
xor dil, 38
lea rsi, qword ptr [rbx + rsi]
and rsi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rsi], sil
and rcx, 0b1111111111111 # instrumentation
cmovle edx, dword ptr [r14 + rcx]
sub dl, dl
and rsi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rsi], cl
btr rax, rcx
adc rax, -97
or dl, 1 # instrumentation
and rcx, 0b1111111111111 # instrumentation
or word ptr [r14 + rcx], 0b1000000000000000 # instrumentation
bsf dx, word ptr [r14 + rcx]
test dl, 7
btr rsi, 107
add bl, 8 # instrumentation
cmovnl ebx, esi
mov cx, dx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
cmp cl, 56
test dl, dil
and rdi, 0b1111111111111 # instrumentation
cmovp si, word ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rdx], dl
and dx, 104
and rcx, 0b1111111111111 # instrumentation
and word ptr [r14 + rcx], 34
xadd ax, cx
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], cl
or rsi, -81
and rdx, 0b1111111111111 # instrumentation
xor rdi, qword ptr [r14 + rdx]
add bl, cl
and rbx, 0b1111111111111 # instrumentation
xadd qword ptr [r14 + rbx], rdi
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsr rax, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
and bl, byte ptr [r14 + rcx]
lea rcx, qword ptr [rdi]
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
