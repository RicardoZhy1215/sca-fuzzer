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
test si, bx
or cl, cl
clc
and rbx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rbx], eax
and rsi, 0b1111111111111 # instrumentation
cmovnle edi, dword ptr [r14 + rsi]
setle bl
bt rcx, 9
cwd
adc rdi, -6
adc rbx, rsi
or cl, bl
and rax, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rax]
lea ecx, qword ptr [rax + rsi + 14119]
and rdi, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rdi], 4
sub cl, 113
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 70
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
sub bl, al
cmovl rax, rdi
cmp rdx, -93
not cl
and rcx, 0b1111111111111 # instrumentation
cmovnz rcx, qword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], di
cwd
and rdx, 0b1111111111111 # instrumentation
imul bx, word ptr [r14 + rdx], -99
mov edi, -649398574
or al, al
add cl, bl
or al, -62
cmp dil, 101
and rax, 0b1111111111111 # instrumentation
not word ptr [r14 + rax]
setbe cl
and rbx, 0b1111111111111 # instrumentation
adc bx, word ptr [r14 + rbx]
imul dl
test al, -13
and rdi, 0b1111111111111 # instrumentation
cmovb bx, word ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
and ecx, 0b111 # instrumentation
btc dword ptr [r14 + rcx], ecx
inc rax
cwd
and rbx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rbx], 18
and rax, 0b1111111111111 # instrumentation
imul word ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsf ecx, dword ptr [r14 + rdx]
add bl, -90 # instrumentation
sbb cl, -49
cmovnl esi, edx
and rdi, 0b1111111111111 # instrumentation
not qword ptr [r14 + rdi]
xor ax, -24419
and rdx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdx], cl
lea ax, qword ptr [rax]
and rdi, 0b1111111111111 # instrumentation
cmovnbe cx, word ptr [r14 + rdi]
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
or cl, al
and rdi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdi], dl
and eax, eax
lea rdx, qword ptr [rbx + rax]
cmp al, dil
lea eax, qword ptr [rdi]
imul eax
add al, 118 # instrumentation
cmovl rdi, rax
movsx di, dl
btc rbx, 180
sbb eax, -499260408
dec al
or al, 61
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 828490101
cmovs esi, ecx
sbb dl, dl
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add rax, -131214649
setnb dl
and rbx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rbx]
test edx, -1862009087
xor cl, al
lea rdi, qword ptr [rsi + rdx + 59982]
movzx esi, si
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsf eax, dword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rcx], edi
and rbx, 0b1111111111111 # instrumentation
xor cx, word ptr [r14 + rbx]
or ebx, 1 # instrumentation
and edx, ebx # instrumentation
shr edx, 1 # instrumentation
div ebx
add bl, -26 # instrumentation
and rsi, 0b1111111111111 # instrumentation
adc si, word ptr [r14 + rsi]
xor si, 115
adc ax, -6514
setbe al
and rdi, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
btc word ptr [r14 + rdi], bx
xor dl, -52
and rdx, 0b1111111111111 # instrumentation
cmovp di, word ptr [r14 + rdx]
add al, bl
and rsi, 0b1111111111111 # instrumentation
cmovp si, word ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], -29
and rbx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rbx], al
stc
and rdi, 0b1111111111111 # instrumentation
cmovp ax, word ptr [r14 + rdi]
cmp al, dl
setz cl
btr rdi, rax
lea rdi, qword ptr [rdi + rdi + 63717]
and rax, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rax], di
cmovle ax, si
cmp rsi, rax
xor ax, -12054
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
