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
nop esi
test bl, 48
lea rax, qword ptr [rdi]
and rax, 0b1111111111111 # instrumentation
bts word ptr [r14 + rax], 2
or bl, -112
cmovo dx, si
cmovb bx, si
setnbe sil
movsx rdi, dl
or sil, 68
dec sil
add al, 37
cmovz rcx, rcx
lea di, qword ptr [rdi + rsi]
or di, 1 # instrumentation
and dx, di # instrumentation
shr dx, 1 # instrumentation
div di
sub al, -35
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
sub sil, bl
and rcx, 0b1111111111111 # instrumentation
sbb dword ptr [r14 + rcx], 115
and rdi, 0b1111111111111 # instrumentation
add word ptr [r14 + rdi], bx
cmovo rsi, rax
lea rcx, qword ptr [rdx]
and rax, 0b1111111111111 # instrumentation
cmovnl ecx, dword ptr [r14 + rax]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
test di, ax
cmovnl rsi, rbx
adc ax, ax
and rdx, 0b1111111111111 # instrumentation
adc edi, dword ptr [r14 + rdx]
lea ax, qword ptr [rdi + rcx]
and rcx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmovno rcx, qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rax], 125
and rax, 0b1111111111111 # instrumentation
adc rsi, qword ptr [r14 + rax]
nop
and rsi, 0b1111111111111 # instrumentation
cmovb rdx, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmovno cx, word ptr [r14 + rdx]
xchg esi, edx
movsx rdx, cx
and rdx, 0b1111111111111 # instrumentation
cmovnl si, word ptr [r14 + rdx]
lea cx, qword ptr [rsi + rbx]
and rbx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rbx], eax
and rax, 0b1111111111111 # instrumentation
mul word ptr [r14 + rax]
lea ebx, qword ptr [rax + rax]
lea edx, qword ptr [rbx + rdx + 15451]
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rbx], dl
and rcx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rcx], -29
and rcx, 0b1111111111111 # instrumentation
and word ptr [r14 + rcx], di
sbb rsi, rsi
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx]
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
and rdi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdi], 50
and rax, 0b1111111111111 # instrumentation
sbb al, byte ptr [r14 + rax]
setz dl
not di
cmp dx, -123
cmovle rbx, rcx
lea rax, qword ptr [rax + rdi + 46682]
and rcx, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rcx]
and rdi, 0b1111111111111 # instrumentation
cmovz ecx, dword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
cmovl edi, dword ptr [r14 + rcx]
adc dx, ax
and rdi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rdi], 6
cmpxchg cl, dl
neg ecx
and rax, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
bt dword ptr [r14 + rax], edi
mov dl, -96
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
mov cl, al
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], bl
and rcx, 0b1111111111111 # instrumentation
cmovp rbx, qword ptr [r14 + rcx]
cmovnp rbx, rdx
and rbx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rbx]
lea bx, qword ptr [rbx + rdi + 51192]
and rcx, 0b1111111111111 # instrumentation
cmovnl ecx, dword ptr [r14 + rcx]
btr ecx, 246
add al, 35 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rax]
setns al
cmovno cx, bx
imul ecx
mov rdi, rax
dec cx
and rcx, 0b1111111111000 # instrumentation
lock bts dword ptr [r14 + rcx], 1
add cl, 20 # instrumentation
cmovs rbx, rdi
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], -123
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi
and si, cx
and rsi, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rsi], -109
and rax, 0b1111111111111 # instrumentation
and rbx, qword ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rcx], ecx
and rdi, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rdi], 48
imul rdx, rcx, 82
test rax, 1099400950
sbb rax, 115
and rsi, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rbx]
setns bl
and rax, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rax], dl
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], -64
cmc
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
