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
and rsi, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rsi], al
or eax, edi
and rdi, 0b1111111111111 # instrumentation
add di, word ptr [r14 + rdi]
lea edx, qword ptr [rdi + rcx]
setnb dl
cmovnl esi, ecx
lea rdx, qword ptr [rsi]
cmovp eax, edx
and rcx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rcx]
lea rax, qword ptr [rdx]
imul si, di
stc
add edx, edi
and rbx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rbx]
add al, -91 # instrumentation
sets al
and rsi, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rsi], 4
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111111 # instrumentation
or cx, word ptr [r14 + rax]
setz al
lea rbx, qword ptr [rsi]
adc dl, 113
and rdx, 0b1111111111111 # instrumentation
cmovnbe rax, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdi], 110
setp dil
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], dl
and rdx, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rdi]
movsx di, sil
nop ebx
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 0b1000000000000000 # instrumentation
bsr ax, word ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rcx], 2
add rcx, 42
and rcx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rcx], 51
add eax, 2104623598
setnb al
sbb cl, bl
or dl, 1 # instrumentation
add bl, -43 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rdi], ax
and rax, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rax], -18
mov di, -5313
cwd
and rdx, 0b1111111111111 # instrumentation
cmovnle ebx, dword ptr [r14 + rdx]
btr si, 48
bt edi, ebx
and rbx, 0b1111111111111 # instrumentation
and rax, qword ptr [r14 + rbx]
xchg rdi, rdi
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 0b1000 # instrumentation
and byte ptr [r14 + rbx], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rbx]
mov dil, 30
and rbx, 0b1111111111111 # instrumentation
imul ebx, dword ptr [r14 + rbx], 95
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
and rdx, 0b1111111111111 # instrumentation
and edx, dword ptr [r14 + rdx]
movzx di, bl
xor cl, -74
sub dl, cl
add al, 61
and rcx, 0b1111111111111 # instrumentation
imul si, word ptr [r14 + rcx], 29
add cl, -49 # instrumentation
setnp dl
lea rbx, qword ptr [rbx + rcx + 48030]
and dx, 124
lea esi, qword ptr [rcx]
xor sil, 79
xor si, -30
or bl, 97
setnz cl
setbe sil
xor dl, 7
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], 7
add rbx, 33
and rsi, 0b1111111111111 # instrumentation
movsx rsi, word ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdi], eax
cmovno rbx, rcx
and rax, 0b1111111111111 # instrumentation
xor di, word ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rdx]
sub dl, al
sub dx, 59
bswap esi
and rsi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rsi], ax
or dl, -13
adc al, bl
adc cl, cl
and rax, 0b1111111111111 # instrumentation
xor esi, dword ptr [r14 + rax]
cmovo rdi, rcx
cmovnbe edx, edi
nop ecx
or al, cl
or esi, 0b1000000000000000000000000000000 # instrumentation
bsr ebx, esi
add bl, -80 # instrumentation
lea di, qword ptr [rsi + rax + 54358]
and rax, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rax], -95
and rdi, 0b1111111111111 # instrumentation
movzx edi, byte ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bt qword ptr [r14 + rcx], rsi
and rdi, 0b1111111111111 # instrumentation
test word ptr [r14 + rdi], si
and rbx, 0b1111111111111 # instrumentation
setp byte ptr [r14 + rbx]
sub eax, -1087438711
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], sil
and rdx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
cmovns cx, word ptr [r14 + rcx]
or di, 1 # instrumentation
and dx, di # instrumentation
shr dx, 1 # instrumentation
div di
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
