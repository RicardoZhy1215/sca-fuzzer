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
movzx rax, sil 
and rsi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsf rbx, qword ptr [r14 + rsi] 
and dl, bl 
and rbx, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rbx] 
bt rax, 201 
and rcx, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rcx] 
bts edx, esi 
and rdi, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rdi], 2 
btr bx, si 
setz cl 
sub dil, 30 
cmp dl, -91 
add bl, 98 
lea si, qword ptr [rcx + rbx + 64884] 
adc esi, 125 
test al, 111 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, 44 # instrumentation
and rdi, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rdi], 1 
and rdx, 0b1111111111111 # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rsi] 
mov esi, ecx 
test al, bl 
or eax, 106692377 
and rcx, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rcx], dx 
sub al, -101 
bts rax, 119 
add cl, 115 # instrumentation
cmovle bx, ax 
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], 15 
and rsi, 0b1111111111111 # instrumentation
cmovz rcx, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx] 
cmpxchg bl, dl 
and rcx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rcx], edx 
and rsi, 0b1111111111111 # instrumentation
add word ptr [r14 + rsi], bx 
or edi, 0b1000000000000000000000000000000 # instrumentation
bsr ecx, edi 
add cl, -124 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rbx], cl 
and cl, al 
lea rdx, qword ptr [rsi + rcx] 
and rax, 0b1111111111111 # instrumentation
sbb dl, byte ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rax] 
adc cl, 22 
and rdi, 0b1111111111111 # instrumentation
and rsi, 0b111 # instrumentation
bts qword ptr [r14 + rdi], rsi 
and rdx, 0b1111111111111 # instrumentation
and eax, dword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rcx], ax 
clc  
not si 
or di, bx 
and rax, 0b1111111111111 # instrumentation
sub rcx, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
cmovnbe esi, dword ptr [r14 + rcx] 
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
add cl, 87 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovnbe rsi, rax 
and rdi, 0b1111111111111 # instrumentation
cmovnz edi, dword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rbx], dl 
xor dl, dl 
add dil, cl 
cmp rdx, rdi 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], 41 
and ax, di 
sahf  
lea esi, qword ptr [rbx + rdi + 45503] 
and rax, 0b1111111111111 # instrumentation
xor di, word ptr [r14 + rax] 
lea rdx, qword ptr [rsi] 
and rax, 0b1111111111111 # instrumentation
movsx rcx, word ptr [r14 + rax] 
adc bl, 65 
sbb ax, -27443 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
bts edx, 162 
and rdi, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bt word ptr [r14 + rdi], si 
and rax, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rax], 1 
and rsi, 0b1111111111111 # instrumentation
sbb edx, dword ptr [r14 + rsi] 
bts rcx, 107 
add bl, 103 # instrumentation
and rdx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rdx] 
or cl, 69 
sbb al, bl 
mov edx, -801702932 
test bl, bl 
lea di, qword ptr [rdi + rax + 56563] 
and rsi, 0b1111111111111 # instrumentation
lea rdx, qword ptr [rdx + rcx] 
and al, bl 
and rdi, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rdi] 
cmovnle edi, ecx 
stc  
lea esi, qword ptr [rsi] 
and rdi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdi], dl 
and rbx, 0b1111111111111 # instrumentation
xor rbx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
btr word ptr [r14 + rsi], bx 
lea rbx, qword ptr [rbx + rsi + 514] 
and rbx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rbx], 6 
adc al, 12 
and rdx, 0b1111111111111 # instrumentation
imul qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
movzx esi, word ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rbx] 
clc  
sub dil, -17 
lea bx, qword ptr [rdi + rbx] 
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], dil 
add ecx, -51 
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
