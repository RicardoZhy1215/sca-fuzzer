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
add bl, -2 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdi] 
test bl, bl 
movzx bx, dl 
xor bl, bl 
and di, cx 
adc ax, 1484 
imul rdi, rax, 98 
add di, 112 
sbb eax, 179934110 
not bl 
adc dl, al 
and rdi, 0b1111111111111 # instrumentation
cmpxchg word ptr [r14 + rdi], ax 
cmp dl, bl 
and rdx, 0b1111111111111 # instrumentation
not dword ptr [r14 + rdx] 
and dl, 0 
mul rcx 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
sub bl, dl 
and rbx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rbx], 21 
mov ecx, edx 
bt si, si 
add al, 42 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovo rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmovno rdi, qword ptr [r14 + rax] 
adc al, 118 
bts edi, 130 
add cl, -68 # instrumentation
setle dl 
sub dil, -29 
mov ebx, eax 
sub rdx, 80 
and rsi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rsi], dx 
adc bx, si 
lea si, qword ptr [rsi + rax + 35525] 
and rsi, 0b1111111111111 # instrumentation
adc rcx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax 
and rax, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rax], 90 
and rax, 0b1111111111111 # instrumentation
and byte ptr [r14 + rax], bl 
and rdx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdx], ebx 
neg rdx 
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], 118 
not dl 
and rdx, 0b1111111111000 # instrumentation
and rbx, 0b111 # instrumentation
lock btr qword ptr [r14 + rdx], rbx 
inc di 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], 7 
and rbx, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rbx], 50 
and rsi, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rsi], 37 
and rax, 0b1111111111111 # instrumentation
cmovnle eax, dword ptr [r14 + rax] 
neg rbx 
cmovl edi, edx 
test eax, 744152300 
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
cmp cx, word ptr [r14 + rdx] 
xor ax, 16724 
cmovnl edx, ecx 
sbb al, -23 
and rbx, 0b1111111111111 # instrumentation
cmovnb di, word ptr [r14 + rbx] 
add cl, bl 
lea ebx, qword ptr [rdx + rsi + 16674] 
and rdi, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdi], edx 
inc eax 
sub dil, -16 
setns cl 
xchg ebx, edx 
cmpxchg rax, rcx 
sets bl 
cmp rbx, rbx 
lea rdi, qword ptr [rcx + rdi + 2778] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdi, 0b1111111111000 # instrumentation
and cx, 0b111 # instrumentation
lock btr word ptr [r14 + rdi], cx 
nop rdi 
imul rax, rdx 
add bl, 74 # instrumentation
mov di, -7715 
cmovo di, dx 
setz dl 
and rcx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rcx], 6 
add cl, -65 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnl bx, word ptr [r14 + rcx] 
xor cl, cl 
or dl, 33 
and dx, dx 
cmovo esi, esi 
btc ebx, 159 
cmovnbe si, cx 
or eax, -128942472 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rsi 
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], sil 
and rdx, 0b1111111111111 # instrumentation
sbb dl, byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
dec word ptr [r14 + rsi] 
or bl, -74 
adc cl, 12 
xor al, bl 
and rcx, 0b1111111111111 # instrumentation
sub rsi, qword ptr [r14 + rcx] 
cmovbe rbx, rsi 
and dl, al 
and rdi, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rdi], 40 
and rax, 0b1111111111111 # instrumentation
sbb bx, word ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rax], al 
and rdx, 0b1111111111111 # instrumentation
xor rsi, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 1 # instrumentation
and dx, word ptr [r14 + rdi] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], -14 
sbb rdi, 28 
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
