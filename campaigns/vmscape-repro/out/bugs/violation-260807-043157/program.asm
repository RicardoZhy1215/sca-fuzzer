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
bt rsi, 68 
xor cl, dl 
or rax, -593471372 
and rdi, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rdi] 
sbb dx, -31 
and rbx, 0b1111111111111 # instrumentation
cmovnb rsi, qword ptr [r14 + rbx] 
add dl, cl 
xor bl, al 
lea esi, qword ptr [rbx + rdi] 
and rbx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rbx], 5 
and rdx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdx], 92 
add al, al 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rax] 
cmovnle edi, edi 
add dl, 110 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rbx, 0b1111111111111 # instrumentation
bt qword ptr [r14 + rbx], 1 
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], cl 
and rdi, 0b1111111111111 # instrumentation
cmovnle di, word ptr [r14 + rdi] 
add si, cx 
movzx ebx, dl 
and rax, 0b1111111111111 # instrumentation
cmovnp edx, dword ptr [r14 + rax] 
sbb al, al 
adc al, -76 
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 0b1000 # instrumentation
and byte ptr [r14 + rbx], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi 
xadd rbx, rcx 
and rdi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rdi] 
setnz bl 
and rcx, 0b1111111111111 # instrumentation
cmovnp bx, word ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rdx], 6 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rax] 
add al, -40 # instrumentation
adc sil, -63 
bts ax, bx 
and rax, 0b1111111111111 # instrumentation
cmovnbe rbx, qword ptr [r14 + rax] 
or si, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv si 
add bl, 8 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rbx], edx 
and rcx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdi], bl 
setbe al 
and rax, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rax], 1 
add dl, -65 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rbx], bl 
bts dx, 26 
and rcx, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rcx], dx 
neg rdx 
and rdx, 0b1111111111111 # instrumentation
and bx, word ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
cmovns si, word ptr [r14 + rbx] 
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
lea rsi, qword ptr [rbx + rdi + 47911] 
sub cl, -25 
and rcx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rcx], al 
cmovno eax, esi 
adc sil, dl 
and rax, 0b1111111111111 # instrumentation
add word ptr [r14 + rax], -97 
cmovnle di, cx 
or al, -102 
inc ax 
add ax, -1091 
adc cl, cl 
add sil, bl 
and rax, 0b1111111111111 # instrumentation
or word ptr [r14 + rax], 1 # instrumentation
and dx, word ptr [r14 + rax] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rax] 
add bl, 3 # instrumentation
and rsi, 0b1111111111111 # instrumentation
mov di, word ptr [r14 + rsi] 
cmovo bx, si 
xor ecx, 9 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdi, 0b1111111111111 # instrumentation
or rbx, qword ptr [r14 + rdi] 
xor dl, 115 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], 7 
and rsi, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rsi], 1 
and rsi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rsi], bl 
xadd ebx, edx 
mul bx 
add bl, -106 # instrumentation
lea si, qword ptr [rdx] 
setp sil 
test cl, -84 
and rdi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rdi], rdi 
cmovno rax, rbx 
and rsi, 0b1111111111111 # instrumentation
adc edx, dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mov dx, word ptr [r14 + rdi] 
xor al, -68 
and rax, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rax] 
or cl, 82 
xor bl, dil 
and rdx, 0b1111111111111 # instrumentation
cmovnz rcx, qword ptr [r14 + rdx] 
lea eax, qword ptr [rax] 
xor dl, 92 
and rax, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rax], -111 
and rcx, 0b1111111111111 # instrumentation
cmovbe eax, dword ptr [r14 + rcx] 
cmp ax, -26073 
test eax, 1086344910 
imul rdi, rax, -83 
add al, -122 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovp ecx, dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
add sil, byte ptr [r14 + rax] 
sbb di, 44 
add rax, 1672024653 
sbb dil, 101 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], ax 
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
