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
.macro.set_h2g_target.vm.function_vm_train: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]
.exit_main_1:
.function_main_2:
.bb_main_2.entry:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.set_h2g_target.vmvictim.function_vmvictim_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.switch_h2g.vmvictim.2: nop qword ptr [rax + 0xff]
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_measure: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_4: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.3: nop qword ptr [rax + 0xff]
.exit_main_3:
.function_main_4:
.bb_main_4.entry:
.macro.landing_g2h.main_4: nop qword ptr [rax + 0xff]
nop  
.exit_main_4:
.section .data.vmvictim
.function_vmvictim_1:
.bb_vmvictim_1.entry:
.macro.landing_h2g.vmvictim_1: nop qword ptr [rax + 0xff]
add al, -30 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setnp dl 
cmovnbe cx, si 
cmovl cx, ax 
cmovnb rsi, rsi 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], bl 
cmovl rax, rbx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], -1683144711 
bts rax, rbx 
cmp esi, -44 
lea cx, qword ptr [rdx] 
cmovbe ax, ax 
cmp dil, -119 
not dil 
and rdi, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdi] 
not ecx 
sub cl, dl 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
add cl, -101 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rdi], 65 
cmpxchg bl, cl 
adc eax, 576502350 
test eax, -291267920 
add dx, -82 
imul rax, rdi 
cld  
sub rsi, 125 
add cl, bl 
and rax, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rax], rdi 
and rcx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rcx], 40 
and rdi, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdi], dil 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], -89 
and rax, 0b1111111111111 # instrumentation
adc rbx, qword ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rax], 0 
and rdx, 0b1111111111111 # instrumentation
and rbx, 0b111 # instrumentation
btc qword ptr [r14 + rdx], rbx 
and rax, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rax], rdi 
and rcx, 0b1111111111111 # instrumentation
add dword ptr [r14 + rcx], edx 
cmovnbe edx, esi 
and dl, 101 
and rsi, 0b1111111111111 # instrumentation
sub dil, byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rdx], rax 
and rdx, 0b1111111111111 # instrumentation
cmovno ecx, dword ptr [r14 + rdx] 
xadd al, al 
or al, 11 
and rsi, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rsi], bx 
imul rdx, rdx, 43 
or esi, 1 # instrumentation
and edx, esi # instrumentation
shr edx, 1 # instrumentation
div esi 
and rdi, 0b1111111111111 # instrumentation
and eax, dword ptr [r14 + rdi] 
movsx bx, sil 
or dl, 59 
lea bx, qword ptr [rcx] 
lfence  # noremove
.macro.switch_g2h.main.vmvictim_2: nop qword ptr [rax + 0xff]
lfence  
.exit_vmvictim_3:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start.vm_0: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_0:
.function_vm_train:
.bb_vm_train.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
and al, al 
or al, bl 
lea rdx, qword ptr [rcx + rdi + 37436] 
add bl, bl 
not dl 
btr esi, 53 
and rcx, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rcx], -89 
mov cl, cl 
cmovnz rax, rax 
cmp rbx, -30 
xadd rdx, rdi 
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl 
xchg rsi, rax 
cmp cl, cl 
sub dil, -57 
cmovnl rdi, rbx 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
add dl, 104 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], eax 
and rbx, 0b1111111111111 # instrumentation
cmovs edi, dword ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rcx], 115 
and rdi, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rdi], ax 
setp al 
and rsi, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rsi], 118 
and rcx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rcx] 
sbb rax, 868338120 
cmc  
adc dl, 59 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rax], 7 
mov dx, cx 
and rdi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdi], -21 
cmovs dx, dx 
and rdx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rdx] 
lfence  # noremove
.macro.switch_g2h.main.vm_1: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_3:
.function_vm_measure:
.bb_vm_measure.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_measure:
.section .data.main
.test_case_exit:nop
