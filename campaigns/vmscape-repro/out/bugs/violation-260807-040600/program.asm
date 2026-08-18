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
add bl, -81 # instrumentation
lea rbx, qword ptr [rbx + rbx + 28210] 
bswap rsi 
sbb dl, 37 
lea dx, qword ptr [rdi] 
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], bl 
and rsi, 0b1111111111000 # instrumentation
and edx, 0b111 # instrumentation
lock btr dword ptr [r14 + rsi], edx 
lea rax, qword ptr [rcx + rbx] 
mov dl, al 
and bl, bl 
xor cl, dl 
mov cl, 59 
lea edx, qword ptr [rdi] 
dec di 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], -116 
sbb bl, -96 
and rbx, 0b1111111111111 # instrumentation
cmovb esi, dword ptr [r14 + rbx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
neg bl 
movsx di, sil 
test rax, rdx 
and rax, 0b1111111111111 # instrumentation
setbe byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
btc word ptr [r14 + rsi], dx 
lea dx, qword ptr [rax] 
and rdi, 0b1111111111111 # instrumentation
and dil, byte ptr [r14 + rdi] 
add al, 100 # instrumentation
lea rdx, qword ptr [rsi + rax] 
lahf  
and rbx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rbx], rax 
cmovp rcx, rax 
and rax, 0b1111111111111 # instrumentation
or esi, dword ptr [r14 + rax] 
add ax, 31208 
and rbx, 0b1111111111111 # instrumentation
movsx ebx, word ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rbx], 58 
and rsi, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rsi], 101 
or edx, 0b1000000000000000000000000000000 # instrumentation
bsf ebx, edx 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], -1 
and rcx, 0b1111111111111 # instrumentation
adc eax, dword ptr [r14 + rcx] 
sbb bl, cl 
and rcx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rcx], -52 
xchg rsi, rax 
xadd cx, di 
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rcx 
xor rax, rsi 
and rsi, 0b1111111111111 # instrumentation
and qword ptr [r14 + rsi], -119 
and rax, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
movzx ax, byte ptr [r14 + rsi] 
and rax, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rax], bl 
and rdx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
test dword ptr [r14 + rax], -897758109 
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
lea ax, qword ptr [rdi + rdx] 
and rsi, 0b1111111111111 # instrumentation
test word ptr [r14 + rsi], bx 
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], 87 
setle bl 
and al, al 
lea edi, qword ptr [rsi + rdi + 16405] 
btc ax, 95 
sbb edi, -65 
sub cl, dl 
sbb si, -122 
cmovl rbx, rbx 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], 46 
sbb rdi, rdi 
and rdx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rdx], ecx 
movsx edi, bx 
inc bx 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea dx, qword ptr [rdx + rdx] 
test eax, esi 
xor eax, 92 
imul ebx, ecx 
and rax, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rdi], cx 
lea ax, qword ptr [rdx + rcx + 20867] 
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], -116 
and rcx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rcx], 62 
and rcx, 0b1111111111111 # instrumentation
sub dil, byte ptr [r14 + rcx] 
btc rdx, 89 
btc cx, 220 
cmpxchg edx, edi 
test eax, -279197900 
dec ax 
cmovnle rdx, rdi 
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], 71 
and rdi, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rdi] 
or bx, 0b1000000000000000 # instrumentation
bsr cx, bx 
add dl, 127 # instrumentation
lea cx, qword ptr [rdi + rcx] 
and rsi, 0b1111111111111 # instrumentation
cmovle ecx, dword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rcx], dl 
sets al 
or dl, -78 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rbx 
cmovz ax, cx 
and eax, ecx 
imul eax, esi, 65 
add al, 117 # instrumentation
cmovnbe rax, rdi 
and rax, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rax] 
xadd ax, dx 
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx] 
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
