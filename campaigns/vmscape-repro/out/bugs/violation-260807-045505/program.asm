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
xor edi, -87 
imul rsi, rbx, 59 
cmp rbx, rdx 
and rdx, 0b1111111111111 # instrumentation
xor bl, byte ptr [r14 + rdx] 
xor dl, -19 
adc rax, -1123566434 
and rdx, 0b1111111111111 # instrumentation
movsx ecx, byte ptr [r14 + rdx] 
setbe al 
cmp eax, -645092559 
and rdx, 0b1111111111111 # instrumentation
cmp ax, -26307 
xchg eax, ebx 
or bl, 0b1000 # instrumentation
and bl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv bl 
mul sil 
cmovb dx, dx 
lea edx, qword ptr [rcx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
cmp rax, -41807279 
adc cl, al 
and rbx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rbx], sil 
adc eax, -618128025 
and rbx, 0b1111111111111 # instrumentation
movzx ecx, byte ptr [r14 + rbx] 
lea rdx, qword ptr [rdx + rdx + 60464] 
mov bx, cx 
lea di, qword ptr [rdi + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmovs eax, dword ptr [r14 + rcx] 
add rcx, -40 
lea rax, qword ptr [rdx + rdi + 12429] 
and rdx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdx], dl 
and rsi, 0b1111111111111 # instrumentation
cmovz rax, qword ptr [r14 + rsi] 
xchg rcx, rdi 
not eax 
mul cl 
and rsi, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rsi], 1 
bts rbx, 192 
cmpxchg cx, bx 
test dl, dl 
cmovnb cx, si 
and rdx, 0b1111111111111 # instrumentation
adc cx, word ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rcx], dl 
and rdx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdx], 77 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], 16 
add ax, 14631 
and rbx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rbx], eax 
cmp cl, dl 
and rcx, 0b1111111111111 # instrumentation
cmovnle bx, word ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
movsx rsi, word ptr [r14 + rax] 
xor rax, -2143623134 
and rdi, 0b1111111111111 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi] 
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
imul ax, ax, -3 
cmpxchg al, bl 
setz cl 
and rdx, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rdx], 7 
not ax 
xadd dl, sil 
cmovnl rdx, rdi 
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
btr rdi, rdi 
or dl, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
and al, cl 
lea ebx, qword ptr [rax] 
and rax, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rax], rax 
or rsi, 0b1000000000000000000000000000000 # instrumentation
bsf rax, rsi 
and rcx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rcx], -86 
sub dl, -50 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add al, 8 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovl eax, dword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rcx], rsi 
and rdi, 0b1111111111111 # instrumentation
cmovnb rcx, qword ptr [r14 + rdi] 
sets cl 
and rcx, 0b1111111111111 # instrumentation
setz byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
cmovns edi, dword ptr [r14 + rdx] 
xor dx, di 
and rsi, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rsi], bl 
sbb dl, al 
xor dl, dl 
and rbx, 0b1111111111111 # instrumentation
sub dword ptr [r14 + rbx], 117 
cmp bl, 62 
and rbx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rbx], ecx 
and rax, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rax], -25 
and rsi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rsi], 789264404 
and rdi, 0b1111111111111 # instrumentation
cmovno rcx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi] 
movzx si, dil 
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], eax 
xor eax, 38348595 
and rdx, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsr rdi, qword ptr [r14 + rdx] 
add bl, -76 # instrumentation
setnbe cl 
and rcx, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmovnp esi, dword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rsi], bl 
cmovnle si, ax 
dec cl 
and rsi, 0b1111111111111 # instrumentation
mov cl, byte ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
and rbx, 0b111 # instrumentation
btc qword ptr [r14 + rsi], rbx 
neg ecx 
cmovz di, bx 
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
