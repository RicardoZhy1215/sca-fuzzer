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
sub eax, eax 
btr eax, ebx 
and rdi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rdi], edx 
adc edx, ebx 
sbb dl, -73 
test bl, al 
clc  
and rsi, 0b1111111111111 # instrumentation
mov esi, dword ptr [r14 + rsi] 
or eax, -909368706 
lea rax, qword ptr [rbx] 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 1 # instrumentation
and edx, dword ptr [r14 + rcx] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rcx] 
lea bx, qword ptr [rdx] 
xchg ebx, edx 
and rdx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rdx], ebx 
add dl, -46 
mul sil 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add rax, 48 
and rdi, 0b1111111111111 # instrumentation
cmovz si, word ptr [r14 + rdi] 
and rbx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rbx], -107 
and rdx, 0b1111111111111 # instrumentation
and bx, 0b111 # instrumentation
btc word ptr [r14 + rdx], bx 
add dl, -95 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], -41 
and rsi, 0b1111111111111 # instrumentation
cmovb rdx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
neg word ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
or al, dl 
test rdi, rax 
and rbx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rbx], di 
and rcx, 0b1111111111000 # instrumentation
lock dec byte ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
cmovns di, word ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
and dl, byte ptr [r14 + rsi] 
inc edx 
sub ebx, esi 
and rbx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rbx], cl 
cmp rax, 1225052350 
lea bx, qword ptr [rsi + rcx] 
or rdi, -70 
and rax, 0b1111111111111 # instrumentation
movsx bx, byte ptr [r14 + rax] 
or esi, 74 
sub bl, dl 
and rcx, 0b1111111111111 # instrumentation
cmovnz rax, qword ptr [r14 + rcx] 
setnz cl 
and rsi, 0b1111111111111 # instrumentation
cmovns ecx, dword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rdi], al 
inc rdx 
cmp bl, bl 
cmp sil, 104 
and rcx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rcx], -71 
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
add bl, -111 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rcx, 0b1111111111111 # instrumentation
cmovnz esi, dword ptr [r14 + rcx] 
lea ax, qword ptr [rdi] 
and rax, 328856057 
or eax, -1569744042 
setl al 
cmpxchg eax, esi 
test rbx, rcx 
add al, -26 
and rbx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rbx], eax 
adc al, cl 
test dx, -12412 
and rcx, 0b1111111111111 # instrumentation
or rbx, qword ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rbx 
nop  
btc di, 162 
and rax, 0b1111111111111 # instrumentation
cmovb eax, dword ptr [r14 + rax] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
movsx rsi, di 
and rsi, 0b1111111111111 # instrumentation
mul word ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], bl 
cbw  
and rbx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rbx], cl 
and rdi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rdi], -65 
xor cl, dl 
and rbx, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rbx] 
lea eax, qword ptr [rsi + rax] 
and rcx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rcx], edx 
sub bl, bl 
movsx rsi, ax 
and rax, 0b1111111111111 # instrumentation
cmovnbe eax, dword ptr [r14 + rax] 
neg rdi 
adc eax, 545073106 
and rdi, 0b1111111111111 # instrumentation
cmovno edi, dword ptr [r14 + rdi] 
or rax, -397407126 
and eax, 2 
sub bl, al 
and rbx, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rbx] 
lea si, qword ptr [rdi + rbx + 59345] 
and bx, si 
and rbx, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rbx] 
test sil, 26 
and rcx, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mov dl, byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rsi] 
xadd ax, dx 
add al, 51 
and rsi, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
or word ptr [r14 + rdx], 1 # instrumentation
add bl, 77 # instrumentation
setnl cl 
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
