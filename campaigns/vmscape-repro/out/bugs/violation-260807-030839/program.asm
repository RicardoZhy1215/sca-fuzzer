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
and rbx, 0b1111111111111 # instrumentation
and si, word ptr [r14 + rbx] 
and ecx, ebx 
cmovbe cx, ax 
xchg rax, rax 
and rdi, 0b1111111111111 # instrumentation
imul word ptr [r14 + rdi] 
neg ebx 
test dl, -2 
and rbx, 0b1111111111000 # instrumentation
lock inc byte ptr [r14 + rbx] 
setno bl 
add al, -128 
and rsi, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rsi], ax 
lea edx, qword ptr [rdx + rdi + 25350] 
cmovle ax, dx 
and rdi, 0b1111111111111 # instrumentation
cmovle eax, dword ptr [r14 + rdi] 
or al, 52 
or cl, -58 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, -96 # instrumentation
adc al, -75 
and rbx, 0b1111111111111 # instrumentation
or bx, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], al 
and rdx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rdx], -44 
or rax, -894104346 
and rsi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rsi], -108 
xor bl, al 
and rcx, 0b1111111111111 # instrumentation
cmovs rsi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmovz rax, qword ptr [r14 + rdi] 
xchg rcx, rcx 
setl al 
lea dx, qword ptr [rdx + rdx + 30688] 
and rdx, 0b1111111111111 # instrumentation
cmovle edi, dword ptr [r14 + rdx] 
adc bl, bl 
and rax, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rax], 6 
add cl, bl 
lea bx, qword ptr [rcx + rdi] 
lea ax, qword ptr [rdi + rbx + 60259] 
cmc  
and rdx, 0b1111111111111 # instrumentation
cmovbe edx, dword ptr [r14 + rdx] 
not eax 
add dl, -14 
mul bl 
lea si, qword ptr [rdx + rsi + 21426] 
and rbx, 0b1111111111111 # instrumentation
and qword ptr [r14 + rbx], 99 
setnbe bl 
cmovno edi, eax 
cmovno cx, ax 
and rsi, 0b1111111111111 # instrumentation
sub cx, word ptr [r14 + rsi] 
sbb cl, al 
add al, bl 
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
movzx si, cl 
cmp cl, -14 
and rax, 0b1111111111111 # instrumentation
cmovnle dx, word ptr [r14 + rax] 
setnle bl 
sub rcx, rbx 
sub bl, -117 
cmovns dx, dx 
btc eax, esi 
cwde  
cmp eax, 1750659073 
and rdi, 0b1111111111111 # instrumentation
and eax, dword ptr [r14 + rdi] 
lea rcx, qword ptr [rbx + rax + 28920] 
setnb bl 
and al, al 
mov al, 89 
cmovz si, dx 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
mov al, dl 
imul al 
cmp bl, cl 
mov dx, si 
lea ecx, qword ptr [rcx + rcx] 
and rdi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rdi], edi 
adc al, bl 
or rcx, 0b1000000000000000000000000000000 # instrumentation
bsf rdi, rcx 
and rbx, 0b1111111111000 # instrumentation
and edx, 0b111 # instrumentation
lock bts dword ptr [r14 + rbx], edx 
cmovz rbx, rcx 
and rdi, 0b1111111111111 # instrumentation
mul word ptr [r14 + rdi] 
mov ecx, 789660227 
sub cl, 75 
cmovnz bx, ax 
btc dx, 135 
add cl, 51 # instrumentation
and rdi, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rcx], al 
cmovl dx, ax 
bts rcx, 23 
or ecx, 1 # instrumentation
and edx, ecx # instrumentation
shr edx, 1 # instrumentation
div ecx 
and rcx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rcx], 1 # instrumentation
and edx, dword ptr [r14 + rcx] # instrumentation
shr edx, 1 # instrumentation
div dword ptr [r14 + rcx] 
and rcx, 0b1111111111111 # instrumentation
and di, 0b111 # instrumentation
btc word ptr [r14 + rcx], di 
and rdi, 0b1111111111111 # instrumentation
setb byte ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmovnb rdi, qword ptr [r14 + rax] 
lea esi, qword ptr [rbx + rsi] 
cmp dl, -7 
and rsi, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rsi], rdi 
and rdx, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rdx], 89 
and rdx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdx], al 
xor dil, 2 
and rbx, 0b1111111111000 # instrumentation
lock sub dword ptr [r14 + rbx], 84 
and rdi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdi], al 
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
