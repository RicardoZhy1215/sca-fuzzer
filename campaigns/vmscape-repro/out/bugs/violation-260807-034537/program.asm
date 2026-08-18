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
and rdx, 0b1111111111000 # instrumentation
lock xor word ptr [r14 + rdx], -127 
setbe al 
adc si, cx 
imul rcx, rdi, 0 
add al, 54 # instrumentation
and rdx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rdx] 
cmovle bx, dx 
sbb sil, -33 
cmovz ecx, eax 
lea ax, qword ptr [rsi + rsi] 
test rax, -512450426 
btr rdi, rbx 
and rdx, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rdx], -35 
cmovb rdi, rdi 
lea rax, qword ptr [rdx] 
add bx, 22 
lea rsi, qword ptr [rax + rcx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add dl, -102 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovno rdx, qword ptr [r14 + rcx] 
setz cl 
and rax, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rax], dil 
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax] 
sbb dl, dl 
and rsi, 0b1111111111111 # instrumentation
xor rsi, qword ptr [r14 + rsi] 
adc ax, -29727 
or eax, 1362940110 
and rcx, 0b1111111111111 # instrumentation
cmovle ecx, dword ptr [r14 + rcx] 
and cx, -79 
and rax, 0b1111111111111 # instrumentation
mov edi, dword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
add bl, byte ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], -108 
movzx rbx, ax 
lea di, qword ptr [rax + rsi] 
setnle dil 
and rbx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rbx], al 
lea rdx, qword ptr [rbx + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rdx], bl 
and rdx, 0b1111111111111 # instrumentation
setnp byte ptr [r14 + rdx] 
cmovz ax, bx 
and rdx, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rdx] 
sub si, 63 
sbb ax, -27798 
and edx, 88 
cmp si, -83 
cmp cl, al 
mov bl, al 
sub dl, bl 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], 127 
and rsi, 0b1111111111111 # instrumentation
cmovns ecx, dword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
btc word ptr [r14 + rdi], 4 
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
add al, 97 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setnb cl 
and rcx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rcx] 
and rbx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rbx], bl 
and rcx, 0b1111111111111 # instrumentation
imul word ptr [r14 + rcx] 
xor dl, 68 
or bl, 1 # instrumentation
mov ax, 1 # instrumentation
div bl 
lea rcx, qword ptr [rcx + rcx] 
and rcx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rcx], 5 
sub al, al 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], 0b1000000000000000000000000000000 # instrumentation
bsf ecx, dword ptr [r14 + rdx] 
lea rdi, qword ptr [rdx] 
and rdi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdi], bl 
movsx ebx, dl 
sub rdi, rcx 
cmovp ax, di 
and rdx, 0b1111111111111 # instrumentation
inc qword ptr [r14 + rdx] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
lea ax, qword ptr [rcx + rdx] 
and rax, -1737968400 
cmp dl, al 
and rdi, 0b1111111111111 # instrumentation
cmovs si, word ptr [r14 + rdi] 
xor eax, 1361616602 
and rbx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rbx], cl 
and rsi, 0b1111111111111 # instrumentation
and edi, 0b111 # instrumentation
bt dword ptr [r14 + rsi], edi 
add al, 38 # instrumentation
setns sil 
lea dx, qword ptr [rax + rdx] 
sbb eax, -120956857 
and rdi, 0b1111111111111 # instrumentation
or byte ptr [r14 + rdi], cl 
and rcx, 0b1111111111111 # instrumentation
cmovnle ecx, dword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
test word ptr [r14 + rax], bx 
sub rcx, -104 
cdq  
and rsi, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rsi], -97 
and rdx, 0b1111111111111 # instrumentation
cmp ecx, dword ptr [r14 + rdx] 
cmovle ebx, edi 
and rbx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rbx], 57 
cmovnl rcx, rcx 
and rdx, 0b1111111111111 # instrumentation
cmovle edx, dword ptr [r14 + rdx] 
cmovno ax, ax 
and rcx, 0b1111111111111 # instrumentation
cmovs bx, word ptr [r14 + rcx] 
lea esi, qword ptr [rcx] 
cmovo rdi, rax 
and cl, bl 
and rcx, 0b1111111111111 # instrumentation
or eax, dword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], 59 
sbb dl, bl 
setle dl 
and rcx, 0b1111111111111 # instrumentation
imul rdx, qword ptr [r14 + rcx], 116 
add cl, 53 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovz rsi, qword ptr [r14 + rdi] 
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
