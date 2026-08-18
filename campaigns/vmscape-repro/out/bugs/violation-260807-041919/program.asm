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
add al, 44 # instrumentation
sbb rdx, -103 
setnb cl 
lea bx, qword ptr [rbx + rcx + 61058] 
and rcx, 0b1111111111111 # instrumentation
sbb bx, word ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rax], ebx 
cdq  
xadd ebx, edi 
sub eax, -470515852 
add al, bl 
cmovno esi, edi 
setnl sil 
not bl 
sub al, 36 
and rbx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rbx], 1 
add bl, 77 # instrumentation
cmovnp cx, bx 
btr ebx, 81 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 47 
bt dx, 139 
cmovbe edx, ecx 
lea ebx, qword ptr [rax + rdx + 10227] 
and cx, -29 
adc dl, 88 
and rdi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rdi], dil 
and rax, 0b1111111111111 # instrumentation
cmovnb edi, dword ptr [r14 + rax] 
and rax, 0b1111111111111 # instrumentation
movsx esi, byte ptr [r14 + rax] 
cmovp cx, ax 
and rbx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rbx], -78 
and rdi, 0b1111111111111 # instrumentation
imul edx, dword ptr [r14 + rdi], -128 
add dl, 25 # instrumentation
cmovp rdi, rbx 
and rcx, 0b1111111111111 # instrumentation
sub esi, dword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
and si, word ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
movsx eax, word ptr [r14 + rcx] 
cmovns rax, rdx 
not rax 
cmovnp ecx, esi 
nop rsi 
adc ebx, eax 
and rdi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdi], -113 
mov al, dl 
sub dl, al 
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], cl 
xor rcx, 94 
cmp al, cl 
xor eax, -205137811 
and rax, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rax], -33 
and rcx, 0b1111111111111 # instrumentation
mov bx, word ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rdx], -96 
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
add dl, -8 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
setl bl 
clc  
and bl, 61 
add dil, 47 
and rsi, 0b1111111111111 # instrumentation
imul rax, qword ptr [r14 + rsi], -84 
sbb ax, 15240 
add dl, dl 
and rcx, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rdx] 
lea cx, qword ptr [rax] 
cmovl ebx, ebx 
xor ecx, edi 
lea eax, qword ptr [rdx + rdx + 1017] 
and rax, 0b1111111111111 # instrumentation
cmp dil, byte ptr [r14 + rax] 
mov esi, eax 
or cl, cl 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, -111 # instrumentation
and rsi, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rsi], bl 
lea esi, qword ptr [rbx] 
and rbx, 0b1111111111111 # instrumentation
and ecx, 0b111 # instrumentation
btr dword ptr [r14 + rbx], ecx 
and rbx, 0b1111111111111 # instrumentation
cmovnz ebx, dword ptr [r14 + rbx] 
mov sil, al 
lea esi, qword ptr [rdi] 
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 1 # instrumentation
and dx, word ptr [r14 + rbx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rbx] 
add cl, -101 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovnle eax, dword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock sbb word ptr [r14 + rsi], 85 
mov rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rcx], eax 
and rbx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rbx], ecx 
and rcx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rcx], al 
not dl 
and rcx, 0b1111111111111 # instrumentation
cmp ecx, dword ptr [r14 + rcx] 
xor dl, al 
and rax, 0b1111111111000 # instrumentation
and edx, 0b111 # instrumentation
lock btr dword ptr [r14 + rax], edx 
and rdx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdx], 57 
and rsi, 0b1111111111000 # instrumentation
and ebx, 0b111 # instrumentation
lock btc dword ptr [r14 + rsi], ebx 
or ecx, -16 
and rdx, 0b1111111111111 # instrumentation
or ecx, dword ptr [r14 + rdx] 
lea ebx, qword ptr [rbx + rcx + 22253] 
xchg di, ax 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
neg cl 
and rbx, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rdi], -95 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
or dl, sil 
setnle al 
and rax, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rax], 26 
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
