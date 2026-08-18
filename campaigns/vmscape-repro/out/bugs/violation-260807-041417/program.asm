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
add cl, 118 # instrumentation
cmovnl ax, di 
bt esi, esi 
or ax, -18182 
setnle dil 
lea si, qword ptr [rcx + rax + 23071] 
movsx ax, sil 
and cl, dl 
sbb eax, 110 
and rbx, 0b1111111111111 # instrumentation
adc al, byte ptr [r14 + rbx] 
test al, cl 
lea bx, qword ptr [rsi + rcx] 
sbb bl, cl 
and rbx, 0b1111111111111 # instrumentation
movsx ebx, byte ptr [r14 + rbx] 
adc rax, rcx 
imul edx 
add cl, -40 # instrumentation
setns dil 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add cl, 38 # instrumentation
and rax, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rax], ebx 
movsx edi, dl 
and dl, dl 
xchg dl, dl 
adc al, 67 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rbx, 0b1111111111000 # instrumentation
and rcx, 0b111 # instrumentation
lock btc qword ptr [r14 + rbx], rcx 
or dl, al 
and rax, 0b1111111111000 # instrumentation
lock xor dword ptr [r14 + rax], 38 
or ax, 3788 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], 56 
lea rsi, qword ptr [rcx] 
and dil, 126 
and rsi, 0b1111111111111 # instrumentation
cmovbe ebx, dword ptr [r14 + rsi] 
cmc  
btr rax, 112 
xor cl, bl 
cmovno eax, esi 
and rdx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdx] 
lea edi, qword ptr [rbx] 
sub bl, dl 
or dl, al 
and rdx, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rdx] 
bts edx, ecx 
btr dx, 117 
cmp rax, 21 
sbb bl, cl 
and rsi, 0b1111111111111 # instrumentation
imul rdi, qword ptr [r14 + rsi], -106 
lea edi, qword ptr [rbx + rdi] 
test cl, 90 
clc  
add cl, cl 
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
movzx eax, bl 
and rsi, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rsi], 116 
sub rsi, -77 
sbb al, 78 
sbb ecx, 62 
and rcx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rcx], rbx 
and rsi, 0b1111111111111 # instrumentation
cmovnb rdi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
xor ebx, dword ptr [r14 + rcx] 
setns cl 
cmovs rbx, rsi 
lea ax, qword ptr [rdi + rdi + 21504] 
movzx rax, di 
xchg rax, rcx 
cmovnp ax, di 
or rax, 0b1000000000000000000000000000000 # instrumentation
bsf rdi, rax 
btr dx, dx 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
movzx si, byte ptr [r14 + rbx] 
add rdx, -99 
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl 
add dl, -61 # instrumentation
sbb al, dl 
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], dl 
lea cx, qword ptr [rax + rdi] 
xor al, bl 
cmp al, dl 
and rbx, 0b1111111111111 # instrumentation
or dl, byte ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmovnp rsi, qword ptr [r14 + rcx] 
and bl, cl 
and rbx, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rbx] 
cmpxchg ecx, edx 
add rbx, -90 
xadd dil, bl 
setz dl 
and eax, 2037706443 
and bl, -35 
add dil, -111 
and rdx, 0b1111111111111 # instrumentation
neg dword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovs ebx, dword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmovno bx, word ptr [r14 + rcx] 
sbb al, cl 
and rcx, 0b1111111111111 # instrumentation
cmp ecx, dword ptr [r14 + rcx] 
cmovl eax, eax 
and rdx, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], cl 
setnle dl 
add dl, 65 
and rcx, 0b1111111111111 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rcx], bl 
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
