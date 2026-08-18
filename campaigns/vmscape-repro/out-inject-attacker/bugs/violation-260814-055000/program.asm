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
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
sub bx, -12 
and rax, 0b1111111111111 # instrumentation
or byte ptr [r14 + rax], 0b1000 # instrumentation
and byte ptr [r14 + rax], 0b11111000 # instrumentation
and rcx, 0b1111111111000 # instrumentation
and rbx, 0b111 # instrumentation
lock btr qword ptr [r14 + rcx], rbx 
xor rax, 944439145 
movsx ebx, al 
add cl, dl 
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rdx 
cmpxchg rsi, rax 
cmovz dx, bx 
movzx ecx, al 
imul dl 
add bl, -53 # instrumentation
lea rdx, qword ptr [rdi] 
cmovnbe esi, edi 
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi] 
lea rbx, qword ptr [rbx] 
lea rbx, qword ptr [rip +.function_vmvictim_3] # noremove
jmp rbx # noremove
.exit_vmvictim_1:
.function_vmvictim_3:
.bb_vmvictim_3.entry:
add al, 116 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovnbe ecx, dword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rdi] 
cmovnbe dx, ax 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi 
imul si, dx, -120 
and rbx, 0b1111111111111 # instrumentation
btc qword ptr [r14 + rbx], 0 
and rcx, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rcx], esi 
and rdx, 0b1111111111111 # instrumentation
add word ptr [r14 + rdx], 52 
lea di, qword ptr [rdi + rsi] 
and rbx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rbx], al 
sub bl, dl 
inc si 
not si 
cmovle rax, rdx 
or dl, 1 # instrumentation
add bl, 127 # instrumentation
lea rdx, qword ptr [rax] 
cmovnbe esi, ecx 
and rdi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdi] 
setp dl 
and rdx, 0b1111111111111 # instrumentation
test word ptr [r14 + rdx], 19727 
and rcx, 0b1111111111111 # instrumentation
cmovnl si, word ptr [r14 + rcx] 
lea si, qword ptr [rbx + rdi] 
cmovnl edx, ecx 
stc  
adc al, 43 
sets al 
and rsi, 0b1111111111111 # instrumentation
cmovnp edx, dword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdx], 1574902950 
or al, 0b1000 # instrumentation
and al, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv al 
add bl, 111 # instrumentation
setle sil 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rsi] 
add cl, 64 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovno rsi, qword ptr [r14 + rcx] 
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
add al, al 
or esi, edx 
sub eax, eax 
and rax, 0b1111111111111 # instrumentation
imul rdx, qword ptr [r14 + rax] 
movsx ecx, bl 
xadd dl, cl 
xor dl, bl 
lea ecx, qword ptr [rcx + rsi + 4390] 
cmp di, 11 
lea di, qword ptr [rsi] 
sub dil, cl 
and rcx, 0b1111111111111 # instrumentation
cmovnb cx, word ptr [r14 + rcx] 
cmovnz bx, bx 
and rcx, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rcx] 
cmovo cx, si 
cmp rax, 1280300028 
lea rax, qword ptr [rip +.function_vm_3] # noremove
jmp rax # noremove
.exit_vm_train:
.function_vm_3:
.bb_vm_3.entry:
and rcx, 0b1111111111111 # instrumentation
not qword ptr [r14 + rcx] 
btr si, di 
add bl, 44 # instrumentation
setnp dl 
xchg cl, bl 
and rax, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rax], dx 
movzx edx, bl 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
btr ecx, edi 
xadd rdi, rcx 
cmovnb ecx, ebx 
add dl, cl 
cmovnp cx, dx 
add ax, 24116 
cmp rax, -841896418 
and rcx, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rcx] 
lea rbx, qword ptr [rsi] 
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
