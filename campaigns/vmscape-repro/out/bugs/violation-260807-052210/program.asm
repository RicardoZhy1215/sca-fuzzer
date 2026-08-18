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
or bx, 111 
inc dil 
cmovns rbx, rdi 
add dl, 57 
and rax, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rax], al 
dec al 
cmovz cx, ax 
test dil, -17 
and rsi, 0b1111111111111 # instrumentation
test word ptr [r14 + rsi], si 
setbe cl 
and rcx, 0b1111111111111 # instrumentation
cmovp rdi, qword ptr [r14 + rcx] 
test al, 27 
cmc  
or dil, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv dil 
and rdi, 0b1111111111111 # instrumentation
xor ecx, dword ptr [r14 + rdi] 
movzx edx, dl 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], bl 
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], -105 
xchg di, ax 
xor cl, bl 
sahf  
and rdi, 0b1111111111111 # instrumentation
nop qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
test word ptr [r14 + rdi], -29501 
and rsi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rsi], 102 
or al, 1 # instrumentation
mov ax, 1 # instrumentation
div al 
add cl, -9 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovs dx, word ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rcx 
neg esi 
adc rax, -78 
xor ecx, edx 
lea dx, qword ptr [rdi] 
btr di, 60 
and rcx, 0b1111111111111 # instrumentation
cmovb rdi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdi] 
add dl, -98 # instrumentation
setle bl 
and rcx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rcx], sil 
lea edi, qword ptr [rbx + rcx] 
and rsi, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rsi], rdx 
cmpxchg rdx, rax 
and rax, 0b1111111111111 # instrumentation
cmovo rcx, qword ptr [r14 + rax] 
add dx, di 
bts di, 183 
add dl, 98 # instrumentation
and rdx, 0b1111111111111 # instrumentation
movzx dx, byte ptr [r14 + rdx] 
cmovp ecx, esi 
nop esi 
and rdi, 0b1111111111000 # instrumentation
xchg dword ptr [r14 + rdi], ecx 
nop  
test cl, cl 
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
add bl, 120 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rbx, 0b1111111111111 # instrumentation
cmovnbe edx, dword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmovnz edx, dword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rdx], al 
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], dl 
lea ecx, qword ptr [rsi] 
or si, si 
setnp cl 
lea rcx, qword ptr [rdi + rax + 7413] 
lea edx, qword ptr [rbx + rcx + 41856] 
and dil, 84 
cmovb rdx, rax 
cmovp rdx, rax 
setnle cl 
cmp rax, -1453780577 
and sil, -106 
imul bl 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add cl, -1 # instrumentation
and rdx, 0b1111111111111 # instrumentation
adc word ptr [r14 + rdx], 29 
mov rdx, 5516519095662175338 
lea ebx, qword ptr [rax] 
sbb dl, -102 
movsx edx, al 
xor bx, 47 
and rbx, 0b1111111111111 # instrumentation
cmovno rbx, qword ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg dword ptr [r14 + rcx], edi 
cmovle rax, rdi 
xchg al, bl 
or al, bl 
and rcx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rcx], eax 
not bl 
test dil, 77 
bt di, si 
and rbx, 0b1111111111111 # instrumentation
test qword ptr [r14 + rbx], rax 
or dl, dl 
and rdi, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rdi] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], -58 
movzx rdx, di 
xadd edi, esi 
and rbx, 0b1111111111111 # instrumentation
sbb rbx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rsi] 
adc si, ax 
and rdi, 0b1111111111111 # instrumentation
or cx, word ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock btc word ptr [r14 + rax], 1 
and rdi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rdi] 
and dl, al 
and rax, 0b1111111111111 # instrumentation
cmovns bx, word ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rbx], dx 
and rdi, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rdi], esi 
and rcx, 0b1111111111111 # instrumentation
btc word ptr [r14 + rcx], 0 
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
