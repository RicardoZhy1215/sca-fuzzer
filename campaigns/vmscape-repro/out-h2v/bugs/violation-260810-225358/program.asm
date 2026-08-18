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
add cl, -24 # instrumentation
and rdx, 0b1111111111111 # instrumentation
adc edi, dword ptr [r14 + rdx] 
or esi, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv esi 
sub cl, 37 
cmovs rcx, rsi 
and rbx, 0b1111111111111 # instrumentation
cmovz esi, dword ptr [r14 + rbx] 
bt ax, cx 
and rax, 0b1111111111111 # instrumentation
dec word ptr [r14 + rax] 
xor sil, -56 
movzx edi, sil 
and rsi, 0b1111111111111 # instrumentation
xor al, byte ptr [r14 + rsi] 
lea rsi, qword ptr [rsi + rbx] 
or dl, 1 # instrumentation
test rcx, rbx 
adc di, 37 
and rcx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rcx], bl 
or rcx, rax 
lea rax, qword ptr [rip +.function_main_gadget] # noremove
jmp rax # noremove
.exit_main_1:
.function_main_gadget:
.bb_main_gadget.entry:
add cl, 8 # instrumentation
cmovnp rcx, rax 
setle bl 
cmp rbx, rbx 
xchg rdx, rsi 
sbb dl, al 
adc bl, cl 
and rcx, 0b1111111111111 # instrumentation
imul ax, word ptr [r14 + rcx] 
add dl, dl 
and rsi, 0b1111111111111 # instrumentation
mov word ptr [r14 + rsi], si 
and rdx, 0b1111111111111 # instrumentation
and esi, 0b111 # instrumentation
bts dword ptr [r14 + rdx], esi 
add al, -70 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovnle dx, word ptr [r14 + rdi] 
and bl, bl 
imul eax, ecx 
cmovnb rbx, rdx 
and rax, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rbx], 0b1000000000000000000000000000000 # instrumentation
bsf ecx, dword ptr [r14 + rbx] 
add al, 77 # instrumentation
and rsi, 0b1111111111111 # instrumentation
adc dil, byte ptr [r14 + rsi] 
movsx ebx, si 
setl dl 
movsx ebx, bl 
and rax, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rax], dl 
and rbx, 0b1111111111111 # instrumentation
test word ptr [r14 + rbx], -22809 
test sil, -43 
and rcx, 0b1111111111111 # instrumentation
cmovno bx, word ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
sub ecx, dword ptr [r14 + rdi] 
cmovno edx, ebx 
and rax, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rdi], -37 
and rbx, 0b1111111111111 # instrumentation
cmpxchg byte ptr [r14 + rbx], sil 
and rcx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rcx] 
and rcx, 0b1111111111000 # instrumentation
lock add byte ptr [r14 + rcx], dl 
lfence  
.macro.set_h2g_target.vm.function_vm_1: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_2: nop qword ptr [rax + 0xff]
xor rax, rax # noremove
xor rbx, rbx # noremove
xor rcx, rcx # noremove
xor rdx, rdx # noremove
xor rsi, rsi # noremove
xor rdi, rdi # noremove
.macro.switch_h2g.vm.1: nop qword ptr [rax + 0xff]
.exit_main_gadget:
.function_main_2:
.bb_main_2.entry:
.macro.landing_g2h.main_2: nop qword ptr [rax + 0xff]
.macro.fault_handler: nop qword ptr [rax + 0xff]
.patch_placeholder_fault_handler:
.macro.set_h2g_target.vm.function_vm_2: nop qword ptr [rax + 0xff]
.macro.set_g2h_target.main.function_main_3: nop qword ptr [rax + 0xff]
.macro.switch_h2g.vm.2: nop qword ptr [rax + 0xff]
.exit_main_2:
.function_main_3:
.bb_main_3.entry:
.macro.landing_g2h.main_3: nop qword ptr [rax + 0xff]
nop  
.exit_main_3:
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
add al, 59 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
sbb dx, bx 
sbb bl, dil 
cmovbe dx, ax 
add rdx, rcx 
adc edi, -115 
and rax, 0b1111111111111 # instrumentation
cmovnp edi, dword ptr [r14 + rax] 
sbb cl, 113 
dec cl 
cmpxchg bl, dil 
sbb sil, -62 
adc rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
cmovnl esi, dword ptr [r14 + rcx] 
setbe dl 
xor bl, al 
movzx bx, bl 
seto cl 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 2019396165 
and rax, 0b1111111111111 # instrumentation
and rdi, 0b111 # instrumentation
btr qword ptr [r14 + rax], rdi 
mul ecx 
add al, -16 # instrumentation
setz dil 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsf rdi, qword ptr [r14 + rdi] 
or dx, 0b1000 # instrumentation
and dl, 0b11111000 # instrumentation
add cl, -75 # instrumentation
setnp dl 
and rbx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rbx], 2 
and rdx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rdx], di 
adc di, 10 
cmovl ax, bx 
lea rsi, qword ptr [rbx] 
test rsi, rcx 
cmovnbe rsi, rbx 
and rsi, 0b1111111111111 # instrumentation
movzx esi, word ptr [r14 + rsi] 
lea edx, qword ptr [rdi + rbx + 7483] 
not dl 
and rbx, 0b1111111111111 # instrumentation
mul byte ptr [r14 + rbx] 
adc eax, 1007930178 
cmp bl, cl 
xadd bx, dx 
and rax, 0b1111111111000 # instrumentation
and rcx, 0b111 # instrumentation
lock bts qword ptr [r14 + rax], rcx 
and rcx, 0b1111111111111 # instrumentation
inc dword ptr [r14 + rcx] 
xadd al, al 
and rcx, 0b1111111111111 # instrumentation
cmovnb ecx, dword ptr [r14 + rcx] 
adc rdi, rbx 
neg edx 
setnb dl 
cmovns si, ax 
and rdx, 0b1111111111111 # instrumentation
cmovp rsi, qword ptr [r14 + rdx] 
xor rax, -1913599453 
cmovnz eax, edx 
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
