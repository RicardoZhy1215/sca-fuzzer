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
add cl, 48 # instrumentation
adc si, ax 
adc rcx, 46 
and rdx, 0b1111111111111 # instrumentation
cmovnle ebx, dword ptr [r14 + rdx] 
and bl, al 
inc rdi 
and rbx, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rbx], -51 
and rdi, 0b1111111111111 # instrumentation
cmovnbe edx, dword ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rcx], al 
bswap rax 
and rsi, 0b1111111111111 # instrumentation
add al, 113 # instrumentation
sbb sil, -63 
and rax, 0b1111111111111 # instrumentation
and rdi, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rdx], -41 
xchg al, sil 
cmovbe eax, eax 
cbw  
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
add al, 98 # instrumentation
cmovnl edx, esi 
and rdx, rdi 
adc dx, di 
sbb cx, -21 
and rcx, 0b1111111111111 # instrumentation
sbb word ptr [r14 + rcx], si 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdx 
cmovs rbx, rdx 
cmp cx, dx 
dec di 
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax] 
bt rbx, 204 
or eax, 125299768 
and rdi, 0b1111111111111 # instrumentation
sbb bx, word ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
cmovnb dx, word ptr [r14 + rdx] 
and al, al 
cmovnl si, cx 
cmovnle rdx, rsi 
mov bx, bx 
mov eax, -395873777 
and rax, 0b1111111111111 # instrumentation
sub al, byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmovnb rbx, qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
xadd word ptr [r14 + rbx], ax 
and rdx, 0b1111111111111 # instrumentation
and byte ptr [r14 + rdx], cl 
mul edi 
add al, -87 # instrumentation
cmovns dx, si 
and rsi, 0b1111111111111 # instrumentation
cmp cx, word ptr [r14 + rsi] 
setz dl 
and rsi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rsi], edx 
and rdi, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdi] 
or cl, dl 
and rdx, 0b1111111111111 # instrumentation
cmovb rsi, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
cmovns ebx, dword ptr [r14 + rdx] 
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
add al, 53 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmc  
setl dl 
sbb eax, 1761156337 
xor bl, 12 
btr di, 254 
lea di, qword ptr [rbx + rsi + 30446] 
sub di, 27 
and rdi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rdi], edx 
setz al 
cmovnz rsi, rcx 
add al, 56 
movsx ax, al 
lea si, qword ptr [rdx + rbx] 
sub ax, 4004 
mov cl, dl 
btr ecx, eax 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add al, -27 # instrumentation
and rax, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rax], rbx 
bts rbx, 215 
or rsi, 0b1000000000000000000000000000000 # instrumentation
bsf rsi, rsi 
and rax, 0b1111111111111 # instrumentation
and word ptr [r14 + rax], ax 
lea rsi, qword ptr [rcx + rsi] 
mul dl 
and rax, 0b1111111111111 # instrumentation
nop word ptr [r14 + rax] 
nop  
and rsi, 0b1111111111000 # instrumentation
xchg byte ptr [r14 + rsi], bl 
xchg dil, al 
test rax, 1297449600 
cmovno ecx, edx 
and rsi, 0b1111111111111 # instrumentation
xor word ptr [r14 + rsi], ax 
and rax, 0b1111111111000 # instrumentation
and cx, 0b111 # instrumentation
lock bts word ptr [r14 + rax], cx 
lea eax, qword ptr [rax + rsi + 46979] 
and al, -112 
test rax, 456891032 
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx] 
and bl, 92 
cmp rcx, rsi 
or edi, 7 
inc esi 
xor cl, bl 
xor rdx, -104 
lea edx, qword ptr [rax + rax] 
and rcx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rcx], 247390833 
mul al 
adc dx, 76 
stc  
lea rax, qword ptr [rax + rsi + 16577] 
xor bl, al 
and rdi, 0b1111111111111 # instrumentation
movzx dx, byte ptr [r14 + rdi] 
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
