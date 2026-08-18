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
add al, 68 # instrumentation
cmovnle ax, ax 
cmovs rbx, rcx 
cmp eax, -779858846 
lea rdx, qword ptr [rbx + rax] 
or ax, 0b1000000000000000 # instrumentation
bsr si, ax 
cmovnz eax, ecx 
lea rbx, qword ptr [rbx + rcx + 17632] 
lea edi, qword ptr [rsi] 
and cl, bl 
mov bl, bl 
test al, -60 
sbb eax, 1124285868 
bt cx, 180 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
bts rax, 77 
cmp rbx, -51 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
lea cx, qword ptr [rdx + rdx + 17617] 
add di, 34 
test si, dx 
and rax, 0b1111111111111 # instrumentation
cmovnp rcx, qword ptr [r14 + rax] 
and rbx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rbx], al 
cmpxchg rbx, rcx 
setbe bl 
and rdx, 0b1111111111111 # instrumentation
cmovns bx, word ptr [r14 + rdx] 
sub cl, dl 
lahf  
and rax, 0b1111111111111 # instrumentation
cmp al, byte ptr [r14 + rax] 
lea ebx, qword ptr [rdx] 
lea si, qword ptr [rdx + rcx + 60781] 
sbb rcx, -26 
and rcx, 0b1111111111111 # instrumentation
mul word ptr [r14 + rcx] 
setb sil 
sbb al, cl 
xor sil, -102 
adc edi, ecx 
and rax, 0b1111111111111 # instrumentation
or dword ptr [r14 + rax], 0b1000000000000000000000000000000 # instrumentation
bsf eax, dword ptr [r14 + rax] 
xor bx, cx 
cmovnp ecx, edi 
and rsi, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rsi], 79 
xadd edx, eax 
and rdx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rdx], -81 
cmp cl, -97 
and bl, dl 
bt edx, 47 
imul rbx, rdx 
and rsi, 0b1111111111111 # instrumentation
imul rcx, qword ptr [r14 + rsi] 
sub dil, cl 
and rsi, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rsi], bl 
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
xor bl, 110 
and rax, 0b1111111111111 # instrumentation
xadd dword ptr [r14 + rax], esi 
lea si, qword ptr [rax + rcx] 
and rdi, 0b1111111111111 # instrumentation
sub dil, byte ptr [r14 + rdi] 
lea dx, qword ptr [rbx + rsi + 20702] 
sbb rbx, rbx 
cmp bl, al 
setnb dl 
setz al 
or al, 14 
lea rbx, qword ptr [rcx + rax] 
lea rcx, qword ptr [rdi + rbx] 
or rcx, 0b1000000000000000000000000000000 # instrumentation
bsf rcx, rcx 
btc dx, di 
and dl, bl 
xchg dx, si 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rdx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdx], cl 
btr rcx, rax 
add dl, 33 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rdx] 
add al, 15 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovns edi, dword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
and cx, 0b111 # instrumentation
bts word ptr [r14 + rdi], cx 
and rax, 0b1111111111111 # instrumentation
cmovnb dx, word ptr [r14 + rax] 
neg al 
cmpxchg dil, cl 
not si 
and rsi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rsi], 0b1000000000000000000000000000000 # instrumentation
bsr rbx, qword ptr [r14 + rsi] 
and dx, -84 
and rdx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rdx] 
add ax, -1797 
and rdi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdi], -109 
and rbx, 0b1111111111000 # instrumentation
and bx, 0b111 # instrumentation
lock btc word ptr [r14 + rbx], bx 
add dl, 124 # instrumentation
and rdi, 0b1111111111111 # instrumentation
cmovbe cx, word ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
nop word ptr [r14 + rsi] 
setnl sil 
and rdi, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rdi], 41 
not edi 
and rsi, 0b1111111111111 # instrumentation
dec byte ptr [r14 + rsi] 
adc ax, 16286 
test eax, -772650567 
lea edx, qword ptr [rax + rsi] 
test bl, bl 
lea di, qword ptr [rcx + rsi + 26454] 
and rdx, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rdx] 
setnbe dl 
lea rdi, qword ptr [rcx] 
and rax, 0b1111111111111 # instrumentation
btc dword ptr [r14 + rax], 1 
cmpxchg dx, ax 
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
