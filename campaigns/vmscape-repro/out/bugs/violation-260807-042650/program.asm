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
xor rax, 794594828 
mov dl, cl 
movzx ebx, bx 
xchg ebx, eax 
lea rbx, qword ptr [rcx] 
sub bl, al 
lea edi, qword ptr [rdx + rbx + 23018] 
and rax, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
movzx ecx, word ptr [r14 + rbx] 
lea bx, qword ptr [rbx] 
and rdx, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rdx] 
lea rsi, qword ptr [rcx + rcx + 11254] 
adc ax, -9752 
xor esi, 48 
lea rax, qword ptr [rcx] 
cmp esi, 7 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
and rax, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rax], al 
and rax, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rax], 5 
and rdx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rdx], -128 
cmp cl, bl 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx 
and rcx, 0b1111111111000 # instrumentation
lock sub byte ptr [r14 + rcx], dl 
cmovns ebx, edx 
and rax, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rax], ecx 
imul esi, ecx 
add dl, 92 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovp rdi, qword ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
or qword ptr [r14 + rdi], -122 
or si, 1 # instrumentation
and dx, si # instrumentation
shr dx, 1 # instrumentation
div si 
add cl, 37 # instrumentation
adc cl, cl 
and bl, cl 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], -35 
xor sil, -50 
imul rsi 
and rdx, 0b1111111111111 # instrumentation
cmp byte ptr [r14 + rdx], dl 
and rdx, 0b1111111111111 # instrumentation
cmovns ax, word ptr [r14 + rdx] 
add eax, -1697488560 
and rdx, 0b1111111111111 # instrumentation
cmovz si, word ptr [r14 + rdx] 
xor edi, edi 
lea rcx, qword ptr [rax + rdx + 51376] 
and rax, 0b1111111111000 # instrumentation
lock xadd word ptr [r14 + rax], dx 
cmovnbe rcx, rdx 
and rsi, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rsi], 125 
and rax, 0b1111111111111 # instrumentation
btc word ptr [r14 + rax], 1 
add al, -121 # instrumentation
and rcx, 0b1111111111111 # instrumentation
seto byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
not dword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rcx], -357868585 
and rbx, 0b1111111111111 # instrumentation
or sil, byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
setnl byte ptr [r14 + rax] 
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
add al, 107 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
cmovnb esi, ecx 
add cl, 62 
cmovbe di, ax 
and rbx, 0b1111111111111 # instrumentation
movsx eax, word ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock not byte ptr [r14 + rcx] 
sbb esi, 95 
imul rbx 
add dl, 20 # instrumentation
cmovs ecx, esi 
seto dl 
and rdi, 0b1111111111111 # instrumentation
movzx edx, word ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
adc edi, dword ptr [r14 + rax] 
xor dl, cl 
sub dx, cx 
and cx, -50 
add ebx, edi 
and rax, 0b1111111111111 # instrumentation
neg word ptr [r14 + rax] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
add al, 31 # instrumentation
cmovnle esi, ecx 
cmovnbe ax, dx 
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rsi], si 
test al, -128 
and rbx, 0b1111111111111 # instrumentation
adc byte ptr [r14 + rbx], al 
not sil 
lea rax, qword ptr [rsi] 
and rcx, 0b1111111111111 # instrumentation
cmovnl rdx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmp bx, word ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
inc word ptr [r14 + rdi] 
mul ecx 
and rax, 0b1111111111111 # instrumentation
movsx edi, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
and byte ptr [r14 + rsi], bl 
btc eax, 249 
add cl, 80 # instrumentation
lea eax, qword ptr [rsi + rdx + 58465] 
cmovle rax, rax 
and rdx, 0b1111111111000 # instrumentation
and edi, 0b111 # instrumentation
lock btr dword ptr [r14 + rdx], edi 
sub al, -92 
and rcx, 0b1111111111111 # instrumentation
cmovnl esi, dword ptr [r14 + rcx] 
and rcx, 0b1111111111000 # instrumentation
lock add word ptr [r14 + rcx], 99 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx 
sbb al, dl 
and rax, 0b1111111111111 # instrumentation
cmovnz bx, word ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock adc qword ptr [r14 + rsi], rsi 
and rdi, 102 
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 66 
and rsi, 0b1111111111111 # instrumentation
and si, word ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mov dword ptr [r14 + rdi], -1374846084 
and rsi, 0b1111111111111 # instrumentation
bt word ptr [r14 + rsi], 0 
and rcx, 0b1111111111000 # instrumentation
lock or dword ptr [r14 + rcx], 94 
and rax, 0b1111111111111 # instrumentation
setl byte ptr [r14 + rax] 
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
