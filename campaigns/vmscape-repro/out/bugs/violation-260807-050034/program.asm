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
lea ebx, qword ptr [rax] 
xchg dil, al 
cmp cl, 82 
adc dl, al 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], -15 
add cl, -34 
cmovno ax, dx 
and rsi, 0b1111111111111 # instrumentation
cmovnz dx, word ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
and rcx, 0b111 # instrumentation
bts qword ptr [r14 + rdi], rcx 
lea rbx, qword ptr [rdx + rax + 47769] 
nop rdx 
cmpxchg cl, bl 
and rsi, 0b1111111111111 # instrumentation
test byte ptr [r14 + rsi], -40 
and rax, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rax], edi 
neg ebx 
and cl, dl 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
btc ecx, ecx 
add al, dl 
and rdx, 0b1111111111111 # instrumentation
cmovnp edi, dword ptr [r14 + rdx] 
xor ax, -29049 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx 
and rdi, 0b1111111111111 # instrumentation
mov si, word ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
or word ptr [r14 + rdi], 0b1000000000000000 # instrumentation
bsf bx, word ptr [r14 + rdi] 
add bl, -2 # instrumentation
cmovno cx, si 
and rdx, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rdx], rsi 
nop bx 
and rdx, 0b1111111111111 # instrumentation
xadd byte ptr [r14 + rdx], bl 
and rax, 0b1111111111111 # instrumentation
cmovbe ax, word ptr [r14 + rax] 
bts dx, ax 
and rdi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdi], al 
and rbx, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rcx], 6 
and rdx, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bts word ptr [r14 + rdx], si 
and rdx, 0b1111111111111 # instrumentation
add al, 77 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovnb rdx, qword ptr [r14 + rcx] 
movzx rsi, di 
add cl, al 
setns cl 
and rdi, 0b1111111111111 # instrumentation
cmovle edx, dword ptr [r14 + rdi] 
sub ebx, -34 
and rdx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdx], dl 
lea cx, qword ptr [rbx + rdi + 35561] 
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], cl 
seto dl 
and rcx, 118 
and rax, 0b1111111111111 # instrumentation
mov eax, dword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
cmovno si, word ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rdi] 
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
and rdi, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdi], bl 
or rdx, 71 
add al, 118 
lea si, qword ptr [rsi + rcx + 9708] 
bswap rsi 
and rdx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rdx], bl 
lea rdi, qword ptr [rbx + rcx + 23913] 
sub sil, al 
lea rbx, qword ptr [rax] 
lea rdx, qword ptr [rdi] 
imul rbx, rcx, -74 
movzx edx, dx 
lea di, qword ptr [rdi + rcx + 64843] 
or dl, -99 
add rax, 1538761527 
xadd eax, edi 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
and rbx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rbx], al 
and rdx, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdx], eax 
or dl, cl 
and rcx, 0b1111111111000 # instrumentation
and dx, 0b111 # instrumentation
lock btc word ptr [r14 + rcx], dx 
and rdx, 0b1111111111111 # instrumentation
and si, 0b111 # instrumentation
bts word ptr [r14 + rdx], si 
btr bx, di 
and rdi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rdi], al 
cmovb eax, esi 
lea esi, qword ptr [rsi] 
and rdx, 0b1111111111000 # instrumentation
lock sbb dword ptr [r14 + rdx], 93 
test rcx, -159651608 
cmp rbx, rbx 
lea bx, qword ptr [rax + rbx + 31347] 
imul ecx 
add bl, 44 # instrumentation
and rbx, 0b1111111111111 # instrumentation
nop dword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmovp rcx, qword ptr [r14 + rdi] 
and rax, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rax] 
lea esi, qword ptr [rdx + rdx + 52107] 
and rsi, 0b1111111111111 # instrumentation
and ebx, 0b111 # instrumentation
bt dword ptr [r14 + rsi], ebx 
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rbx 
and rdx, 0b1111111111111 # instrumentation
or cl, byte ptr [r14 + rdx] 
or ax, 11303 
adc cl, al 
btr bx, 174 
and rdi, 0b1111111111111 # instrumentation
sbb rbx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
and rax, rcx 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], -124 
and rbx, 0b1111111111111 # instrumentation
cmovp si, word ptr [r14 + rbx] 
seto cl 
and rdx, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rdx], 20 
cmovnb edi, edi 
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
