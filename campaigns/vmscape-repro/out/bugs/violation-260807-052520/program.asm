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
or cl, 0b1000 # instrumentation
and cl, 0b11111000 # instrumentation
mov ax, 1 # instrumentation
idiv cl 
add bl, 30 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnp dx, word ptr [r14 + rdx] 
adc dl, bl 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
lea rcx, qword ptr [rax + rsi + 62393] 
and rsi, 0b1111111111111 # instrumentation
and dl, byte ptr [r14 + rsi] 
lea rax, qword ptr [rdi] 
setb cl 
not bl 
seto dl 
inc rcx 
setnb cl 
mul sil 
or rdx, 0b1000000000000000000000000000000 # instrumentation
bsr rcx, rdx 
add cl, 87 # instrumentation
mov di, bx 
and rbx, 0b1111111111111 # instrumentation
cmovl si, word ptr [r14 + rbx] 
lea rax, qword ptr [rip +.function_vmvictim_1] # noremove
jmp rax # noremove
.exit_vmvictim_0:
.function_vmvictim_1:
.bb_vmvictim_1.entry:
or al, -103 
btr bx, 19 
cmovb di, si 
and rdx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rdx], 30 
xor al, cl 
cmovnz cx, si 
test ecx, ecx 
and rdx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock btr qword ptr [r14 + rax], 1 
or al, -107 
lea edx, qword ptr [rsi + rdi + 50738] 
and rdi, 0b1111111111111 # instrumentation
mov bl, byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
xor rsi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rax], rbx 
and rsi, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rsi], 84 
cmovnb si, cx 
lea rax, qword ptr [rsi + rdi] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx 
and rcx, 0b1111111111111 # instrumentation
cmpxchg qword ptr [r14 + rcx], rbx 
and rcx, 0b1111111111111 # instrumentation
movzx si, byte ptr [r14 + rcx] 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], 92 
and rsi, 0b1111111111000 # instrumentation
lock btr dword ptr [r14 + rsi], 2 
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rbx], si 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsr si, word ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rdi], -816584455 
and rcx, 0b1111111111111 # instrumentation
xor word ptr [r14 + rcx], 61 
and rsi, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rsi], 34 
cmovns si, ax 
and rcx, 0b1111111111000 # instrumentation
lock not word ptr [r14 + rcx] 
seto dl 
xchg dx, ax 
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
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv word ptr [r14 + rsi] 
not ecx 
lea cx, qword ptr [rcx] 
test ebx, -1488332964 
and rsi, 57 
xor bl, -51 
or rax, 562937352 
and rdi, 0b1111111111111 # instrumentation
cmovl edi, dword ptr [r14 + rdi] 
cmovo bx, bx 
cmovnz di, cx 
cmovnbe rbx, rdx 
and dil, -5 
cmovl ecx, esi 
btc si, bx 
lea si, qword ptr [rsi] 
lea rbx, qword ptr [rbx] 
lea rbx, qword ptr [rip +.function_vm_3] # noremove
jmp rbx # noremove
.exit_vm_1:
.function_vm_3:
.bb_vm_3.entry:
imul ebx 
and rcx, 0b1111111111111 # instrumentation
bt dword ptr [r14 + rcx], 1 
sub rax, -2066167108 
imul dl 
and rbx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rbx], dl 
and rdi, 0b1111111111111 # instrumentation
cmovz rax, qword ptr [r14 + rdi] 
lea bx, qword ptr [rcx + rsi] 
mov rbx, 9026357695792696538 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi 
sub rdx, -128 
adc ecx, ebx 
and rdi, 0b1111111111111 # instrumentation
adc dword ptr [r14 + rdi], esi 
and rbx, 0b1111111111000 # instrumentation
lock sbb byte ptr [r14 + rbx], dl 
and rbx, 0b1111111111111 # instrumentation
cmovz di, word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
sbb dl, byte ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov al, byte ptr [r14 + rax] 
lea rdi, qword ptr [rdi + rax + 13502] 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rsi 
bts rbx, 175 
and rsi, 0b1111111111111 # instrumentation
mov cl, byte ptr [r14 + rsi] 
and rax, 0b1111111111000 # instrumentation
lock and byte ptr [r14 + rax], -105 
mov sil, -86 
cmovl eax, eax 
xchg sil, sil 
and rsi, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rsi], 65 
and rbx, 0b1111111111111 # instrumentation
sbb qword ptr [r14 + rbx], 70 
or ax, 29776 
and rcx, 0b1111111111111 # instrumentation
or byte ptr [r14 + rcx], dl 
mul rsi 
and rbx, 0b1111111111111 # instrumentation
test byte ptr [r14 + rbx], 30 
imul dl 
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
