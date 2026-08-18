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
.patch_placeholder:
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
add cl, -33 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovnp eax, dword ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock and dword ptr [r14 + rbx], edi 
and rax, 0b1111111111111 # instrumentation
add ax, word ptr [r14 + rax] 
setnz al 
or edi, 0b1000 # instrumentation
and dil, 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv edi 
and rax, 0b1111111111000 # instrumentation
lock and word ptr [r14 + rax], -33 
and rdx, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rdx] 
mov dl, 120 
and rdx, 0b1111111111111 # instrumentation
cmovno ebx, dword ptr [r14 + rdx] 
lea dx, qword ptr [rcx + rcx + 2092] 
and rdx, 0b1111111111000 # instrumentation
lock xor byte ptr [r14 + rdx], -27 
and rcx, 0b1111111111111 # instrumentation
cmovnb rcx, qword ptr [r14 + rcx] 
cmovbe eax, esi 
xor al, bl 
imul bx, di, -80 
and rax, 0b1111111111111 # instrumentation
btr qword ptr [r14 + rax], 5 
and rsi, 0b1111111111111 # instrumentation
or bl, byte ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock neg qword ptr [r14 + rdi] 
cmovns ebx, edi 
and rsi, 0b1111111111111 # instrumentation
or word ptr [r14 + rsi], 0b1000000000000000 # instrumentation
bsf si, word ptr [r14 + rsi] 
add al, al 
lahf  
and rcx, 0b1111111111000 # instrumentation
lock add dword ptr [r14 + rcx], edi 
sub rax, rdx 
and rax, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rax], 69 
cmovs rdx, rsi 
and rsi, 0b1111111111111 # instrumentation
cmovno rbx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rdi] 
or dx, dx 
test dx, ax 
and rbx, 0b1111111111111 # instrumentation
nop word ptr [r14 + rbx] 
and rsi, rdx 
and eax, -43 
add bl, 33 
and rsi, 0b1111111111000 # instrumentation
lock neg byte ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
setle byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
btr dword ptr [r14 + rdx], 6 
lea esi, qword ptr [rbx] 
and rsi, 0b1111111111111 # instrumentation
cmovz edi, dword ptr [r14 + rsi] 
adc dl, bl 
or si, 0b1000 # instrumentation
and sil, 0b11111000 # instrumentation
and dx, 0b11 # instrumentation
idiv si 
xor al, bl 
cmovnl rsi, rbx 
and rax, 0b1111111111111 # instrumentation
test byte ptr [r14 + rax], 81 
xor cl, dl 
setle bl 
cmp rcx, rsi 
and rax, 1330433608 
and rdx, 0b1111111111111 # instrumentation
cmovnb edi, dword ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], -89 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], -7 
sub dl, bl 
and rcx, 0b1111111111111 # instrumentation
cmpxchg dword ptr [r14 + rcx], ecx 
cwde  
mul dil 
and rsi, 0b1111111111000 # instrumentation
lock inc dword ptr [r14 + rsi] 
xadd cl, al 
and rdx, 0b1111111111111 # instrumentation
cmp ecx, dword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
adc di, word ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock adc word ptr [r14 + rdx], ax 
and rax, 0b1111111111111 # instrumentation
neg word ptr [r14 + rax] 
and rax, 0b1111111111000 # instrumentation
lock btc dword ptr [r14 + rax], 3 
and rax, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rax], 6 
imul cl 
.macro.switch_g2h.main.vmvictim_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vmvictim_0:
.section .data.vm
.function_vm_0:
.bb_vm_0.entry:
.macro.landing_h2g.vm_0: nop qword ptr [rax + 0xff]
.macro.measurement_start: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.vm_0: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_0:
.function_vm_1:
.bb_vm_1.entry:
.macro.landing_h2g.vm_1: nop qword ptr [rax + 0xff]
add cl, -111 # instrumentation
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
and rax, 0b1111111111111 # instrumentation
adc rax, qword ptr [r14 + rax] 
lea rdx, qword ptr [rdi] 
sub cl, -115 
lea ebx, qword ptr [rsi + rcx + 34599] 
and rax, 0b1111111111111 # instrumentation
movsx si, byte ptr [r14 + rax] 
adc dl, cl 
and rbx, 0b1111111111111 # instrumentation
xor dword ptr [r14 + rbx], 67 
and rdi, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rdi], bl 
and rdi, 0b1111111111111 # instrumentation
add byte ptr [r14 + rdi], dl 
and rax, 0b1111111111111 # instrumentation
imul di, word ptr [r14 + rax], 84 
add cl, -119 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovle ax, word ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
lock adc dword ptr [r14 + rsi], edi 
sub dl, cl 
and al, -97 
and rdi, 0b1111111111111 # instrumentation
adc bl, byte ptr [r14 + rdi] 
xchg cx, ax 
and rdx, 0b1111111111111 # instrumentation
dec qword ptr [r14 + rdx] 
and rax, -76 
lea rsi, qword ptr [rsi + rdi + 36558] 
sbb al, -39 
and rbx, 0b1111111111111 # instrumentation
dec dword ptr [r14 + rbx] 
imul edi, edx 
and rax, 0b1111111111000 # instrumentation
lock bts qword ptr [r14 + rax], 2 
or sil, -98 
xadd rcx, rdi 
or ecx, -105 
btr edx, esi 
add cl, -101 # instrumentation
cmovo eax, ecx 
bts di, 69 
cmp esi, eax 
cmp sil, 82 
add cx, cx 
cmovs edi, ebx 
bswap edi 
bt esi, 81 
sub al, 27 
and edx, edi 
test ax, -18386 
and rax, 0b1111111111111 # instrumentation
cmp word ptr [r14 + rax], 100 
and rbx, 0b1111111111111 # instrumentation
movsx rbx, word ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
test qword ptr [r14 + rsi], rax 
cmovns ecx, ebx 
and rdi, 0b1111111111111 # instrumentation
adc rdx, qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
imul byte ptr [r14 + rdx] 
lea rbx, qword ptr [rbx] 
and rcx, 0b1111111111111 # instrumentation
movzx ebx, byte ptr [r14 + rcx] 
sub al, al 
movzx si, bl 
and rdx, 0b1111111111000 # instrumentation
lock not dword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
imul di, word ptr [r14 + rcx] 
add al, -4 # instrumentation
setnl bl 
and rdx, 0b1111111111111 # instrumentation
and rax, 0b111 # instrumentation
bt qword ptr [r14 + rdx], rax 
cmc  
and rbx, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rbx] 
cmp eax, 275043641 
and rcx, 0b1111111111111 # instrumentation
xor dl, byte ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
or eax, dword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdx], -9 
and rbx, 0b1111111111000 # instrumentation
lock dec dword ptr [r14 + rbx] 
and rax, 1372799512 
add al, dl 
movzx rax, dil 
and rdi, 0b1111111111111 # instrumentation
sbb ecx, dword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
add word ptr [r14 + rbx], -50 
lfence  # noremove
.macro.measurement_end.vm_1: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.1: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_1:
.function_vm_2:
.bb_vm_2.entry:
.macro.landing_h2g.vm_2: nop qword ptr [rax + 0xff]
.macro.measurement_end.vm_2: nop qword ptr [rax + 0xff]
.macro.switch_g2h.main.2: nop qword ptr [rax + 0xff]
lfence  
.exit_vm_2:
.section .data.main
.test_case_exit:nop
