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
and rbx, 0b1111111111111 # instrumentation
sub byte ptr [r14 + rbx], cl
and rcx, 0b1111111111111 # instrumentation
inc word ptr [r14 + rcx]
mov esi, -866389609
or ebx, 0b1000000000000000000000000000000 # instrumentation
bsf ecx, ebx
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rcx
and rcx, 0b1111111111111 # instrumentation
cmovb esi, dword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg byte ptr [r14 + rbx], al
cmovno edi, ebx
and rdx, 0b1111111111000 # instrumentation
lock sub word ptr [r14 + rdx], cx
setle cl
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], -7
xchg bl, cl
lea rbx, qword ptr [rcx]
cmovb eax, ebx
and rcx, 0b1111111111000 # instrumentation
lock or byte ptr [r14 + rcx], -82
btr rsi, rbx
and rbx, 0b1111111111111 # instrumentation
or word ptr [r14 + rbx], 1 # instrumentation
and dx, word ptr [r14 + rbx] # instrumentation
shr dx, 1 # instrumentation
div word ptr [r14 + rbx]
add al, -23 # instrumentation
lea rsi, qword ptr [rdx]
cmovnl edi, edi
or rax, 2106302
btr ebx, 131
add cl, -46 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnl rdi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rdi], 0b1000000000000000000000000000000 # instrumentation
bsr edi, dword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
add byte ptr [r14 + rcx], cl
and dl, dl
lea rax, qword ptr [rbx + rsi]
and rdx, 0b1111111111111 # instrumentation
mov ax, word ptr [r14 + rdx]
and esi, 70
and rbx, 0b1111111111111 # instrumentation
test dword ptr [r14 + rbx], -1600928048
and rbx, 0b1111111111111 # instrumentation
cmovnle ecx, dword ptr [r14 + rbx]
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], 95
or cl, 1 # instrumentation
mov ax, 1 # instrumentation
div cl
or ebx, 0b1000000000000000000000000000000 # instrumentation
bsf esi, ebx
add al, 117 # instrumentation
and rcx, 0b1111111111111 # instrumentation
cmovs si, word ptr [r14 + rcx]
bts di, 226
cmp si, di
sub cl, 47
and rdi, 0b1111111111111 # instrumentation
adc word ptr [r14 + rdi], 72
sub rdi, 64
lea ebx, qword ptr [rax + rdx]
and rax, 0b1111111111111 # instrumentation
setno byte ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
add cl, byte ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
and rcx, 0b111 # instrumentation
lock bts qword ptr [r14 + rcx], rcx
cmp eax, -1930622434
and rax, 0b1111111111111 # instrumentation
cmovl di, word ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
neg byte ptr [r14 + rsi]
dec di
and rcx, 0b1111111111000 # instrumentation
lock adc byte ptr [r14 + rcx], 46
xor sil, 105
cmovp rax, rdx
xchg edi, eax
cmovz rdi, rbx
sbb bl, sil
lea ebx, qword ptr [rsi + rbx]
and rax, 0b1111111111000 # instrumentation
lock inc word ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
or dword ptr [r14 + rsi], 0b1000 # instrumentation
and byte ptr [r14 + rsi], 0b11111000 # instrumentation
and edx, 0b11 # instrumentation
idiv dword ptr [r14 + rsi]
add bl, -33 # instrumentation
cmovns rax, rax
inc cl
and rsi, 0b1111111111111 # instrumentation
sbb byte ptr [r14 + rsi], dil
and rbx, 0b1111111111111 # instrumentation
imul dword ptr [r14 + rbx]
add al, -121 # instrumentation
and rsi, 0b1111111111111 # instrumentation
sets byte ptr [r14 + rsi]
lea cx, qword ptr [rax + rbx]
and rax, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
setnb byte ptr [r14 + rdi]
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
xor rax, rax # noremove
mov rax, qword ptr [r14 +0x2000] # noremove
mov rbx, qword ptr [r14 +0x2008] # noremove
mov rcx, qword ptr [r14 +0x2010] # noremove
mov rdx, qword ptr [r14 +0x2018] # noremove
mov rsi, qword ptr [r14 +0x2020] # noremove
mov rdi, qword ptr [r14 +0x2028] # noremove
mfence  # noremove
xor edx, ebx
setnb al
and rdx, 0b1111111111111 # instrumentation
xor qword ptr [r14 + rdx], rdx
lea esi, qword ptr [rcx + rcx + 61885]
and rcx, 0b1111111111111 # instrumentation
xor ecx, dword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rcx], 0
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rcx
nop
and rcx, 0b1111111111111 # instrumentation
xor byte ptr [r14 + rcx], sil
and rdi, 0b1111111111111 # instrumentation
mov byte ptr [r14 + rdi], al
mov ax, -10449
and rax, 0b1111111111111 # instrumentation
adc di, word ptr [r14 + rax]
mul bl
and rbx, 0b1111111111111 # instrumentation
bts dword ptr [r14 + rbx], 3
and rax, 0b1111111111111 # instrumentation
sub qword ptr [r14 + rax], rsi
cmp dl, 37
and rsi, 0b1111111111111 # instrumentation
cmovs ebx, dword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
setns byte ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
and dil, bl
and rcx, 0b1111111111111 # instrumentation
cmovnz ebx, dword ptr [r14 + rcx]
std
not si
dec rdi
and rcx, 0b1111111111111 # instrumentation
and esi, 0b111 # instrumentation
btr dword ptr [r14 + rcx], esi
add dl, -60 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnl cx, word ptr [r14 + rax]
imul edi, eax
sbb cl, -36
cmp al, cl
and rcx, 0b1111111111111 # instrumentation
cmp dword ptr [r14 + rcx], eax
cmovnz rdi, rbx
and rax, 0b1111111111111 # instrumentation
cmovle dx, word ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
test word ptr [r14 + rdx], ax
and rcx, 0b1111111111111 # instrumentation
not qword ptr [r14 + rcx]
and rsi, 0b1111111111000 # instrumentation
lock sbb qword ptr [r14 + rsi], rbx
lea rax, qword ptr [rdi + rdx + 29831]
and rax, 0b1111111111111 # instrumentation
setnz byte ptr [r14 + rax]
lea rsi, qword ptr [rdi]
and rsi, 0b1111111111111 # instrumentation
adc qword ptr [r14 + rsi], 55
and rcx, 0b1111111111111 # instrumentation
or dl, byte ptr [r14 + rcx]
and ebx, -122
setb cl
and rbx, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
btr word ptr [r14 + rbx], dx
and rsi, 0b1111111111111 # instrumentation
bts qword ptr [r14 + rsi], 4
cmp eax, ecx
and rdi, 0b1111111111111 # instrumentation
add edi, dword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
and dx, 0b111 # instrumentation
lock btc word ptr [r14 + rbx], dx
lea ax, qword ptr [rax]
and rbx, 0b1111111111111 # instrumentation
and dx, 0b111 # instrumentation
bts word ptr [r14 + rbx], dx
lea rcx, qword ptr [rax]
and rsi, 0b1111111111000 # instrumentation
lock btr word ptr [r14 + rsi], 5
setnbe bl
or cl, 111
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg word ptr [r14 + rdx], ax
test al, cl
and rbx, 0b1111111111111 # instrumentation
cmovns eax, dword ptr [r14 + rbx]
neg eax
test bl, 95
and rsi, 0b1111111111000 # instrumentation
lock or word ptr [r14 + rsi], -51
and rsi, 0b1111111111111 # instrumentation
add dword ptr [r14 + rsi], ebx
and rsi, 0b1111111111000 # instrumentation
lock neg dword ptr [r14 + rsi]
cmovnl rsi, rcx
xor di, dx
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
