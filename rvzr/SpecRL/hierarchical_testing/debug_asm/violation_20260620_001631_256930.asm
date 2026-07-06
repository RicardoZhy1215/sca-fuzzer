.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2536 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3472 
pcmpeqd xmm6, xmm0 
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2752 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1272 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2168 
sub rsi, rax 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1832 
setb sil 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1880 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4280 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 392 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7256 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx] 
psubq xmm4, xmm0 
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 864 
cmp rax, rdx 
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3968 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6928 
setnz al 
and rbx, 0b1111111111111 # instrumentation
cmovnle rdi, qword ptr [r14 + rbx] 
jmp .bb_0.1 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3472 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7032 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 264 
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx] 
paddq xmm4, xmm5 
and rdx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdx] 
setnz al 
pxor xmm5, xmm5 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5576 
setnz al 
neg rax 
dec rax 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7280 
and rbx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rbx] 
mov rax, 5520 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
