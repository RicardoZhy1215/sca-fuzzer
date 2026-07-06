.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx
and rdx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
mov rsi, rax
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdx
setnz sil
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx]
and rcx, rbx
and rsi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rsi]
setnz bl
and rdx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rbx
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rcx
lea rsi, qword ptr [rdx + rsi + 1]
psubq xmm2, xmm1
and rdi, rbx
sub rsi, rbx
setnz bl
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi
pextrq rsi, xmm7, 0
jmp .bb_0.1
.bb_0.1:
pand xmm1, xmm6
not rbx
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
pxor xmm5, xmm3
not rbx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5400
cmp rbx, rdi
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rbx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rbx]
mov rdi, rcx
dec rdx
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
