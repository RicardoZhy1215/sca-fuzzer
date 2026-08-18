.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdi]
pxor xmm0, xmm1
pextrq rcx, xmm4, 0
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rcx
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rcx
por xmm7, xmm0
and rbx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rbx]
inc rsi
mov rdi, rdx
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rcx
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax
pand xmm7, xmm6
adc rcx, rsi
por xmm3, xmm6
por xmm3, xmm7
and rcx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rax
and rcx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rcx]
and rax, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rax]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx
and rdx, rbx
setl al
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
