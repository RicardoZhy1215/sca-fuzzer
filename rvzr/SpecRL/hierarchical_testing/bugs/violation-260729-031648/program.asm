.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setnz cl
setz sil
lea rsi, qword ptr [rsi + rsi + 1]
mov rax, 4984
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax
mov rcx, rax
setz bl
setz bl
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
por xmm6, xmm4
and rdx, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rdi
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
pextrq rax, xmm3, 0
and rcx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rcx]
pcmpeqd xmm0, xmm6
and rbx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2448
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rbx
por xmm6, xmm1
pextrq rdi, xmm4, 0
movq xmm5, rcx
dec rbx
por xmm3, xmm5
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rsi
pmuludq xmm3, xmm2
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
and rax, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rax]
lea rcx, qword ptr [rcx + rcx + 1]
lea rbx, qword ptr [rax + rbx + 1]
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5832
and rsi, rbx
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rdi
pxor xmm0, xmm6
pmuludq xmm6, xmm4
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdx
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
xor rax, rcx
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
lea rbx, qword ptr [rax + rbx + 1]
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rsi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
