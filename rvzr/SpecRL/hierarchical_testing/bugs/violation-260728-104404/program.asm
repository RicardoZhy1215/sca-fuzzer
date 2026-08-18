.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pxor xmm4, xmm2
adc rsi, rdx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
setz sil
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
pxor xmm4, xmm3
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
adc rcx, rax
and rax, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rax]
lea rcx, qword ptr [rsi + rcx + 1]
sub rcx, rsi
pextrq rbx, xmm5, 0
and rdx, rbx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax
neg rdx
mov rbx, rcx
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rax
inc rbx
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdi
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rdi]
pxor xmm1, xmm4
pextrq rcx, xmm1, 0
imul rcx, rsi
and rsi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rsi]
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rax
and rcx, rdi
inc rdi
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rcx
mov rdx, rcx
mov rcx, rax
xor rcx, rcx
dec rcx
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rdi
mov rbx, rdi
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2088
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax]
por xmm0, xmm1
pmuludq xmm4, xmm6
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rbx
and rdi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx]
por xmm4, xmm1
mov rdx, 4664
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
