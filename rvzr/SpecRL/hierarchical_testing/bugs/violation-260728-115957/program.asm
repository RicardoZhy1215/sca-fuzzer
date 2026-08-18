.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setb bl
setb dil
cmp rbx, rcx
and rsi, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rsi]
cmp rcx, rdx
and rdx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdx]
mov rbx, 5000
mov rcx, rax
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
pxor xmm2, xmm0
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rsi
sub rbx, rcx
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax
and rcx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rcx]
pmuludq xmm4, xmm6
and rax, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rax]
por xmm7, xmm0
and rsi, rbx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx
lea rax, qword ptr [rdx + rax + 1]
pextrq rax, xmm7, 0
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rcx
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx]
setb bl
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdx
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1016
paddq xmm6, xmm7
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi
and rbx, rax
and rcx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdx
setb sil
lea rcx, qword ptr [rax + rcx + 1]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4920
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2448
and rdi, rax
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rbx
psubq xmm6, xmm0
and rdi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rdi]
inc rcx
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi]
neg rbx
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
inc rax
sub rcx, rcx
por xmm4, xmm5
pextrq rbx, xmm1, 0
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
