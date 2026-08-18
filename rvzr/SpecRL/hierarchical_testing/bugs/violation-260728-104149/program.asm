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
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
