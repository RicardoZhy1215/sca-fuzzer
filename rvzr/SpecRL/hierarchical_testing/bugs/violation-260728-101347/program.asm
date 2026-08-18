.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
setb dil
adc rdx, rdx
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rsi
pextrq rcx, xmm5, 0
neg rsi
and rdi, rdi
test rcx, rdx
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rdx
por xmm7, xmm4
setnz dil
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
