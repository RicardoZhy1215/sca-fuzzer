.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
test rsi, rcx
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdx
setnz al
jmp .bb_0.1
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rcx]
pextrq rbx, xmm5, 0
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
por xmm7, xmm2
and rsi, 0b1111111111111 # instrumentation
cmovs rdx, qword ptr [r14 + rsi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
