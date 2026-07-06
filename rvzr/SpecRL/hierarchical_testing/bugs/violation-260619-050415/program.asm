.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
sub rdi, rdx
and rdx, 0b1111111111111 # instrumentation
cmovns rsi, qword ptr [r14 + rdx]
por xmm0, xmm4
pextrq rbx, xmm7, 0
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rdx
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi
pand xmm4, xmm2
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rdx
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
and rdx, 0b1111111111111 # instrumentation
cmovbe rbx, qword ptr [r14 + rdx]
lea rsi, qword ptr [rdx + rsi + 1]
neg rsi
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rsi
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rcx
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
