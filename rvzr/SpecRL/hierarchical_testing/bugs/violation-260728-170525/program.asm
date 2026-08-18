.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rsi, rdi
mov rcx, rax
sbb rbx, rsi
pextrq rax, xmm4, 0
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi]
adc rsi, rbx
setl al
sbb rcx, rcx
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdx
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rbx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
sbb rdx, rdi
pxor xmm6, xmm5
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
