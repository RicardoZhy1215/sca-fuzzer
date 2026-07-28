.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, -77 # instrumentation
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rbx]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rcx]
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdx]
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdi]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
imul rsi, rax
jp .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rax
imul rbx, rax
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
