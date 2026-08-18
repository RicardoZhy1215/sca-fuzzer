.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi
and rcx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rcx]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7800
and rcx, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
cmovnl rbx, qword ptr [r14 + rsi]
inc rcx
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
inc rdi
adc rbx, rbx
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rcx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rax
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
pextrq rcx, xmm1, 0
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
mov rsi, rbx
and rbx, 0b1111111111111 # instrumentation
cmovns rbx, qword ptr [r14 + rbx]
and rdi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
