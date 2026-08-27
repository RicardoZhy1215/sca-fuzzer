.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
mov rax, 1608
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi
and rbx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rbx]
dec rdx
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdi
setl cl
inc rdx
setl cl
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
adc rbx, rbx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rcx, rbx
adc rcx, rsi
imul rcx, rbx
not rax
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1432
lea rax, qword ptr [rsi + rax + 1]
setnz cl
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3592
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
