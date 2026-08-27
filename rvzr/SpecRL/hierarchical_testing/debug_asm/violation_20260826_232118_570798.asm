.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rsi] 
adc rax, rdx 
sbb rcx, rcx 
setnz al 
imul rdx, rax 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rdx 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rsi 
cmp rax, rbx 
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
