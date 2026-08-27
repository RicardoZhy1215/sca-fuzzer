.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rax] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rsi 
lea rbx, qword ptr [rsi + rbx + 1] 
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 1944 
imul rcx, rsi 
setnz bl 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rax 
and rdx, rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rax 
and rsi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rsi] 
imul rax, rsi 
and rbx, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
