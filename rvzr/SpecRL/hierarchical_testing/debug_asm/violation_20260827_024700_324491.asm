.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rsi 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rsi 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi 
not rdi 
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi] 
mov rcx, 6840 
not rax 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
not rax 
setnz al 
test rdx, rax 
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rcx 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi 
and rsi, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rsi] 
setnz dl 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdx 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx 
mov rax, 6264 
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
