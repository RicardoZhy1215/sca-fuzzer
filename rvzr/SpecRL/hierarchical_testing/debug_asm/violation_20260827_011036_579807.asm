.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx] 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx 
and rdi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax 
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdx 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdx 
setnz dil 
setl sil 
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rax 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
setb cl 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
dec rdx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rsi 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rcx 
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rcx 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rax 
not rcx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rax 
and rdi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 3568 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
