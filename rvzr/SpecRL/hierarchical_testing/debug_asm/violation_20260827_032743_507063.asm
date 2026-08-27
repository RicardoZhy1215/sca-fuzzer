.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx] 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi 
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rdi 
imul rbx, rdx 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rsi 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdx 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rax 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rsi 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rax 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi 
and rax, rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rax 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rsi 
setnz cl 
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx] 
setnz sil 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rcx 
lea rdi, qword ptr [rdi + rdi + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
