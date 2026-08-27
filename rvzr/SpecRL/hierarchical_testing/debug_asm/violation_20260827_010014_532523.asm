.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rbx] 
and rdi, rsi 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rcx 
or rdx, rdx 
and rdi, 0b1111111111111 # instrumentation
cmovnl rax, qword ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdi 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rax] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx] 
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx] 
dec rax 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rcx 
and rcx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdi 
mov rcx, 7104 
and rsi, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rsi] 
imul rdi, rdx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rsi 
and rcx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rcx] 
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rax 
setnz cl 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rax 
or rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
