.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmp rcx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rax] 
inc rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax 
and rax, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rax] 
inc rbx 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rax 
test rdx, rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
imul rdi, rdi 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax 
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi] 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi 
and rdi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdi] 
and rcx, rdx 
and rax, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rcx] 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
