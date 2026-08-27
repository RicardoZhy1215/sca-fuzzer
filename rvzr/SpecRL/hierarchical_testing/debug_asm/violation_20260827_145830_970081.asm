.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, rax 
mov rdi, 7072 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rax 
imul rcx, rbx 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rsi 
lea rdi, qword ptr [rax + rdi + 1] 
setl bl 
imul rdi, rsi 
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax 
mov rcx, 2216 
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rax 
mov rsi, 5216 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi 
test rdi, rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
lea rcx, qword ptr [rsi + rcx + 1] 
lea rbx, qword ptr [rsi + rbx + 1] 
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax] 
imul rdi, rdi 
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rbx] 
or rdx, rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
