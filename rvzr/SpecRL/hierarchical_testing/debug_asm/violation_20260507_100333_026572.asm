.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5792 
mov rax, 1376 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1296 
lea rbx, qword ptr [rsi + rbx + 1] 
mov rbx, 6640 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
mov rdx, 7952 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi 
and rsi, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 2456 
mov rcx, 7968 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
add rax, qword ptr [r14 + rdx] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3696 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
