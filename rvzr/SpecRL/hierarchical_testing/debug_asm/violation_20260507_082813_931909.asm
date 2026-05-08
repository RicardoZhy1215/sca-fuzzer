.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rcx, qword ptr [rbx + rcx + 1] 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx 
mov rdi, rsi 
mov rbx, 7056 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdi 
mov rsi, 1336 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rax 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rsi 
xor rsi, rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi 
lea rdi, qword ptr [rcx + rdi + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7872 
mov rcx, rbx 
mov rsi, rdx 
lea rcx, qword ptr [rdi + rcx + 1] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3544 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
