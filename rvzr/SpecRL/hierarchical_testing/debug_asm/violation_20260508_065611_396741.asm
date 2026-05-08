.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2592 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
lea rbx, qword ptr [rcx + rbx + 1] 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
xor rcx, rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1904 
lea rsi, qword ptr [rsi + rsi + 1] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 744 
mov rdx, 6792 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3120 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rdx 
xor rax, rsi 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rax 
lea rbx, qword ptr [rsi + rbx + 1] 
mov rcx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
