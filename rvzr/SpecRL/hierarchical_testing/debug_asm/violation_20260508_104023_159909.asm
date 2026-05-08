.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rdx, 4584 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6264 
xor rsi, rsi 
lea rdi, qword ptr [rcx + rdi + 1] 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 4504 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rax 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rcx 
lea rcx, qword ptr [rdx + rcx + 1] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi 
xor rax, rbx 
and rsi, 0b1111111111111 # instrumentation
add rbx, qword ptr [r14 + rsi] 
lea rcx, qword ptr [rsi + rcx + 1] 
mov rcx, rsi 
xor rax, rbx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2360 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rcx 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5624 
xor rbx, rcx 
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rbx 
and rcx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5456 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rax 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
