.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6056 
mov rbx, 3216 
xor rdi, rbx 
mov rax, 2360 
lea rdi, qword ptr [rcx + rdi + 1] 
mov rsi, 5592 
and rdi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rbx 
mov rdx, rcx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3888 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi 
xor rbx, rsi 
mov rcx, rdi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7720 
mov rcx, 2712 
lea rbx, qword ptr [rcx + rbx + 1] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5800 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4736 
lea rdx, qword ptr [rbx + rdx + 1] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 584 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rcx 
xor rcx, rcx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
xor rbx, rsi 
mov rdx, 624 
lea rbx, qword ptr [rbx + rbx + 1] 
and rax, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rax] 
mov rax, rcx 
xor rcx, rbx 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
mov rdi, 6784 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rcx 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdi 
lea rsi, qword ptr [rsi + rsi + 1] 
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi] 
mov rdx, rcx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 800 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 6848 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4552 
lea rcx, qword ptr [rdi + rcx + 1] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
