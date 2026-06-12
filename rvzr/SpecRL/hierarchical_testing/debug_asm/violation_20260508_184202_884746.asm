.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
xor rdi, rsi 
mov rcx, rsi 
mov rcx, 504 
mov rcx, 4416 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rdi 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
xor rax, rdx 
mov rdx, rsi 
lea rbx, qword ptr [rdi + rbx + 1] 
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rsi 
and rax, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rax] 
xor rdi, rdi 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
mov rbx, 5480 
xor rcx, rdx 
xor rdi, rdi 
lea rdx, qword ptr [rdx + rdx + 1] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
mov rax, rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 8160 
xor rdi, rsi 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx 
lea rcx, qword ptr [rcx + rcx + 1] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 3264 
xor rsi, rcx 
mov rcx, rbx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 8160 
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi] 
lea rsi, qword ptr [rcx + rsi + 1] 
xor rdx, rsi 
xor rcx, rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
