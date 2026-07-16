.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add cl, 45 # instrumentation
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
lea rdi, qword ptr [rdx + rdi + 1] 
mov rdi, 2752 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
mov rdi, 5304 
mov rdi, 7048 
lea rcx, qword ptr [rax + rcx + 1] 
mov rdi, 2592 
mov rdi, 4808 
mov rax, 5600 
mov rcx, 4784 
mov rcx, 7824 
mov rax, 2920 
jnb .bb_0.1 
jmp .exit_0 
.bb_0.1:
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
lea rdi, qword ptr [rdx + rdi + 1] 
imul rdi, rcx 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rdi 
test rdi, rdx 
mov rdi, 2336 
mov rbx, 4448 
mov rcx, 1600 
mov rcx, 6088 
mov rax, 5064 
mov rcx, 5488 
setb dl 
mov rcx, 248 
mov rcx, 6016 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
