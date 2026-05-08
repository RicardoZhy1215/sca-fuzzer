.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdi 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 4984 
xor rsi, rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rdi 
mov rdx, rcx 
xor rdx, rsi 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 4656 
xor rsi, rdi 
xor rax, rcx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 3864 
mov rbx, 4656 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx 
mov rax, rsi 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rcx 
and rsi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 1104 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
