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
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
