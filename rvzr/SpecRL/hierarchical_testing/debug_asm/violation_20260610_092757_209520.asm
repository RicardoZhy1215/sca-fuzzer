.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rcx, 3144 
sub rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rbx] 
xor rbx, rcx 
sub rdi, rax 
mov rdi, rbx 
xor rbx, rcx 
xor rdi, rcx 
xor rsi, rcx 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
sub rdi, rcx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rcx 
xor rdi, rcx 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rcx 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rax] 
xor rsi, rcx 
xor rdi, rcx 
sub rbx, rcx 
xor rdi, rcx 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
sbb rdi, rsi 
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rax] 
xor rdi, rcx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
