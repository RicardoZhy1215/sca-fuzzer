.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rax, 5880 
mov rbx, rcx 
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rcx 
mov rsi, rdi 
sub rax, rdx 
and rcx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rcx] 
xor rdi, rsi 
mov rsi, 1528 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rbx 
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rbx 
mov rdi, rbx 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdi, qword ptr [r14 + rax] 
mov rdi, rax 
mov rdi, rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], rbx 
mov rax, rdi 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx 
mov rsi, rbx 
xor rax, rdi 
and rsi, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdi] 
mov rbx, 1376 
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rdi 
xor rbx, rcx 
mov rdi, 4120 
xor rdx, rbx 
sbb rdx, rbx 
sub rcx, rbx 
and rcx, 0b1111111111111 # instrumentation
cmp rdx, rdx # instrumentation
cmovz rdx, qword ptr [r14 + rcx] 
mov rax, 7224 
xor rbx, rcx 
sbb rbx, rdi 
mov rdi, 7 
and rax, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rax] 
xor rax, rdi 
mov rax, rbx 
xor rsi, rax 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6632 
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdi 
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rdx] 
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
and rax, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rax] 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rcx 
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rbx] 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
sbb rbx, rbx 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
