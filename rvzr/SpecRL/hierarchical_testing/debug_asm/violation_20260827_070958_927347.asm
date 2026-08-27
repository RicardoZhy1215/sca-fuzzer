.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rbx 
and rbx, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5776 
sub rdx, rsi 
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
cmp rdi, rdi 
or rax, rax 
mov rcx, 2904 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
test rcx, rdi 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax 
adc rcx, rdi 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
imul rcx, rax 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdx 
xor rcx, rbx 
setz dl 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax 
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdi 
inc rcx 
lea rax, qword ptr [rcx + rax + 1] 
inc rbx 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rdx 
lea rax, qword ptr [rsi + rax + 1] 
mov rdi, rax 
and rbx, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rax 
mov rax, 3920 
lea rdi, qword ptr [rsi + rdi + 1] 
setnz cl 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rcx 
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rsi 
mov rcx, 144 
and rbx, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rbx] 
test rcx, rcx 
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
