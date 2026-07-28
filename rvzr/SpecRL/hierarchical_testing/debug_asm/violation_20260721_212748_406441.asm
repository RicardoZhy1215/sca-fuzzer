.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
mov rcx, 6000 
and rbx, 0b1111111111111 # instrumentation
mov rsi, qword ptr [r14 + rbx] 
not rcx 
lea rax, qword ptr [rbx + rax + 1] 
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdi 
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx] 
neg rcx 
cmp rax, rax 
and rdi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdi] 
and rdx, rdi 
setnz dl 
neg rbx 
sub rbx, rdi 
not rdi 
and rbx, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rbx] 
and rcx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx] 
adc rcx, rcx 
dec rbx 
and rdx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rcx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
sub rbx, rbx 
test rdi, rdx 
and rbx, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rbx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5536 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
