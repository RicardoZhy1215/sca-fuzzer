.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7336 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4144 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3072 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5992 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2464 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7776 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1248 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7928 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 8080 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3072 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7160 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2080 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2080 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rax 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1088 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6960 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1472 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5432 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 6272 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rax 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5480 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rax 
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rcx] 
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 4144 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5200 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5576 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5200 
lea rsi, qword ptr [rax + rsi + 1] 
not rdi 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdi 
neg rdx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2080 
xor rdi, rcx 
inc rdx 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rcx 
inc rdx 
inc rdx 
inc rdx 
inc rdx 
lea rdx, qword ptr [rcx + rdx + 1] 
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rcx 
setz dil 
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx 
and rcx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rcx 
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rcx 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
