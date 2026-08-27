.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
lea rbx, qword ptr [rcx + rbx + 1]
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rbx
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
dec rdi
setnz al
setz al
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdi
lea rax, qword ptr [rdx + rax + 1]
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rsi
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rsi]
setl dil
and rdx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rdx]
setnz cl
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx
imul rdx, rcx
and rdx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rdx]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rdx
setl al
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi
setnz al
and rsi, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rsi]
lea rcx, qword ptr [rsi + rcx + 1]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdi
inc rbx
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
mov rcx, 6952
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rsi
and rax, 0b1111111111111 # instrumentation
add qword ptr [r14 + rax], rsi
setnz al
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rbx
setb dil
mov rcx, 352
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rsi
and rax, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
