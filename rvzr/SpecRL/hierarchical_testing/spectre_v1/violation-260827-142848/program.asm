.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rbx
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi]
and rax, rsi
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rax
lea rbx, qword ptr [rax + rbx + 1]
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
adc rax, rbx
imul rsi, rdi
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi
imul rcx, rdx
setnz dl
and rdi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdi]
setnz dil
imul rdi, rsi
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx]
and rsi, rdi
and rbx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rbx]
imul rdx, rbx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rbx
and rbx, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rbx]
test rcx, rdx
setnz cl
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rax
setnz cl
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2720
and rax, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rdi
mov rsi, 2952
imul rdx, rsi
mov rcx, 8152
and rax, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rbx]
adc rbx, rbx
and rbx, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
