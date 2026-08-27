.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rbx
and rax, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 2168
or rsi, rax
lea rcx, qword ptr [rbx + rcx + 1]
and rdx, rdi
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx
setz dil
setnz al
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rax
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
setl bl
imul rax, rdx
setnz cl
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rdx
sub rsi, rdi
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rdx
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
setnz dl
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rbx]
and rdx, rdx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx
and rbx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rsi], rax
mov rcx, 344
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx
and rdx, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rax
adc rcx, rcx
setb dl
setl cl
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 5120
lea rcx, qword ptr [rcx + rcx + 1]
test rcx, rsi
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx
setnz cl
and rsi, rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
