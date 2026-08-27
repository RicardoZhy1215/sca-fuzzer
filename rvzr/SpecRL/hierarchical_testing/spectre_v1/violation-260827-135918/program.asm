.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
imul rbx, rax
dec rdi
imul rcx, rax
mov rcx, 4912
dec rax
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi
and rdx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx
test rcx, rsi
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx
sub rsi, rdx
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rsi
dec rcx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2784
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rcx
test rcx, rsi
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3320
or rdi, rbx
lea rsi, qword ptr [rsi + rsi + 1]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
and rcx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rax
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi]
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi
and rdi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx]
not rcx
imul rdx, rbx
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
