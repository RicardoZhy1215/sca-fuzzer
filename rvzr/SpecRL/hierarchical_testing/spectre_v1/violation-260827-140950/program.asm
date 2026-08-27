.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 5096
and rax, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rax]
neg rsi
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7992
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6400
cmp rdx, rdi
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rax
and rdx, rbx
test rcx, rdi
and rax, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rax]
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rdx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
mov rax, 1232
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rsi]
dec rdx
and rbx, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rbx]
and rbx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rbx]
adc rbx, rbx
and rax, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rbx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1016
imul rcx, rdx
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi
imul rbx, rax
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rbx
and rdx, rsi
and rbx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
