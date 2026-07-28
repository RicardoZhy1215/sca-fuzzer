.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add bl, -16 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rax]
or rdx, rbx
setb al
lea rax, qword ptr [rdi + rax + 1]
sbb rdx, rdx
sbb rsi, rdi
lea rsi, qword ptr [rdx + rsi + 1]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 3480
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rdx
imul rcx, rax
jp .bb_0.1
jmp .exit_0
.bb_0.1:
setnz al
and rdi, 0b1111111111111 # instrumentation
cmp rax, rax # instrumentation
cmovz rax, qword ptr [r14 + rdi]
setz bl
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rbx
sbb rbx, rdx
and rdx, 0b1111111111111 # instrumentation
add rdi, qword ptr [r14 + rdx]
lea rax, qword ptr [rdi + rax + 1]
setnz sil
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], rax
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 464
and rdx, 0b1111111111111 # instrumentation
cmp rdi, qword ptr [r14 + rdx]
and rbx, 0b1111111111111 # instrumentation
add rsi, qword ptr [r14 + rbx]
and rdx, 0b1111111111111 # instrumentation
cmp rbx, rbx # instrumentation
cmovz rbx, qword ptr [r14 + rdx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
