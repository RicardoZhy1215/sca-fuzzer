.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add al, 42 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdx
sub rdi, rdx
and rdx, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdx]
lea rdi, qword ptr [rbx + rdi + 1]
lea rdi, qword ptr [rdx + rdi + 1]
imul rsi, rsi
setnz dil
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rbx
neg rdi
cmp rdi, rdx
and rcx, rax
jp .bb_0.1
jmp .exit_0
.bb_0.1:
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
imul rsi, rax
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rax
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
not rdi
and rdx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdx]
and rax, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rax], rdx
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 6480
and rdi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdi], rdx
mov rdi, 7560
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
