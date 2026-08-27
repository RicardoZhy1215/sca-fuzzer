.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, rbx
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
setnz sil
setb al
neg rbx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx
dec rcx
lea rdi, qword ptr [rsi + rdi + 1]
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rsi
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rsi
lea rdi, qword ptr [rdx + rdi + 1]
and rdx, rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rsi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
and rax, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rax]
lea rsi, qword ptr [rsi + rsi + 1]
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rcx
and rdi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdi], rdx
lea rsi, qword ptr [rsi + rsi + 1]
setnz dl
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rsi
setl cl
and rdi, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rdi]
setb sil
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
