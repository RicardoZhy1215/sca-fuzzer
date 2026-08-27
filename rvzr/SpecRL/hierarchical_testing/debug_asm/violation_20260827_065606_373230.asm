.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rbx] 
and rbx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rsi 
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rsi 
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi 
mov rcx, 1488 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rdi 
and rbx, 0b1111111111111 # instrumentation
cmovnle rbx, qword ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rsi] 
mov rcx, 5832 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7512 
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rsi] 
sbb rcx, rcx 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rbx 
and rbx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
inc rax 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rax 
and rbx, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
clc  # instrumentation
cmovnbe rax, qword ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi 
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rbx 
inc rsi 
and rax, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
