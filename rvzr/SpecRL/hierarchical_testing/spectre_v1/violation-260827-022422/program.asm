.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdx
setnz sil
neg rdi
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rdx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdx], rax
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 2232
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rbx
lea rdi, qword ptr [rsi + rdi + 1]
setnz dl
and rax, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rax]
setb bl
adc rdi, rdx
and rdx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rbx, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
