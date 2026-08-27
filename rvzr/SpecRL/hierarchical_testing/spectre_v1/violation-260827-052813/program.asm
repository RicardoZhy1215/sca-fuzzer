.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx
setnz al
not rdx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rdi, rsi
and rsi, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rsi]
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rcx], rsi
and rdi, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdi]
or rdi, rax
neg rdi
and rsi, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rdx]
and rsi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rsi]
setb dil
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
