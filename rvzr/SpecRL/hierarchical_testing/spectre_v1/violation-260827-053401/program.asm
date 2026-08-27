.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax]
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rdx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rbx
inc rax
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rdx
inc rdi
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
