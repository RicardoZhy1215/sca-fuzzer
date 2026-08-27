.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, rbx 
lea rsi, qword ptr [rsi + rsi + 1] 
setnz sil 
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rax] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rax 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
