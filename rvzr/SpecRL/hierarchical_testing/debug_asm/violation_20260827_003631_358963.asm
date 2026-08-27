.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
dec rbx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rcx 
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx] 
xor rbx, rdi 
and rbx, rdi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
lea rdi, qword ptr [rsi + rdi + 1] 
or rax, rdx 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 352 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
cmovs rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rsi 
and rdi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rdi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
