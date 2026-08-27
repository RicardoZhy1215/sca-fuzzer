.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi 
and rax, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rax], rax 
and rdx, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi 
sbb rcx, rcx 
setb bl 
and rsi, rbx 
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
test rbx, rdi 
inc rax 
and rbx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rbx] 
cmp rdi, rax 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rbx 
setb bl 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rcx 
and rdi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
