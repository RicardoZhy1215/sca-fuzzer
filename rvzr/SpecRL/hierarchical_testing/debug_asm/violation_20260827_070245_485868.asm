.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
setl al 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5632 
setb bl 
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi 
and rdi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rbx 
cmp rsi, rbx 
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax] 
setl dil 
and rsi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rdx] 
dec rdi 
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
