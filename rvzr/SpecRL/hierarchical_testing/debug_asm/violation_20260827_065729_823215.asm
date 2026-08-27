.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax] 
setb dl 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
setl cl 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx 
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rsi] 
setb dl 
dec rdi 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rax], rbx 
setl cl 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
setb dil 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rdx 
setb dl 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
