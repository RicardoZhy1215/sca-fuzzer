.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rsi 
and rax, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rax] 
lea rbx, qword ptr [rbx + rbx + 1] 
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx 
lea rcx, qword ptr [rsi + rcx + 1] 
and rdx, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdx] 
and rax, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rax], rcx 
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rbx 
test rbx, rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 616 
sbb rax, rbx 
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rbx] 
setb dl 
and rdi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rdi] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rbx 
and rsi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
