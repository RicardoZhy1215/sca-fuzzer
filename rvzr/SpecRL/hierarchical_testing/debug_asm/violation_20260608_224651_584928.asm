.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rdi] 
sbb rax, rdi 
and rsi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rsi], rcx 
and rcx, 0xff # instrumentation
add rcx, 1 # instrumentation
repe movsb  
xor rax, rdx 
and rbx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rbx] 
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rsi] 
lea rdx, qword ptr [rsi + rdx + 1] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], rdi 
and rbx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
