.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rax] 
lea rcx, qword ptr [rsi + rcx + 1] 
setl dil 
setl cl 
and rcx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
test rcx, rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
clc  # instrumentation
cmovnbe rcx, qword ptr [r14 + rbx] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rdi 
setz cl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
