.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
cmp rdi, rdi 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdi 
adc rcx, rsi 
and rcx, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rax 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 1680 
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rax] 
setz bl 
and rsi, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
clc  # instrumentation
cmovnbe rdi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
not rcx 
test rax, rbx 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rbx 
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdx] 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rbx 
and rax, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rax] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
