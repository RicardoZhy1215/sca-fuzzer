.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rsi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rsi] 
setnz sil 
and rcx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rcx] 
setl cl 
and rdx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rdx] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rbx 
and rdx, rdi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
setl bl 
and rsi, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rbx] 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdx 
and rsi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdi], rsi 
setl cl 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi 
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rax 
mov rcx, 3256 
and rdi, rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
