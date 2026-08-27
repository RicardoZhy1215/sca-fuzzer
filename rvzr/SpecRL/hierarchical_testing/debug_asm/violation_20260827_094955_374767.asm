.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
test rsi, rsi 
mov rdx, 3832 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rbx 
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rsi] 
lea rbx, qword ptr [rcx + rbx + 1] 
and rdi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdi] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rsi 
inc rdi 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi 
and rdx, rsi 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdi 
and rsi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rsi] 
lea rcx, qword ptr [rbx + rcx + 1] 
setl dl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
