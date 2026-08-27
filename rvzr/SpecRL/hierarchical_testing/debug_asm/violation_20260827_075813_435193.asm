.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
and rsi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rsi] 
test rdi, rsi 
and rbx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rbx], rbx 
lea rax, qword ptr [rcx + rax + 1] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 7512 
and rdi, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rsi] 
and rax, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rax] 
test rcx, rdx 
test rbx, rax 
test rdi, rdx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
or rax, 1 # instrumentation
cmovnz rax, qword ptr [r14 + rsi] 
and rbx, rsi 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
