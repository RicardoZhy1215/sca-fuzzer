.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
setnz bl 
lea rcx, qword ptr [rax + rcx + 1] 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rax 
neg rbx 
and rdx, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rdx] 
neg rax 
setb cl 
and rax, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
clc  # instrumentation
cmovnbe rbx, qword ptr [r14 + rax] 
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdi 
lea rsi, qword ptr [rax + rsi + 1] 
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rax 
or rdx, rax 
and rax, 0b1111111111111 # instrumentation
cmovnle rcx, qword ptr [r14 + rax] 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rax] 
cmp rsi, rdx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
not rcx 
and rdi, 0b1111111111111 # instrumentation
cmovle rax, qword ptr [r14 + rdi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
