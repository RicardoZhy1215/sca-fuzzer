.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
lea rdi, qword ptr [rsi + rdi + 1] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax 
mov rax, rsi 
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rax 
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rbx 
xor rdi, rsi 
and rcx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rcx 
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 808 
not rcx 
and rsi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
cmovnz rdx, qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rax, qword ptr [r14 + rdi] 
and rdi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdi] 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
