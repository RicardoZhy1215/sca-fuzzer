.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
xor rdi, rax 
and rcx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rcx] 
and rax, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rax] 
and rbx, 0b1111111111111 # instrumentation
add rcx, qword ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rdi 
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rdx 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rax, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rsi] 
and rcx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rcx] 
and rdi, 0b1111111111111 # instrumentation
cmovl rax, qword ptr [r14 + rdi] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax 
dec rsi 
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rcx 
and rax, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rax 
and rsi, 0b1111111111111 # instrumentation
cmovns rdi, qword ptr [r14 + rsi] 
adc rcx, rdx 
and rax, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rax] 
and rdx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rdx] 
and rsi, 0b1111111111111 # instrumentation
cmovs rdi, qword ptr [r14 + rsi] 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
