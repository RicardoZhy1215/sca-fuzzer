.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
setl al 
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx] 
test rdi, rsi 
and rax, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rax], rdi 
and rdx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdx], rdx 
setnz dl 
test rcx, rax 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rbx 
and rdi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdi], rax 
and rdx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rbx] 
setl dil 
setnz cl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
