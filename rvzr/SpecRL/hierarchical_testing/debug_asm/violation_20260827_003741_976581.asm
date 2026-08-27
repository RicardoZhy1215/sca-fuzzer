.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
setb al 
or rdx, rdi 
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rdx 
setb cl 
and rdx, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdx] 
and rax, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rax] 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi 
and rsi, 0b1111111111111 # instrumentation
cmovle rdx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdx], rax 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 3536 
and rsi, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rdx] 
imul rax, rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
