.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rcx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rcx] 
and rdx, 0b1111111111111 # instrumentation
or rdi, 1 # instrumentation
cmovnz rdi, qword ptr [r14 + rdx] 
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi 
lea rdx, qword ptr [rax + rdx + 1] 
and rsi, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rsi] 
test rdx, rsi 
and rax, rcx 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdx 
adc rbx, rdx 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rsi 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax] 
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 5128 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rsi 
and rsi, 0b1111111111111 # instrumentation
mov rcx, qword ptr [r14 + rsi] 
and rcx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rcx], rcx 
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 4064 
and rdx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdx] 
setb dl 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
