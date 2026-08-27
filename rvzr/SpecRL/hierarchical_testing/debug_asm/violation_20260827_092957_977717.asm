.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rdi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdi], rdi 
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi] 
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx] 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rbx 
setnz dl 
setz dil 
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx] 
and rdx, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 1392 
and rdx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdx] 
and rdx, 0b1111111111111 # instrumentation
movsx rbx, byte ptr [r14 + rdx] 
test rdx, rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
mov rcx, 888 
imul rax, rsi 
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 4160 
setb dl 
and rsi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rsi] 
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 2592 
and rcx, 0b1111111111111 # instrumentation
cmovle rcx, qword ptr [r14 + rcx] 
and rbx, 0b1111111111111 # instrumentation
cmovnl rcx, qword ptr [r14 + rbx] 
mov rdx, rsi 
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdx 
and rdi, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rdi] 
and rax, 0b1111111111111 # instrumentation
cmovns rdx, qword ptr [r14 + rax] 
and rcx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rcx] 
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rdx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
