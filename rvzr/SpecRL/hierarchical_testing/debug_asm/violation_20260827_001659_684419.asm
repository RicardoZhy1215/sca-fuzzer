.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
imul rax, rsi 
xor rdi, rdx 
and rdx, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdx] 
and rdi, 0b1111111111111 # instrumentation
mov rax, qword ptr [r14 + rdi] 
and rcx, 0b1111111111111 # instrumentation
cmp rcx, rcx # instrumentation
cmovz rcx, qword ptr [r14 + rcx] 
setb dl 
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rdx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 5456 
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rbx 
test rcx, rbx 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rdx], rdi 
and rsi, 0b1111111111111 # instrumentation
movsx rax, byte ptr [r14 + rsi] 
adc rax, rbx 
setnz dil 
and rdx, 0b1111111111111 # instrumentation
cmovnle rax, qword ptr [r14 + rdx] 
adc rbx, rax 
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rbx 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
