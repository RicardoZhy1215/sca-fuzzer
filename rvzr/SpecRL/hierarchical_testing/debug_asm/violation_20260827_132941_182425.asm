.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rax], rcx 
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rsi 
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdi 
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rsi, qword ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7032 
and rax, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rax] 
setl al 
and rsi, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rsi] 
and rbx, rdi 
and rax, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rax] 
lea rdx, qword ptr [rsi + rdx + 1] 
and rcx, 0b1111111111111 # instrumentation
mov rbx, qword ptr [r14 + rcx] 
test rsi, rdx 
adc rcx, rdx 
and rbx, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rbx] 
imul rdx, rbx 
and rbx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rbx 
mov rdx, 7160 
setnz cl 
loopne .bb_0.1 
jmp .exit_0 
.bb_0.1:
and rdx, 0b1111111111111 # instrumentation
cmp rsi, rsi # instrumentation
cmovz rsi, qword ptr [r14 + rdx] 
dec rdi 
dec rcx 
and rax, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rax] 
mov rcx, 8000 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rdi 
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rcx 
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rbx 
not rsi 
and rbx, rbx 
and rcx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rcx], rbx 
test rcx, rbx 
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi] 
lea rcx, qword ptr [rbx + rcx + 1] 
and rdx, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rdx] 
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx] 
and rcx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rcx], rsi 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
