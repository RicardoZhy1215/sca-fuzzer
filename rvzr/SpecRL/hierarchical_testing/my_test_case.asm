.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rsi] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rcx 
adc rbx, rsi 
and rsi, 0b1111111111111 # instrumentation
mov rdi, qword ptr [r14 + rsi] 
not rdx 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rdi 
and rsi, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rsi] 
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax 
adc rdi, rdi 
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi] 
and rdi, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdi], rsi 
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], rax 
neg rdi 
not rdx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7640 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rsi 
setb dl 
and rax, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rax] 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7928 
setb bl 
setz cl 
jmp .bb_0.1 
.bb_0.1:
or rax, rbx 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 7848 
and rsi, 0b1111111111111 # instrumentation
cmp rbx, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rsi] 
and rsi, 0b1111111111111 # instrumentation
movzx rdi, byte ptr [r14 + rsi] 
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi 
lea rbx, qword ptr [rsi + rbx + 1] 
xor rdi, rbx 
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 1896 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], rax 
setb dil 
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rsi 
lea rbx, qword ptr [rsi + rbx + 1] 
and rdi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rdi], rcx 
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx] 
and rdi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rdi], rsi 
and rdi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdi], 6864 
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit 
.section .data.main
.test_case_exit:nop
