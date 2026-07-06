.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
lfence
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111111 # instrumentation
lfence
mul qword ptr [r14 + rsi]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], rcx
lfence
adc rbx, rsi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
mov rdi, qword ptr [r14 + rsi]
lfence
not rdx
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
xchg qword ptr [r14 + rdi], rdi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
stc  # instrumentation
lfence
cmovb rbx, qword ptr [r14 + rsi]
lfence
and rbx, 0b1111111111000 # instrumentation
lfence
lock or qword ptr [r14 + rbx], rax
lfence
adc rdi, rdi
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
movsx rdx, byte ptr [r14 + rsi]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock sub qword ptr [r14 + rdi], rsi
lfence
and rax, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rax], rax
lfence
neg rdi
lfence
not rdx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], 7640
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], rsi
lfence
setb dl
lfence
and rax, 0b1111111111111 # instrumentation
lfence
movsx rdi, byte ptr [r14 + rax]
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 7928
lfence
setb bl
lfence
setz cl
lfence
jmp .bb_0.1
.bb_0.1:
lfence
or rax, rbx
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 7848
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmp rbx, qword ptr [r14 + rsi]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
cmp rdi, rdi # instrumentation
lfence
cmovz rdi, qword ptr [r14 + rsi]
lfence
and rsi, 0b1111111111111 # instrumentation
lfence
movzx rdi, byte ptr [r14 + rsi]
lfence
or rsi, 1 # instrumentation
lfence
and rdx, rsi # instrumentation
lfence
shr rdx, 1 # instrumentation
lfence
div rsi
lfence
lea rbx, qword ptr [rsi + rbx + 1]
lfence
xor rdi, rbx
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rbx], 1896
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], rax
lfence
setb dil
lfence
and rbx, 0b1111111111111 # instrumentation
lfence
add qword ptr [r14 + rbx], rsi
lfence
lea rbx, qword ptr [rsi + rbx + 1]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
xchg qword ptr [r14 + rdi], rcx
lfence
and rdx, 0b1111111111000 # instrumentation
lfence
lock dec qword ptr [r14 + rdx]
lfence
and rdi, 0b1111111111000 # instrumentation
lfence
lock add qword ptr [r14 + rdi], rsi
lfence
and rdi, 0b1111111111111 # instrumentation
lfence
mov qword ptr [r14 + rdi], 6864
lfence
.exit_0:
lfence
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
