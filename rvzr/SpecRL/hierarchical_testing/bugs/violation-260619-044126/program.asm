.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
and rsi, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rbx]
pextrq rsi, xmm3, 0
pmuludq xmm7, xmm1
setz sil
setl sil
and rbx, 0b1111111111111 # instrumentation
add rdx, qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
sub rdi, rsi
or rdx, rsi
xor rbx, rcx
and rsi, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rsi]
and rcx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rcx]
and rdi, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdi]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7616
pxor xmm4, xmm3
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx]
not rdi
lea rsi, qword ptr [rsi + rsi + 1]
and rdx, 0b1111111111111 # instrumentation
movzx rsi, byte ptr [r14 + rdx]
and rcx, 0b1111111111111 # instrumentation
movzx rdx, byte ptr [r14 + rcx]
setb dil
and rbx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rsi], 7640
and rbx, 0b1111111111111 # instrumentation
cmovnl rsi, qword ptr [r14 + rbx]
adc rsi, rdx
or rdi, 1 # instrumentation
and rdx, rdi # instrumentation
shr rdx, 1 # instrumentation
div rdi
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rsi
neg rbx
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
xor rsi, rdi
and rcx, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rcx]
setz dil
dec rdi
not rbx
and rdi, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdi]
not rdi
and rsi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rsi], rsi
inc rdi
and rsi, 0b1111111111111 # instrumentation
cmovle rbx, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmovbe rdi, qword ptr [r14 + rdx]
mov rdi, 7640
or rbx, rbx
and rdi, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rdi], rbx
and rax, 0b1111111111111 # instrumentation
cmovbe rax, qword ptr [r14 + rax]
setb dl
jmp .bb_0.1
.bb_0.1:
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
