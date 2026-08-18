.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lea rsi, qword ptr [rcx + rsi + 1]
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rdx, qword ptr [r14 + rcx]
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rbx
inc rdi
setz cl
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdx
xor rbx, rcx
inc rbx
pand xmm0, xmm0
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rsi
and rbx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rbx]
dec rbx
and rdx, 0b1111111111111 # instrumentation
cmovle rsi, qword ptr [r14 + rdx]
dec rcx
and rdx, 0b1111111111111 # instrumentation
or rcx, 1 # instrumentation
cmovnz rcx, qword ptr [r14 + rdx]
pextrq rbx, xmm2, 0
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rdx
sub rax, rax
and rcx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rcx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rdx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rdx], rcx
and rsi, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rsi], rax
dec rax
movq xmm5, rsi
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdx]
adc rcx, rsi
and rbx, 0b1111111111000 # instrumentation
lock xor qword ptr [r14 + rbx], rbx
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
xor rax, rbx
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 2592
setb dil
test rsi, rsi
setl al
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
