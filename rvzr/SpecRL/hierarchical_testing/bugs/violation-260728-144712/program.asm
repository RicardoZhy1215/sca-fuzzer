.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
pand xmm6, xmm0
movd xmm1, ecx
setb cl
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
pextrq rsi, xmm2, 0
dec rsi
and rdx, 0b1111111111111 # instrumentation
or rbx, 1 # instrumentation
cmovnz rbx, qword ptr [r14 + rdx]
adc rbx, rdi
and rbx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rbx]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
paddq xmm0, xmm5
neg rsi
adc rsi, rsi
sub rax, rdi
dec rdx
xor rcx, rsi
and rbx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rbx]
adc rcx, rdi
setb bl
and rsi, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rsi]
adc rbx, rax
sub rdx, rcx
and rdx, 0b1111111111111 # instrumentation
movsx rdi, byte ptr [r14 + rdx]
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rbx
adc rbx, rbx
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rax
and rsi, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rsi]
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rdx
dec rax
and rcx, 0b1111111111111 # instrumentation
mov rdx, qword ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rdx]
and rbx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rbx], rbx
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
cmp rbx, rbx
and rax, 0b1111111111000 # instrumentation
lock dec qword ptr [r14 + rax]
and rax, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rax], rbx
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rdx
pxor xmm7, xmm0
and rdi, 0b1111111111111 # instrumentation
mul qword ptr [r14 + rdi]
and rbx, 0b1111111111111 # instrumentation
cmovs rbx, qword ptr [r14 + rbx]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
