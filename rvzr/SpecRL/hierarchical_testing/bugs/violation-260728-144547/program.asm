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
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
