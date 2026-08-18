.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rbx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rbx], rdi
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx]
and rsi, 0b1111111111111 # instrumentation
add qword ptr [r14 + rsi], rdx
dec rcx
and rsi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rsi]
pextrq rcx, xmm7, 0
and rcx, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rcx]
and rbx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rbx], rdi
adc rbx, rax
xor rcx, rcx
and rbx, 0b1111111111111 # instrumentation
cmovns rcx, qword ptr [r14 + rbx]
and rbx, 0b1111111111000 # instrumentation
lock add qword ptr [r14 + rbx], rcx
dec rbx
and rcx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rcx]
and rdx, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rdx], rcx
and rax, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rax]
not rcx
and rax, 0b1111111111111 # instrumentation
stc  # instrumentation
cmovb rbx, qword ptr [r14 + rax]
and rbx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rbx]
neg rcx
pextrq rbx, xmm3, 0
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rcx
inc rbx
test rsi, rcx
adc rsi, rax
cmp rdx, rsi
pextrq rsi, xmm0, 0
adc rcx, rcx
lea rcx, qword ptr [rdi + rcx + 1]
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rbx
setb dl
and rdi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rdi]
and rbx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rbx], rcx
and rbx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rbx], 7304
or rcx, 1 # instrumentation
and rdx, rcx # instrumentation
shr rdx, 1 # instrumentation
div rcx
and rax, 0b1111111111111 # instrumentation
cmovs rcx, qword ptr [r14 + rax]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
