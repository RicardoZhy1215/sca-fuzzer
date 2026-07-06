.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rax
and rsi, rbx
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
lea rsi, qword ptr [rdi + rsi + 1]
and rdx, 0b1111111111111 # instrumentation
cmovns rax, qword ptr [r14 + rdx]
paddq xmm4, xmm1
and rdi, 0b1111111111111 # instrumentation
movsx rsi, byte ptr [r14 + rdi]
setl dl
psubq xmm0, xmm0
and rdx, 0b1111111111111 # instrumentation
cmp rsi, qword ptr [r14 + rdx]
jmp .bb_0.1
.bb_0.1:
movq xmm1, rdi
and rsi, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rsi], rax
or rsi, rsi
setnz dl
or rbx, 1 # instrumentation
and rdx, rbx # instrumentation
shr rdx, 1 # instrumentation
div rbx
and rdx, 0b1111111111000 # instrumentation
lock inc qword ptr [r14 + rdx]
setl al
pextrq rsi, xmm7, 0
and rax, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rax], 5640
and rsi, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rsi], rsi
and rdi, 0b1111111111111 # instrumentation
cmovl rbx, qword ptr [r14 + rdi]
and rcx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rcx]
and rcx, 0b1111111111111 # instrumentation
cmovl rsi, qword ptr [r14 + rcx]
and rdx, rax
not rdx
setb sil
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
