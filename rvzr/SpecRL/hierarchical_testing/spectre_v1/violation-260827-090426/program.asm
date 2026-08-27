.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
setb cl
or rsi, 1 # instrumentation
and rdx, rsi # instrumentation
shr rdx, 1 # instrumentation
div rsi
test rcx, rax
and rsi, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rsi], rsi
test rcx, rcx
test rdx, rdi
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rax
test rdx, rsi
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rbx
adc rdx, rdx
and rcx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rcx], rbx
adc rcx, rax
sub rcx, rbx
and rdx, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
cmovnz rsi, qword ptr [r14 + rdx]
and rax, 0b1111111111111 # instrumentation
cmp rdi, rdi # instrumentation
cmovz rdi, qword ptr [r14 + rax]
lea rdx, qword ptr [rdi + rdx + 1]
and rdx, 0b1111111111000 # instrumentation
lock sub qword ptr [r14 + rdx], rdi
and rbx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rbx]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rbx
sbb rdi, rsi
setl dl
sbb rdx, rdx
and rsi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rsi]
and rsi, 0b1111111111111 # instrumentation
or rsi, 1 # instrumentation
clc  # instrumentation
cmovnbe rsi, qword ptr [r14 + rsi]
test rdx, rdx
and rcx, 0b1111111111111 # instrumentation
movzx rcx, byte ptr [r14 + rcx]
and rbx, rbx
setnz al
lea rcx, qword ptr [rax + rcx + 1]
and rdx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rdx], rcx
test rax, rcx
and rcx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rcx], rbx
xor rsi, rcx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rsi
and rdi, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rdi]
and rax, 0b1111111111111 # instrumentation
or rdx, 1 # instrumentation
clc  # instrumentation
cmovnbe rdx, qword ptr [r14 + rax]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rsi
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rsi
and rdi, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rdi]
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
