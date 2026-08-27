.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
and rax, 0b1111111111111 # instrumentation
movzx rax, byte ptr [r14 + rax]
and rbx, 0b1111111111000 # instrumentation
lock xadd qword ptr [r14 + rbx], rcx
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 3480
setnz bl
and rdx, 0b1111111111111 # instrumentation
cmovnle rdx, qword ptr [r14 + rdx]
and rcx, 0b1111111111000 # instrumentation
lock cmpxchg qword ptr [r14 + rcx], rdx
and rdi, rsi
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], 1696
sub rdx, rsi
and rbx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rbx], rsi
and rcx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rcx], 5216
and rbx, 0b1111111111111 # instrumentation
movzx rbx, byte ptr [r14 + rbx]
test rdx, rax
and rsi, 0b1111111111111 # instrumentation
cmovl rdx, qword ptr [r14 + rsi]
sbb rsi, rsi
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
imul rdx, rsi
imul rdx, rbx
and rcx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rcx], rax
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rcx, 0b1111111111111 # instrumentation
cmovl rcx, qword ptr [r14 + rcx]
and rax, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rax], rbx
test rdi, rsi
mov rdx, 4168
and rcx, rax
and rax, 0b1111111111111 # instrumentation
movsx rdx, byte ptr [r14 + rax]
xor rsi, rax
or rsi, rdx
and rdi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdi], rsi
setnz dl
and rcx, 0b1111111111000 # instrumentation
xchg qword ptr [r14 + rcx], rdx
and rsi, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rsi], rdi
setl cl
and rax, 0b1111111111000 # instrumentation
lock or qword ptr [r14 + rax], rdi
mov rdx, 3960
and rax, 0b1111111111111 # instrumentation
cmovnle rsi, qword ptr [r14 + rax]
and rdi, 0b1111111111111 # instrumentation
cmovbe rcx, qword ptr [r14 + rdi]
and rdx, 0b1111111111111 # instrumentation
mov qword ptr [r14 + rdx], rbx
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
