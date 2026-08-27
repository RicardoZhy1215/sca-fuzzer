.intel_syntax noprefix
.section .data.main
.function_0:
.bb_0.0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
add dl, -72 # instrumentation
not rdx
mov rcx, 5144
and rdx, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rdx], rsi
mov rcx, 5776
imul rcx, rbx
xor rcx, rcx
and rsi, 0b1111111111000 # instrumentation
lock and qword ptr [r14 + rsi], rdi
and rbx, 0b1111111111111 # instrumentation
add qword ptr [r14 + rbx], rbx
lea rbx, qword ptr [rdi + rbx + 1]
and rdx, 0b1111111111111 # instrumentation
movsx rcx, byte ptr [r14 + rdx]
or rax, 1 # instrumentation
and rdx, rax # instrumentation
shr rdx, 1 # instrumentation
div rax
and rsi, 0b1111111111111 # instrumentation
cmovbe rdx, qword ptr [r14 + rsi]
loopne .bb_0.1
jmp .exit_0
.bb_0.1:
and rsi, 0b1111111111111 # instrumentation
cmovl rdi, qword ptr [r14 + rsi]
and rdx, 0b1111111111111 # instrumentation
cmp qword ptr [r14 + rdx], rbx
test rcx, rax
test rdi, rdx
setz bl
.exit_0:
.macro.measurement_end: nop qword ptr [rax + 0xff]
jmp .test_case_exit
.section .data.main
.test_case_exit:nop
