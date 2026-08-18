.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rsi, 0b1111111100000 # instrumentation
xchg qword ptr [r14 + rsi], rcx
lfence
# mem access: [59] 0x2a00-0x2a00 cl 40:0 | [159] 0x2a00-0x2a00 cl 40:0
xor rcx, rcx
and rcx, 0b1111111111100 # instrumentation
mul qword ptr [r14 + rcx]
add rdi, 0b1111111111100 #
and rax, 0b1111111111100 # instrumentation
mov rcx, qword ptr [r14 + rax]
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
