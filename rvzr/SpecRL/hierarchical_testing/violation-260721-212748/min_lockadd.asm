.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
not rdi
and rdi, 0b1111100000000 # instrumentation
xchg qword ptr [r14 + rdi], rbx
# mem access: [14] 0x2f00-0x1ff8 cl 60:0 | [114] 0x2f00-0x1ff8 cl 60:0
sub rbx, rbx
and rbx, 0b1111111100000 # instrumentation
mov rdi, qword ptr [r14 + rbx]
and rdi, 0b1111111100000 # instrumentation
mov qword ptr [r14 + rdi], 0
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
