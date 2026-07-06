.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lfence

# reduce the entropy of rax
and rax, 0b111111000000

# delay the cond. jump using cmov instead of jne
mov rcx, 0
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]
lea rbx, qword ptr [rbx + rcx + 1]
lea rbx, qword ptr [rbx + rcx - 1]

# reduce the entropy in rbx
and rbx, 0b1000000

cmp rbx, 0
cmovne rax, qword ptr [r14 + rax]  # branch-free equivalent of je .l1

mfence

.section .data.main
.function_end:
.macro.measurement_end: nop qword ptr [rax + 0xff]
.test_case_exit:nop
