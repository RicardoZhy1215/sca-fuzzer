.intel_syntax noprefix
.section .data.main
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
lfence
lfence

# reduce the entropy of rax
and rax, 0b111111000000
lfence

# delay the cond. jump
mov rcx, 0
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence
lea rbx, qword ptr [rbx + rcx + 1]
lfence
lea rbx, qword ptr [rbx + rcx - 1]
lfence

# reduce the entropy in rbx
and rbx, 0b1000000
lfence

cmp rbx, 0
lfence
je .l1  # misprediction
.l0:
lfence
# rbx != 0
mov rax, qword ptr [r14 + rax]
lfence
jmp .l2
.l1:
lfence
# rbx == 0
#mov rax, qword ptr [r14 + 64]
.l2:
lfence
mfence
lfence

.section .data.main
.function_end:
.macro.measurement_end: nop qword ptr [rax + 0xff]
.section .data.main
.test_case_exit:nop
