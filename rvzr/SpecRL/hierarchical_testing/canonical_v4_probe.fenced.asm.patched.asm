.intel_syntax noprefix
.section .data.main

# the leaked value - rcx
# construct a page offset in the range [0x200; 0x900]
.function_0:
.macro.measurement_start: nop qword ptr [rax + 0xff]
and rcx, 0b11100000000
lfence
add rcx, 0x200
lfence

# save the offset into [r14 + 0]
mov qword ptr [r14], rcx
lfence
mfence
lfence

# create a delay on rbx
mov rax, 0
lfence
and rbx, 0b111000
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence
lea rbx, qword ptr [rbx + rax + 1]
lfence
lea rbx, qword ptr [rbx + rax - 1]
lfence

# sequence of potentially aliasing store-load
# if rbx == 0, they alias and rdx = 0x40
# if rbx != 0, they do not alias and rdx = offset saved above
mov qword ptr [r14 + rbx], 0x40  # store offset 0x40
lfence
mov rdx, qword ptr [r14]  # load the offset; misprediction happens here
lfence

# dependent load with the offset
and rdx, 0b111111000000
lfence
mov rdx, qword ptr [r14 + rdx]
lfence
mfence
lfence

.section .data.main
.function_end:
.macro.measurement_end: nop qword ptr [rax + 0xff]
.test_case_exit:nop
