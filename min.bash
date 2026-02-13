./revizor.py minimize -s base.json \
    -c ./violation/minimize.yaml -t ./violation/program.asm \
    -o ./violation/min.asm -i 10 --num-attempts 3 \
    --enable-instruction-pass 1 \
    --enable-simplification-pass 1 \
    --enable-nop-pass 1 \
    --enable-constant-pass 1 \
    --enable-mask-pass 1 \
    --enable-label-pass 1