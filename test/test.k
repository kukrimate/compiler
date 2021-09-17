fn extern $printf($str: *u8, ...) -> i32;

fn export $main() {
    auto $i: i32 = 0i32;
@loop:
    eval $printf("%d\n"u8, $i);
    set $i = $i + 1i32;
    jl @loop, $i, 25i32;
}
