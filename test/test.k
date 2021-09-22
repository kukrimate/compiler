fn extern $printf($str: *u8, ...) -> i32;

static $sarr: [u8; 3u8];
static $z: i32;

fn export $main() {
    auto $arr: [i32; 5u8] = { 0i32, 1i32, 2i32, 3i32, 4i32 };
    auto $i: i32 = 0i32;
@loop:
    eval $printf("%d\n"u8, $arr[$i]);
    jl @loop, $i, 5i32;
}
