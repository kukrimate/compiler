fn extern printf(str: *u8, ...) -> i32;

static sarr: [u8; 3];
static z: i32;

fn export main() {
    auto arr: [i32; 5] = { 0, 1, 2, 3, 4 };
    auto i: i32 = 0;
@loop:
    eval printf("%d\n", arr[i]);
    set i = i + 1;
    jl @loop, i, 5;
}
