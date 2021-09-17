fn extern $printf($str: *u8, ...) -> i32;

fn export $main() {
    auto $arr: [i32; 5u8] = { 0i32, 1i32, 2i32, 3i32, 4i32 };

    eval $printf("%d\n"u8, $arr[0u8]);
    eval $printf("%d\n"u8, $arr[1u8]);
    eval $printf("%d\n"u8, $arr[2u8]);
    eval $printf("%d\n"u8, $arr[3u8]);
    eval $printf("%d\n"u8, $arr[4u8]);
}
