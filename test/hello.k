// Libc function
fn extern $puts($str: *u8) -> i32;

// Entry point
fn export $main($argc: i32, $argv: **u8) -> i32 {
    auto $i: u8;
    eval 5u8 + 5u8 + $i;
    eval $puts("Hello World!" u8);
    ret 0i32;
}
