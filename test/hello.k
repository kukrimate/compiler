// Libc function
fn extern $puts($str: *u8) -> i32;

// Entry point
fn export $main($argc: i32, $argv: **u8) -> i32 {
    eval $puts("Hello World!" u8);
    ret 0i32;
}
