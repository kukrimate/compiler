// Libc function
fn extern $printf($str: *u8, ...) -> i32;

// Entry point
//fn export $main($argc: i32, $argv: **u8) -> i32 {
fn export $main() -> i32 {
    auto $i: u64;
    auto $j: u64;
    auto $t: u64;
    set $i = 1u64;
    set $j = 1u64;
@loop:
    set $t = $j;
    set $j = $i + $j;
    set $i = $t;
    eval $printf("%ld\n" u8, $i);
    jmp @loop;
}
