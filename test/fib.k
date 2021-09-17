// Libc function
fn extern $printf($str: *u8, ...) -> i32;

// Entry point
fn export $main() {
    auto $i: u32 = 1u32;
    auto $j: u32 = 1u32;
    auto $t: u32;
@loop:
    eval $printf("%d\n" u8, $i);
    set $t = $j;
    set $j = $i + $j;
    set $i = $t;
    jmp @loop;
}
