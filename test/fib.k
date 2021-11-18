// Extern function references
fn extern $atoi($str: *u8) -> i32;
fn extern $printf($str: *u8, ...) -> i32;

// Entry point
fn export $main($argc: i32, $argv: *[*u8; 2u8]) {
    // Validate arguments
    jl @err, $argc, 2i32;
    auto $cnt: i32 = $atoi((*$argv)[1u8]);

    auto $i: u64 = 1u64;
    auto $j: u64 = 1u64;
    auto $t: u64;

@loop:
    // Print the required number of fibonacci numbers
    eval $printf("%d\n" u8, $i);
    set $t = $j;
    set $j = $i + $j;
    set $i = $t;

    // Decrease counter, then loop if greater than zero
    set $cnt = $cnt - 1i32;
    jg @loop, $cnt, 0i32;

    // Return success
    ret 0i32;

@err:
    // Error condition when no argument was provided
    eval $printf("Usage: fib CNT\n" u8);
    ret 1i32;
}
