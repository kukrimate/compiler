// Extern function references
fn extern atoi(str: *u8) -> i32;
fn extern printf(str: *u8, ...) -> i32;

// Entry point
fn export main(argc: i32, argv: *[*u8; 2]) -> i32 {
    // Validate arguments
    jl @err, argc, 2;
    auto cnt: i32 = atoi((*argv)[1]);

    auto i = 1i32;
    auto j = 1i32;
    auto t: i32;

@loop:
    // Print the required number of fibonacci numbers
    eval printf("%d\n", i);
    set t = j;
    set j = i + j;
    set i = t;

    // Decrease counter, then loop if greater than zero
    set cnt = cnt - 1;
    jg @loop, cnt, 0;

    // Return success
    ret 0;

@err:
    // Error condition when no argument was provided
    eval printf("Usage: fib CNT\n");
    ret 1;
}
