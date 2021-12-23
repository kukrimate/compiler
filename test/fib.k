// Extern function references
fn extern atoi(str: *u8) -> i32;
fn extern printf(str: *u8, ...) -> i32;

fn print_fib(cnt: i32) {
    auto i = 1i32;
    auto j = 1i32;
    auto t: i32;

    while cnt > 0 {
        // Print the required number of fibonacci numbers
        eval printf("%d\n", i);
        set t = j;
        set j = i + j;
        set i = t;

        // Decrease counter, then loop if greater than zero
        set cnt = cnt - 1;
    }
}

fn export main(argc: i32, argv: *[*u8; 2]) -> i32 {
    if argc < 2 {
        eval printf("Usage: fib CNT\n");
        ret 1;
    } else {
        auto cnt: i32 = atoi((*argv)[1]);
        eval printf("Printing fib(%d)\n", cnt);
        eval print_fib(cnt);
        ret 0;
    }
}
