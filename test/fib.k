// Extern function references
fn extern atoi(str: *u8) -> i32;
fn extern printf(str: *u8, ...) -> i32;

fn print_fib(cnt: i32) {
    let i = 1i32;
    let j = 1i32;
    let t: i32;

    while (cnt > 0) {
        // Print the required number of fibonacci numbers
        printf("%d\n", i);
        t = j;
        j = i + j;
        i = t;

        // Decrease counter, then loop if greater than zero
        cnt = cnt - 1;
    }
}

fn export main(argc: i32, argv: *[*u8; 2]) -> i32 {
    if (argc < 2) {
        printf("Usage: fib CNT\n");
        ret 1;
    } else {
        let cnt: i32 = atoi((*argv)[1]);
        printf("Printing fib(%d)\n", cnt);
        print_fib(cnt);
        ret 0;
    }
}
