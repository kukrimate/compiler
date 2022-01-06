fn extern printf(str: *u8, ...) -> i32;

fn export main() -> i32 {
    let i = 0i32;
    while (i < 100) {
        if (i % 3 == 0 && i % 5 == 0) {
            printf("FizzBuzz\n");
        } else if (i % 3 == 0) {
            printf("Fizz\n");
        } else if (i % 5 == 0) {
            printf("Buzz\n");
        } else {
            printf("%d\n", i);
        }
        i = i + 1;
    }
    ret 0;
}
