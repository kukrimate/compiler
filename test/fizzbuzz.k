fn extern printf(str: *u8, ...) -> i32;

fn export main() -> i32 {
    auto i = 0i32;
    while i < 100 {
        if i % 3 == 0 && i % 5 == 0 {
            eval printf("FizzBuzz\n");
        } else if i % 3 == 0 {
            eval printf("Fizz\n");
        } else if i % 5 == 0 {
            eval printf("Buzz\n");
        } else {
            eval printf("%d\n", i);
        }
        set i = i + 1;
    }
    ret 0;
}
