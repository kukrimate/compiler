fn extern printf(str: *u8, ...) -> i32;

record Foo {
    a: i32,
    b: i32,
};

fn export main() {
    // Initialize record
    auto foo: Foo = { 1, 2 };
    // Print fields
    eval printf("%d\n", foo.a);
    eval printf("%d\n", foo.b);

    // Initialize array, then print its contents
    auto arr: [i32; 5] = { 0, 1, 2, 3, 4 };
    auto i = 0i32;
loop:
    eval printf("%d\n", arr[i]);
    set i = i + 1;
    jl loop, i, 5;
}
