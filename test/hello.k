fn extern puts(str: *u8) -> i32;

fn export main(argc: i32, argv: **u8) -> i32 {
    // Print with a direct call
    eval puts("Hello World!");
    // Print using a function pointer
    auto puts_ptr: *fn(*u8)->i32 = &puts;
    eval (*puts_ptr)("Hello World!");
    ret 0i32;
}
