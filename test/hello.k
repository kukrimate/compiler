fn extern puts(str: *u8) -> i32;

fn export main(argc: i32, argv: **u8) -> i32 {
    // Print with a direct call
    puts("Hello World!");

    // Print using a function pointer
    let puts_ptr: *fn(*u8)->i32 = &puts;
    let puts_ptr2 = &puts; // Type deduction also works
    (*puts_ptr)("Hello World!");
    (*puts_ptr2)("Hello World!");

    ret 0i32;
}
