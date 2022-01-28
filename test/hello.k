fn extern puts(str: *u8) -> i32;

fn export main(argc: i32, argv: **u8) -> i32 {
    puts("Hello World!");
    puts("Directly embedded unicode: 😉");
    puts("Escape encoded unicode: \u1f60d");
    ret 0i32;
}
