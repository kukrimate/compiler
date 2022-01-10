fn extern printf(str: *u8, ...) -> i32;

fn foo(val: bool) -> bool {
    printf("foo\n");
    ret val;
}

fn bar(val: bool) -> bool {
    printf("bar\n");
    ret val;
}

fn export main() {
    foo(true) || bar(true); // Should print only foo
    foo(false) || bar(false); // Should print foo and bar
    foo(false) && bar(false); // Should print only foo
}
