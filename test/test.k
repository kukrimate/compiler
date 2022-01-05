fn extern printf(str: *u8, ...) -> i32;

record Foo {
    a: i32,
    b: i32,
};

static sfoo = Foo {
    a: 1,
    b: 2,
};

static sarr: [i32; 3] = [ 0, 1, 2 ];

fn export main() -> i32 {
    // Statics
    eval printf("sfoo.a = %d\n", sfoo.a);
    eval printf("sfoo.b = %d\n", sfoo.b);

    {
        auto i = 0;
        while (i < 3) {
            eval printf("sarr[%d] = %d\n", i, sarr[i]);
            set i = i + 1;
        }
    }

    // Autos
    auto foo = Foo {
        a: 1,
        b: 2,
    };

    eval printf("foo.a = %d\n", foo.a);
    eval printf("foo.b = %d\n", foo.b);

    {
        auto arr: [i32; 5] = [ 0, 1, 2, 3, 4 ];
        auto i = 0;
        while (i < 5) {
            eval printf("arr[%ld] = %d\n", i, arr[i]);
            set i = i + 1;
        }
    }

    ret 0;
}
