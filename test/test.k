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
    printf("sfoo.a = %d\n", sfoo.a);
    printf("sfoo.b = %d\n", sfoo.b);

    {
        let i = 0;
        while (i < 3) {
            printf("sarr[%d] = %d\n", i, sarr[i]);
            i = i + 1;
        }
    }

    // Autos
    let foo = Foo {
        a: 1,
        b: 2,
    };

    printf("foo.a = %d\n", foo.a);
    printf("foo.b = %d\n", foo.b);

    {
        let arr: [i32; 5] = [ 0, 1, 2, 3, 4 ];
        let i = 0;
        while (i < 5) {
            printf("arr[%ld] = %d\n", i, arr[i]);
            i = i + 1;
        }
    }

    ret 0;
}
